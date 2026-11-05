"""A Gaussian target conditional on categorical state factors, scored as an expected log-density.

``C_deps={"y": ("e",)}`` with ``e`` a categorical state factor and a
:class:`~hybrid_aif.ConditionalGaussian` target holds one Gaussian per level of ``e``. Under the
utility form the objective term is the parent-belief-weighted sum over the levels of the expected
log-density of the predicted observation under that level's Gaussian. These tests pin it against
the unconditional target of each level, against a hand computation (a mixture predictive and a
neural modality), the gradient and the action it induces, the refusals, the compiled faces and the
document round trip.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Box,
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    LinearGaussianObservation,
    LinearGaussianTransition,
    MixtureGaussian,
    MLPConditional,
    System,
    SystemParams,
)
from hybrid_aif.description import DescriptionError, build, describe
from hybrid_aif.factors import lg_point
from hybrid_aif.objectives.continuous import predictive_obs

LOCS = jnp.asarray([[2.0, 0.0], [-2.0, 0.5]])  # the preferred observation under each level of e
CHOL = jnp.asarray([[[0.8, 0.0], [0.1, 0.6]], [[0.5, 0.0], [0.0, 1.0]]])
IDENTITY = jnp.log(jnp.eye(2))  # a perfectly sticky factor: the belief over e never moves


def _leaf(tree, term, key):
    iv = tree[term][key]
    np.testing.assert_allclose(np.asarray(iv.lower), np.asarray(iv.upper), atol=1e-6)
    return np.asarray(iv.lower)


def _system(on="y", conditional=True, **overrides):
    """A 2-D LG state ``x`` pushed by a Box control, a two-level sticky factor ``e`` and an LG
    modality ``y`` over ``x``; the target on ``y`` (or on ``x``) is conditional on ``e``."""
    options = dict(policy_len=2, epistemic=False)
    return System(
        A_deps=((1,),),
        B_deps=((0,), (2,), ()),
        controls=(2,),
        action_names=("a_v",),
        factor_names=("e", "x", "v"),
        modality_names=("y",),
        C_deps={on: ("e",)} if conditional else (),
        **{**options, **overrides},
    )


def _params(target, on="y", e_belief=(0.3, 0.7), likelihood=None):
    return SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=0.9 * jnp.eye(2),
                G=0.5 * jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 2)), chol=0.2 * jnp.eye(2)[None], parent_ndims=1
                ),
            ),
            "e": ConditionalCategorical(log_table=IDENTITY, parent_ndims=1),
        },
        initial_state_prior={
            "x": Gaussian(loc=jnp.asarray([0.2, -0.1]), chol=0.3 * jnp.eye(2)),
            "e": Categorical(logits=jnp.log(jnp.asarray(e_belief))),
        },
        likelihood={
            "y": likelihood
            or LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.2 * jnp.eye(2))
        },
        obs_targets={"y": target} if on == "y" else {},
        state_targets={"x": target} if on == "x" else {},
        control={"v": Box(-3.0 * jnp.ones(2), 3.0 * jnp.ones(2))},
    )


def _conditional():
    return ConditionalGaussian(loc=LOCS, chol=CHOL, parent_ndims=1)


def _level(k):
    return Gaussian(loc=LOCS[k], chol=CHOL[k])


PLAN = {"v": jnp.asarray([[0.4, -0.2], [0.1, 0.3]])}


def _scores(on, e_belief, term, key):
    system, params = _system(on), _params(_conditional(), on, e_belief)
    return _leaf(system.score_policies(params, system.initial_priors(params), PLAN), term, key)


def _flat(on, k, term, key, **kw):
    system = _system(on, conditional=False, **kw)
    params = _params(_level(k), on, (0.5, 0.5))
    return _leaf(system.score_policies(params, system.initial_priors(params), PLAN), term, key)


def _peak(chol):
    """``-log N(m; m, S)`` for ``S = chol chol^T``: the level's log-normaliser, which a
    level-weighted target drops (each level is scored relative to its peak)."""
    chol = np.asarray(chol)
    S = chol @ chol.T if chol.ndim == 2 else np.diag(chol**2)
    return 0.5 * np.linalg.slogdet(2 * np.pi * S)[1]


STEPS = 2  # the plan's length: one predictive Gaussian per step


# ------------------------------------------------------------------------- 1. the toy


@pytest.mark.parametrize("k", [0, 1])
def test_a_one_hot_parent_belief_is_the_levels_unconditional_utility(k):
    belief = (1.0, 0.0) if k == 0 else (0.0, 1.0)
    np.testing.assert_allclose(
        _scores("y", belief, "utility", "y"),
        _flat("y", k, "utility", "y") + STEPS * _peak(CHOL[k]),
        atol=1e-5,
    )


def test_a_soft_parent_belief_weights_the_levels_utilities():
    got = _scores("y", (0.3, 0.7), "utility", "y")
    want = sum(
        w * (_flat("y", k, "utility", "y") + STEPS * _peak(CHOL[k]))
        for w, k in ((0.3, 0), (0.7, 1))
    )
    np.testing.assert_allclose(got, want, atol=1e-5)
    assert not np.allclose(_flat("y", 0, "utility", "y"), _flat("y", 1, "utility", "y"))


def test_the_term_has_a_gradient_in_the_plan_and_the_planner_follows_the_weighted_target():
    system, params = _system(planner="gradient"), _params(_conditional(), e_belief=(0.8, 0.2))
    beliefs = system.initial_priors(params)

    def utility(v):
        return _leaf_sum(system.score_policies(params, beliefs, {"v": v}))

    grad = jax.grad(utility)(PLAN["v"])
    assert np.all(np.isfinite(np.asarray(grad))) and np.abs(np.asarray(grad)).max() > 1e-3
    action, _, _ = system.act(
        jax.random.PRNGKey(0), params, beliefs, selection="deterministic", mode="policy"
    )
    assert float(np.asarray(action["a_v"]).ravel()[0]) > 0.0  # the 0.8 level prefers +x
    flipped = _params(_conditional(), e_belief=(0.1, 0.9))
    action, _, _ = system.act(
        jax.random.PRNGKey(0),
        flipped,
        system.initial_priors(flipped),
        selection="deterministic",
        mode="policy",
    )
    assert float(np.asarray(action["a_v"]).ravel()[0]) < 0.0


def _leaf_sum(tree):
    return tree["utility"]["y"].lower.sum()


# -------------------------------------------------- 2. a mixture belief and a neural modality


def _mixture_system(conditional=True, **overrides):
    options = dict(policy_len=2, C_deps={"y": ("e",)} if conditional else ())
    return System(
        A_deps=((1,),),
        B_deps=((0,), (0,), (2,)),
        controlled=(False, True, False),
        mixture_k=2,
        mixture_schedule="mode_aligned",
        modality_names=("y",),
        factor_names=("m", "x", "e"),
        epistemic=False,
        **{**options, **overrides},
    )


def _mixture_params(target, likelihood, e_belief=(0.3, 0.7)):
    return SystemParams(
        likelihood={"y": likelihood},
        transition={
            "m": ConditionalCategorical(
                log_table=jnp.log(jnp.asarray([[0.8, 0.3], [0.2, 0.7]])), parent_ndims=1
            ),
            "x": LinearGaussianTransition(
                F=0.9 * jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.asarray([[-0.3, 0.0], [0.3, 0.0]])[:, None, :]
                    + jnp.asarray([-0.3, 0.0, 0.3])[None, :, None] * jnp.asarray([1.0, 0.5]),
                    chol=jnp.broadcast_to(0.3 * jnp.eye(2), (2, 3, 2, 2)),
                    parent_ndims=2,
                ),
            ),
            "e": ConditionalCategorical(log_table=IDENTITY, parent_ndims=1),
        },
        initial_state_prior={
            "m": Categorical(logits=jnp.zeros(2)),
            "x": MixtureGaussian(
                log_weights=jnp.log(jnp.asarray([0.4, 0.6])),
                means=jnp.asarray([[-0.8, 0.0], [0.9, 0.2]]),
                chols=jnp.stack([0.4 * jnp.eye(2), 0.5 * jnp.eye(2)]),
            ),
            "e": Categorical(logits=jnp.log(jnp.asarray(e_belief))),
        },
        obs_targets={"y": target},
    )


def _lg_head():
    return LinearGaussianObservation(
        H=jnp.asarray([[1.0, 0.3], [0.0, 1.0]]), c=jnp.zeros(2), noise=0.2 * jnp.eye(2)
    )


def _mixture_utility(conditional, target, likelihood, e_belief=(0.3, 0.7)):
    system = _mixture_system(conditional)
    params = _mixture_params(target, likelihood, e_belief)
    policy = jnp.asarray([[1], [2]], jnp.int32)
    tree = system.score_policies(params, system.initial_priors(params), policy)
    iv = tree["utility"]["y"]
    return np.asarray(iv.lower), np.asarray(iv.upper)


def test_a_mixture_belief_over_the_observed_factor_is_the_weighted_sum_of_the_levels():
    head = _lg_head()
    lo, hi = _mixture_utility(True, _conditional(), head)
    parts = [_mixture_utility(False, _level(k), head) for k in (0, 1)]
    for got, i in ((lo, 0), (hi, 1)):
        want = sum(w * (parts[k][i] + STEPS * _peak(CHOL[k])) for w, k in ((0.3, 0), (0.7, 1)))
        np.testing.assert_allclose(got, want, atol=1e-5)
    assert np.all(np.isfinite(lo))
    # one-hot on a level: that level's own utility
    one = _mixture_utility(True, _conditional(), head, (0.0, 1.0))
    np.testing.assert_allclose(one[0], parts[1][0] + STEPS * _peak(CHOL[1]), atol=1e-5)


def test_a_neural_modality_is_scored_at_its_linearisation_against_the_weighted_levels():
    decl = MLPConditional.from_shapes((2,), 2, key=jax.random.PRNGKey(4), width=6, depth=1)
    system = _system(policy_len=1)
    params = _params(_conditional(), likelihood=decl)
    beliefs = system.initial_priors(params)
    plan = {"v": jnp.asarray([[0.4, -0.2]])}
    got = _leaf(system.score_policies(params, beliefs, plan), "utility", "y")
    assert np.all(np.isfinite(got))
    # hand computation: the predicted state, the head linearised there, the closed form
    ctx = system.planner_context(params, beliefs)
    pred = ctx.transition_fn(ctx.init_beliefs, jnp.asarray([0.4, -0.2]))
    x = pred[1]
    mean, scale = decl._head((x.loc,))
    H = np.asarray(jax.jacfwd(lambda z: decl._head((z,))[0])(x.loc))
    cov = H @ np.asarray(x.chol @ x.chol.T) @ H.T + np.diag(np.asarray(scale) ** 2)
    want = 0.0
    for w, k in ((0.3, 0), (0.7, 1)):
        S = np.asarray(CHOL[k] @ CHOL[k].T)
        d = np.asarray(mean) - np.asarray(LOCS[k])
        ce = 0.5 * (d @ np.linalg.solve(S, d) + np.trace(np.linalg.solve(S, cov)))
        want -= w * ce
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)


# ------------------------------------------------------------- 3. a control and a state parent


def test_a_control_and_a_state_parent_index_the_action_and_average_the_belief():
    """A target over parents (control node, state factor) selects the action's level and
    weights the state's."""
    system = System(
        A_deps=((1,),),
        B_deps=((0,), ()),
        controlled=(False, True),
        policy_len=1,
        factor_names=("e", "x"),
        modality_names=("y",),
        epistemic=False,
        C_deps={"y": ("u_x", "e")},
    )
    params = _params(_conditional()).__class__(
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((3, 2)),
                    chol=jnp.broadcast_to(0.2 * jnp.eye(2), (3, 2, 2)),
                    parent_ndims=1,
                ),
            ),
            "e": ConditionalCategorical(log_table=IDENTITY, parent_ndims=1),
        },
        initial_state_prior={
            "x": Gaussian(loc=jnp.zeros(2), chol=0.3 * jnp.eye(2)),
            "e": Categorical(logits=jnp.log(jnp.asarray([0.25, 0.75]))),
        },
        likelihood={"y": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=jnp.eye(2))},
        obs_targets={
            "y": ConditionalGaussian(
                loc=jnp.stack([LOCS, LOCS + 1.0, LOCS - 1.0]),
                chol=jnp.broadcast_to(CHOL[0], (3, 2, 2, 2)),
                parent_ndims=2,
            )
        },
    )
    beliefs = system.initial_priors(params)
    got = system.score_policies(params, beliefs, jnp.asarray([[1]], jnp.int32))
    q = system.predict_beliefs(params, beliefs, jnp.asarray([[1]], jnp.int32))[0]["x"]
    pred = predictive_obs(q, lg_point(params.likelihood["y"]))
    want = -sum(
        w * float(pred.cross_entropy_to(Gaussian(loc=LOCS[k] + 1.0, chol=CHOL[0])).lower)
        for w, k in ((0.25, 0), (0.75, 1))
    ) + _peak(CHOL[0])
    np.testing.assert_allclose(_leaf(got, "utility", "y"), want, rtol=1e-4)


# ------------------------------------------------------------------------- 4. refusals


def test_a_state_parent_on_a_modality_target_is_refused_under_risk_by_name():
    system, params = _system(decomposition="risk_ambiguity"), _params(_conditional())
    with pytest.raises(
        NotImplementedError,
        match=r"obs_targets\['y'\].*state factor\(s\) \['e'\].*utility form.*takes a state parent",
    ):
        system.score_policies(params, system.initial_priors(params), PLAN)


def test_a_state_parent_that_is_not_categorical_is_refused():
    system = System(
        A_deps=((0,),),
        B_deps=((2,), (), ()),
        controls=(2,),
        action_names=("a_v",),
        factor_names=("x", "w", "v"),
        modality_names=("y",),
        policy_len=1,
        epistemic=False,
        C_deps={"y": ("w",)},
    )
    base = _params(_conditional())
    params = dataclasses.replace(
        base,
        transition={
            "x": base.transition["x"],
            "w": LinearGaussianTransition(
                F=jnp.eye(2),
                innovations=Gaussian(loc=jnp.zeros(2), chol=0.1 * jnp.eye(2)),
            ),
        },
        initial_state_prior={
            "x": base.initial_state_prior["x"],
            "w": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2)),
        },
    )
    with pytest.raises(ValueError, match=r"names the factor 'w', which is not categorical"):
        system.score_policies(params, system.initial_priors(params), PLAN)


# ----------------------------------------------------------- 5. compiled faces and the document


def test_the_compiled_faces_match_eager_and_trace_once():
    system, params = _system(planner="gradient"), _params(_conditional())
    beliefs = system.initial_priors(params)
    compiled = system.compiled()
    q_e, tree_e = system.infer_policies(params, beliefs)
    for _ in range(2):
        q_c, tree_c = compiled.infer_policies(params, beliefs, metrics=True)
    np.testing.assert_allclose(
        np.asarray(tree_c["utility"]["y"].lower),
        np.asarray(tree_e["utility"]["y"].lower),
        atol=1e-4,
        rtol=1e-3,
    )
    act_c, _, _ = compiled.act(jax.random.PRNGKey(0), params, beliefs, selection="deterministic")
    act_e, _, _ = system.act(jax.random.PRNGKey(0), params, beliefs, selection="deterministic")
    np.testing.assert_allclose(np.asarray(act_c["a_v"]), np.asarray(act_e["a_v"]), atol=1e-3)
    assert compiled.trace_counts["infer_policies"] == 1


def test_a_state_parent_round_trips_through_the_document():
    for on in ("y",):
        system, params = _system(on), _params(_conditional(), on)
        document, _ = describe(system, params, arrays="inline")
        built, built_params = build(document)
        assert built.C_deps == ((on, ("e",)),)
        target = (built_params.obs_targets if on == "y" else built_params.state_targets)[on]
        assert isinstance(target, ConditionalGaussian) and target.parent_shape == (2,)
        again, _ = describe(built, built_params, arrays="inline")
        assert again == document
        beliefs = system.initial_priors(params)
        term, key = ("utility", "y") if on == "y" else ("risk", "x")
        np.testing.assert_allclose(
            _leaf(
                built.score_policies(built_params, built.initial_priors(built_params), PLAN),
                term,
                key,
            ),
            _leaf(system.score_policies(params, beliefs, PLAN), term, key),
            atol=1e-6,
        )


def test_the_closed_loop_pass_refuses_a_state_parent_by_name():
    system = _system(planner="gradient", planning_belief="posterior", feedback="linear_quadratic")
    params = _params(_conditional())
    with pytest.raises(ValueError, match=r"obs_targets\['y'\] is conditional on a state factor"):
        system.infer_policies(params, system.initial_priors(params))


# ------------------------------------------------- 6. a diagonal preference, the low-rank leaf

LOC3 = jnp.asarray([[0.5, -0.2, 0.1], [-0.4, 0.3, 0.0]])
SD3 = jnp.asarray([[0.6, 0.9, 0.4], [1.2, 0.5, 0.8]])
EYE3 = jnp.eye(3)


def _decoder():
    return MLPConditional.from_shapes((2,), 3, key=jax.random.PRNGKey(4), width=6, depth=1)


def _diag_target(diagonal):
    chol = SD3 if diagonal else SD3[..., None] * EYE3
    return ConditionalGaussian(loc=LOC3, chol=chol, parent_ndims=1)


def _neural_leaf(target, plan, e_belief=(0.3, 0.7)):
    system = _system(policy_len=1)
    params = _params(target, e_belief=e_belief, likelihood=_decoder())
    return system, params, system.score_policies(params, system.initial_priors(params), plan)


def test_a_diagonal_target_is_declared_by_a_chol_without_its_second_axis():
    diagonal = _diag_target(True)
    assert diagonal.diagonal and not diagonal.shared_chol
    assert not _diag_target(False).diagonal
    np.testing.assert_allclose(
        np.asarray(diagonal.stacked_chol()), np.asarray(_diag_target(False).chol)
    )
    np.testing.assert_allclose(
        np.asarray(diagonal.condition_on((1,)).chol), np.asarray(SD3[1, :, None] * EYE3)
    )
    # one parent with as many levels as dimensions: (d, d) is a shared full factor
    square = ConditionalGaussian(loc=LOCS, chol=CHOL[0], parent_ndims=1)
    assert square.shared_chol and not square.diagonal
    with pytest.raises(ValueError, match="a diagonal standard deviation per parent"):
        ConditionalGaussian(loc=LOC3, chol=SD3[:, :2], parent_ndims=1)
    flat = Gaussian(loc=LOC3[0], chol=SD3[0])
    assert flat.diagonal and not Gaussian(loc=LOC3[0], chol=jnp.eye(3)).diagonal
    np.testing.assert_allclose(np.asarray(flat.dense().chol), np.asarray(jnp.diag(SD3[0])))


def test_the_low_rank_leaf_is_the_dense_leaf_on_a_gaussian_belief_and_so_is_its_gradient():
    plan = {"v": jnp.asarray([[0.4, -0.2]])}
    _, _, fast = _neural_leaf(_diag_target(True), plan)
    _, _, dense = _neural_leaf(_diag_target(False), plan)
    np.testing.assert_allclose(_leaf(fast, "utility", "y"), _leaf(dense, "utility", "y"), atol=1e-5)

    def grad(target):
        system = _system(policy_len=1)
        params = _params(target, likelihood=_decoder())
        beliefs = system.initial_priors(params)
        return jax.grad(
            lambda v: system.score_policies(params, beliefs, {"v": v})["utility"]["y"].lower
        )(plan["v"])

    np.testing.assert_allclose(
        np.asarray(grad(_diag_target(True))), np.asarray(grad(_diag_target(False))), atol=1e-4
    )


def test_an_unconditional_diagonal_target_over_a_neural_modality_is_its_dense_form():
    plan = {"v": jnp.asarray([[0.4, -0.2]])}
    diagonal = Gaussian(loc=LOC3[0], chol=SD3[0])
    dense = Gaussian(loc=LOC3[0], chol=jnp.diag(SD3[0]))
    system = _system(conditional=False, policy_len=1)
    leaves = []
    for target in (diagonal, dense):
        params = _params(target, likelihood=_decoder())
        tree = system.score_policies(params, system.initial_priors(params), plan)
        leaves.append(_leaf(tree, "utility", "y"))
    np.testing.assert_allclose(leaves[0], leaves[1], atol=1e-5)


def _mixture_leaf(target, means=None, policy_len=1):
    system = _mixture_system(policy_len=policy_len)
    params = _mixture_params(target, _decoder())
    if means is not None:
        x = params.initial_state_prior["x"]
        prior = {**params.initial_state_prior, "x": dataclasses.replace(x, means=means)}
        params = dataclasses.replace(params, initial_state_prior=prior)
    policy = jnp.asarray([[1]] * policy_len, jnp.int32)
    tree = system.score_policies(params, system.initial_priors(params), policy)
    return system, params, policy, tree["utility"]["y"]


def test_the_low_rank_leaf_on_a_mixture_belief_matches_the_dense_leaf_and_a_hand_formula():
    system, params, policy, fast = _mixture_leaf(_diag_target(True))
    _, _, _, dense = _mixture_leaf(_diag_target(False))
    np.testing.assert_allclose(np.asarray(fast.lower), np.asarray(dense.lower), atol=1e-5)
    # by hand: each predicted component read through the head at its own mean, then the closed
    # form against each level, weighted by the component and the parent belief
    decl = _decoder()
    q = system.predict_beliefs(params, system.initial_priors(params), policy)[0]
    mix = q["x"]
    want = 0.0
    for k in range(mix.num_components):
        loc, chol = mix.means[k], mix.chols[k]
        H = np.asarray(jax.jacfwd(lambda z: decl._head((z,))[0])(loc))
        mean, scale = decl._head((loc,))
        cov = H @ np.asarray(chol @ chol.T) @ H.T + np.diag(np.asarray(scale) ** 2)
        for w, level in ((0.3, 0), (0.7, 1)):
            S = np.diag(np.asarray(SD3[level]) ** 2)
            d = np.asarray(mean) - np.asarray(LOC3[level])
            ce = 0.5 * (d @ np.linalg.solve(S, d) + np.trace(np.linalg.solve(S, cov)))
            want -= float(jax.nn.softmax(mix.log_weights)[k]) * w * ce
    np.testing.assert_allclose(float(fast.lower), want, rtol=2e-3, atol=2e-3)

    def grad(target):
        def leaf(means):
            return _mixture_leaf(target, means)[3].lower

        return jax.grad(leaf)(params.initial_state_prior["x"].means)

    np.testing.assert_allclose(
        np.asarray(grad(_diag_target(True))), np.asarray(grad(_diag_target(False))), atol=1e-4
    )


def test_the_target_weights_follow_the_beliefs_batch():
    from hybrid_aif.objectives.discrete import resolve_step_targets

    loc = jnp.broadcast_to(LOC3, (4, 2, 3))
    target = ConditionalGaussian(loc=loc, chol=jnp.broadcast_to(SD3, (4, 2, 3)), parent_ndims=1)
    got = resolve_step_targets(
        {"y": target},
        {"y": (("state", 0),)},
        jnp.zeros((1,), jnp.int32),
        (Categorical(logits=jnp.log(jnp.asarray([0.3, 0.7]))),),
    )["y"]
    assert got.weights.shape == (4, 2) and got.loc.shape == (4, 2, 3)
    assert got.chol.shape == (4, 2, 3) and got.diagonal
    np.testing.assert_allclose(np.asarray(got.weights[2]), [0.3, 0.7], rtol=1e-6)


# ------------------------------------------------------ 7. the parent the modality itself reads


def test_a_parent_that_is_a_categorical_parent_of_the_modality_is_refused_by_name():
    system = System(
        A_deps=((1, 2),),
        B_deps=((0,), (0,), (2,)),
        controlled=(False, True, False),
        policy_len=1,
        mixture_k=2,
        mixture_schedule="mode_aligned",
        modality_names=("y",),
        factor_names=("m", "x", "e"),
        epistemic=False,
        C_deps={"y": ("e",)},
    )
    lik = LinearGaussianObservation(
        H=jnp.stack([jnp.eye(2), -jnp.eye(2)]),
        c=jnp.stack([3.0 * jnp.ones(2), -3.0 * jnp.ones(2)]),
        noise=0.2 * jnp.eye(2),
    )
    params = _mixture_params(_conditional(), lik)
    with pytest.raises(NotImplementedError, match=r"obs_targets\['y'\].*conditional on 'e'"):
        system.score_policies(params, system.initial_priors(params), jnp.asarray([[1]], jnp.int32))


def test_a_parent_that_is_the_mode_factor_of_the_observed_mixture_is_refused_by_name():
    system = _mixture_system(C_deps={"y": ("m",)})
    params = _mixture_params(_conditional(), _lg_head())
    with pytest.raises(NotImplementedError, match=r"conditional on 'm'.*mode"):
        system.score_policies(
            params, system.initial_priors(params), jnp.asarray([[1], [2]], jnp.int32)
        )


def test_a_neural_modalitys_own_categorical_parent_is_refused_by_name():
    system = _system(policy_len=1)
    system = dataclasses.replace(system, A_deps=((1, 0),), C_deps=(("y", ("e",)),))
    decl = MLPConditional.from_shapes((2, 2), 3, key=jax.random.PRNGKey(1), width=4, depth=1)
    params = _params(_diag_target(True), likelihood=decl)
    with pytest.raises(NotImplementedError, match=r"obs_targets\['y'\].*conditional on 'e'"):
        system.score_policies(params, system.initial_priors(params), {"v": PLAN["v"][:1]})


# ------------------------------------------------- 8. a diagonal target in the document


def test_a_diagonal_target_declares_on_a_modality_only_and_round_trips():
    system = _system(policy_len=1)
    lik = LinearGaussianObservation(
        H=jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]), c=jnp.zeros(3), noise=0.3 * jnp.eye(3)
    )
    params = _params(_diag_target(True), likelihood=lik)
    document, _ = describe(system, params, arrays="inline")
    built, built_params = build(document)
    target = built_params.obs_targets["y"]
    assert isinstance(target, ConditionalGaussian) and target.diagonal
    again, _ = describe(built, built_params, arrays="inline")
    assert again == document
    plan = {"v": PLAN["v"][:1]}
    np.testing.assert_allclose(
        _leaf(
            built.score_policies(built_params, built.initial_priors(built_params), plan),
            "utility",
            "y",
        ),
        _leaf(system.score_policies(params, system.initial_priors(params), plan), "utility", "y"),
        atol=1e-6,
    )
    flat = _system(conditional=False, policy_len=1)
    state = dataclasses.replace(
        _params(Gaussian(loc=jnp.zeros(2), chol=jnp.ones(2)), "x"), likelihood={"y": lik}
    )
    with pytest.raises(ValueError, match=r"state_targets\['x'\] is a Gaussian with a diagonal"):
        flat.score_policies(state, flat.initial_priors(state), {"v": PLAN["v"][:1]})


def test_only_a_diagonal_target_routes_through_the_low_rank_leaf(monkeypatch):
    from hybrid_aif.objectives import hybrid

    calls = []
    original = hybrid.lowrank_utility
    monkeypatch.setattr(hybrid, "lowrank_utility", lambda *a: calls.append(1) or original(*a))
    plan = {"v": jnp.asarray([[0.4, -0.2]])}
    _neural_leaf(_diag_target(False), plan)
    assert not calls
    _neural_leaf(_diag_target(True), plan)
    assert calls


# ------------------------------------------------ 9. shape rules, refusals and the cost of the grad


def test_a_shared_full_factor_that_is_not_triangular_is_refused_as_a_misdeclared_diagonal():
    with pytest.raises(ValueError, match=r"given in the full \(2, 2, 2\) form"):
        ConditionalGaussian(loc=LOCS, chol=jnp.asarray([[0.5, 0.2], [0.1, 0.3]]), parent_ndims=1)
    ConditionalGaussian(loc=LOCS, chol=jnp.asarray([[0.5, 0.0], [0.1, 0.3]]), parent_ndims=1)


def test_a_gaussian_is_diagonal_by_the_shape_of_loc_not_by_rank():
    assert not Gaussian(loc=jnp.zeros((4, 3)), chol=jnp.eye(3)).diagonal
    assert Gaussian(loc=jnp.zeros((4, 3)), chol=jnp.ones((4, 3))).diagonal


def test_cutting_a_parent_of_a_diagonal_target_to_an_ambiguous_shape_stacks_full_factors():
    from hybrid_aif.composition.merge import _contract_target

    sd = jnp.arange(1, 19, dtype=jnp.float32).reshape(2, 3, 3) * 0.1  # parents (2, 3), d = 3
    target = ConditionalGaussian(loc=jnp.zeros((2, 3, 3)), chol=sd, parent_ndims=2)
    assert target.diagonal
    cut = _contract_target(target, "y", [(0, 1, None)])
    assert not cut.shared_chol and cut.chol.shape == (3, 3, 3)
    np.testing.assert_allclose(
        np.asarray(cut.condition_on((0,)).cov), np.diag(np.asarray(sd[1, 0]) ** 2), rtol=1e-6
    )


def _with(params, **fields):
    return dataclasses.replace(params, **fields)


def test_a_diagonal_gaussian_is_refused_on_a_control_node_a_prior_and_an_innovation():
    system = _system(conditional=False, policy_len=1)
    base = _params(Gaussian(loc=jnp.zeros(2), chol=jnp.ones(2)), "y")
    plan = {"v": PLAN["v"][:1]}
    diag = Gaussian(loc=jnp.zeros(2), chol=jnp.ones(2))
    with pytest.raises(ValueError, match=r"state_targets\['v'\].*diagonal.*obs_targets"):
        bad = _with(base, state_targets={"v": diag})
        system.score_policies(bad, system.initial_priors(bad), plan)
    prior = {**base.initial_state_prior, "x": diag}
    with pytest.raises(ValueError, match=r"initial_state_prior\[1\].*diagonal.*an initial prior"):
        bad = _with(base, initial_state_prior=prior)
        system.score_policies(bad, system.initial_priors(bad), plan)
    x = base.transition["x"]
    inn = dataclasses.replace(x, innovations=diag)
    with pytest.raises(ValueError, match=r"innovations.*diagonal.*an innovation"):
        bad = _with(base, transition={**base.transition, "x": inn})
        system.score_policies(bad, system.initial_priors(bad), plan)
    with pytest.raises(DescriptionError, match="diagonal"):
        describe(system, _with(base, initial_state_prior=prior), arrays="inline")


def test_accumulated_target_refuses_a_diagonal_gaussian():
    from hybrid_aif import accumulated_target

    with pytest.raises(ValueError, match="diagonal standard deviation"):
        accumulated_target(Gaussian(loc=jnp.zeros(2), chol=jnp.ones(2)), jnp.ones(3))


def _shapes(jaxpr):
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:
            if hasattr(v.aval, "shape"):
                yield tuple(v.aval.shape)
        for param in eqn.params.values():
            for sub in param if isinstance(param, tuple | list) else (param,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from _shapes(inner)


def test_the_gradient_of_the_low_rank_leaf_forms_no_observation_by_observation_array():
    p = 37  # no layer of the decoder or other array has two axes of this size
    decl = MLPConditional.from_shapes((2,), p, key=jax.random.PRNGKey(4), width=6, depth=1)
    target = ConditionalGaussian(loc=jnp.zeros((2, p)), chol=jnp.ones((2, p)), parent_ndims=1)
    system = _system(policy_len=1)
    params = _params(target, likelihood=decl)
    beliefs = system.initial_priors(params)

    def leaf(v):
        return system.score_policies(params, beliefs, {"v": v})["utility"]["y"].lower

    jaxpr = jax.make_jaxpr(jax.grad(leaf))(PLAN["v"][:1]).jaxpr
    assert not [s for s in _shapes(jaxpr) if s[-2:] == (p, p)]
