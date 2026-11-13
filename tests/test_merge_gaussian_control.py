"""A linear-Gaussian transition that reads a discrete control node, under ``merge``/``subsystem``
and the composite driver's continuous action nodes.

Identifying the node with another one re-reads its innovation axis as a parent axis and leaves
the tensors alone; pinning a level slices the innovations' control axis (and a per-level ``F`` and
``G``); averaging over the axis is a mixture and is refused.
"""

import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    LinearGaussianObservation,
)
from hybrid_aif.composition import joint_planner, run, sample, with_mirror
from hybrid_aif.composition.merge import Link, merge, subsystem
from hybrid_aif.controls import Box
from hybrid_aif.distributions import MixtureGaussian
from hybrid_aif.factors import LinearGaussianTransition
from hybrid_aif.system import System, SystemParams

KEY = jax.random.PRNGKey(0)
OPTIONS = 2


def _cc(table, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(jnp.asarray(table, jnp.float32), 1e-12)),
        parent_ndims=parent_ndims,
    )


def mission():
    """Slow tier: a two-level leg moved by a two-level option node."""
    system = System(
        A_deps=((0,),),
        B_deps=((0, 1), ()),
        controls=(1,),
        action_names=("option",),
        state_names=("leg",),
        control_names=("u_option",),
        modality_names=("region",),
        policy_len=2,
        planner="gradient",
        num_iters=5,
        mixture_k=4,
        mixture_schedule="mode_aligned",
        planning_belief="prior",
    )
    t = np.zeros((2, 2, OPTIONS))
    t[:, :, 0] = [[0.9, 0.2], [0.1, 0.8]]
    t[:, :, 1] = [[0.3, 0.1], [0.7, 0.9]]
    params = SystemParams(
        likelihood={"region": _cc(0.8 * np.eye(2) + 0.1, 1)},
        transition={"leg": _cc(t, 2)},
        initial_state_prior={"leg": Categorical.from_probs(jnp.asarray([0.7, 0.3]))},
        obs_targets={"region": Categorical(logits=jnp.asarray([0.0, 2.0]))},
    )
    return system, params


def rover(per_level=True, with_float=True):
    """Fast tier: a 2-D x whose innovations are indexed by a terrain state and an option node,
    under a velocity command when ``with_float``."""
    controls = (2, 3) if with_float else (2,)
    states = ("terrain", "x")
    nodes = ("v", "u_opt") if with_float else ("u_opt",)
    deps = (
        ((0,), (0, 2, 3) if with_float else (0, 2), (), ())
        if with_float
        else (
            (0,),
            (0, 2),
            (),
        )
    )
    system = System(
        A_deps=((1,),),
        B_deps=deps,
        controls=controls,
        action_names=("a_v", "a_opt") if with_float else ("a_opt",),
        state_names=states,
        control_names=nodes,
        modality_names=("fix",),
        policy_len=2,
        planner="gradient",
        num_iters=5,
        mixture_k=4,
        mixture_schedule="mode_aligned",
        planning_belief="prior",
    )
    M, n_opt = 2, OPTIONS
    rng = np.random.default_rng(0)
    F_shared = np.asarray([[1.0, 0.1], [0.0, 0.9]])
    F = F_shared + 0.05 * rng.standard_normal((M, n_opt, 2, 2)) if per_level else F_shared
    G = 0.3 * rng.standard_normal((M, n_opt, 2, 2)) if per_level else 0.3 * np.ones((2, 2))
    loc = 0.1 * rng.standard_normal((M, n_opt, 2))
    chol = np.broadcast_to(np.diag([0.1, 0.2]), (M, n_opt, 2, 2)) * (
        1 + 0.3 * rng.random((M, n_opt, 1, 1))
    )
    trans = {
        "terrain": _cc(np.eye(M)[:, :, None], 2),
        "x": LinearGaussianTransition(
            F=jnp.asarray(F, jnp.float32),
            G=jnp.asarray(G, jnp.float32) if with_float else None,
            innovations=ConditionalGaussian(
                loc=jnp.asarray(loc, jnp.float32),
                chol=jnp.asarray(chol, jnp.float32),
                parent_ndims=2,
            ),
        ),
    }
    params = SystemParams(
        likelihood={
            "fix": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.3 * jnp.eye(2))
        },
        transition=trans,
        initial_state_prior={
            "terrain": Categorical.from_probs(jnp.asarray([0.5, 0.5])),
            "x": MixtureGaussian(
                log_weights=jnp.log(jnp.ones(M) / M),
                means=jnp.zeros((M, 2)),
                chols=jnp.broadcast_to(0.5 * jnp.eye(2), (M, 2, 2)),
            ),
        },
        state_targets={"x": Gaussian(loc=jnp.asarray([1.0, 1.0]), chol=0.5 * jnp.eye(2))},
        control={"v": Box(-jnp.ones(2), jnp.ones(2))} if with_float else {},
    )
    return system, params


def two_tier(**kw):
    ms, mp = mission()
    rs, rp = rover(**kw)
    return merge(
        {"mission": ms, "rover": rs},
        {"mission": mp, "rover": rp},
        [Link("mission.option", "rover.u_opt")],
    )


def pinned_standalone(level, per_level=True):
    """The rover declared at one option level: no option node, the innovations sliced there."""
    _, params = rover(per_level=per_level)
    x = params.transition["x"]
    inn = x.innovations
    new = LinearGaussianTransition(
        F=x.F[:, level] if x.F.ndim > 2 else x.F,
        G=x.G[:, level] if x.G.ndim > 2 else x.G,
        innovations=ConditionalGaussian(
            loc=inn.loc[:, level], chol=inn.chol[:, level], parent_ndims=1
        ),
    )
    system = System(
        A_deps=((1,),),
        B_deps=((0,), (0, 2), ()),
        controls=(2,),
        action_names=("a_v",),
        state_names=("terrain", "x"),
        control_names=("v",),
        modality_names=("fix",),
        policy_len=2,
        planner="gradient",
        num_iters=5,
        mixture_k=4,
        mixture_schedule="mode_aligned",
        planning_belief="prior",
    )
    return system, dataclasses.replace(params, transition={**params.transition, "x": new})


def _obs(system):
    return {"fix": jnp.asarray([0.4, -0.2])}


def test_the_two_tier_composite_still_builds():
    cs, cp = two_tier()
    inn = cp.transition["rover.x"].innovations
    assert inn.parent_shape == (2, OPTIONS)  # the identification leaves the axes whole


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("per_level", [True, False])
def test_a_pinned_view_agrees_with_the_rover_declared_at_that_level(level, per_level):
    cs, cp = two_tier(per_level=per_level)
    view, vp = subsystem(cs, cp, "rover", inputs={"mission.option": level})
    ref, rp = pinned_standalone(level, per_level=per_level)
    x = vp.transition["rover.x"]
    assert x.innovations.parent_shape == (2,)
    obs = {"rover.fix": _obs(view)["fix"]}
    qv, _ = view.infer_states(vp, obs)
    qr, _ = ref.infer_states(rp, _obs(ref))
    for a, b in zip(jax.tree.leaves(qv["rover.x"]), jax.tree.leaves(qr["x"]), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    av, _, _ = view.act(KEY, vp, qv, metrics=False)
    ar, _, _ = ref.act(KEY, rp, qr, metrics=False)
    for a, b in zip(jax.tree.leaves(av), jax.tree.leaves(ar), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_the_rover_with_its_option_clamped_predicts_as_the_pinned_view_does():
    """The standalone rover, its option node clamped to a level by the caller, predicts the
    next belief as the view pinned at that level does."""
    cs, cp = two_tier()
    view, vp = subsystem(cs, cp, "rover", inputs={"mission.option": 1})
    rs, rp = rover()
    qv, _ = view.infer_states(vp, {"rover.fix": _obs(view)["fix"]})
    q0, _ = rs.infer_states(rp, _obs(rs))
    push = jnp.asarray([0.3, -0.1])
    nv = view.update_empirical_prior(vp, qv, {"rover.a_v": push})
    nr = rs.update_empirical_prior(rp, q0, {"a_v": push, "a_opt": jnp.asarray(1)})
    for a, b in zip(jax.tree.leaves(nv["rover.x"]), jax.tree.leaves(nr["x"]), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_averaging_over_the_option_is_refused_as_a_mixture():
    cs, cp = joint_planner(*two_tier())
    with pytest.raises(NotImplementedError, match="average Gaussians across control levels"):
        subsystem(cs, cp, "rover")  # the default partner prior is uniform


def test_identifying_a_state_with_the_node_keeps_the_innovations():
    """A relay into a node a linear-Gaussian factor alone reads re-reads its axis only."""
    driver = System(
        A_deps=((0,),),
        B_deps=((0, 1), ()),
        controls=(1,),
        action_names=("pick",),
        state_names=("d",),
        control_names=("u_d",),
        modality_names=("seen",),
        policy_len=2,
        planner="gradient",
        num_iters=5,
        mixture_k=4,
        mixture_schedule="mode_aligned",
        planning_belief="prior",
    )
    dp = SystemParams(
        likelihood={"seen": _cc(np.eye(2), 1)},
        transition={"d": _cc(np.stack([np.eye(2)] * 2, -1), 2)},
        initial_state_prior={"d": Categorical.from_probs(jnp.asarray([0.5, 0.5]))},
    )
    rs, rp = rover(with_float=False)
    four = MixtureGaussian(
        log_weights=jnp.log(jnp.ones(4) / 4),
        means=jnp.zeros((4, 2)),
        chols=jnp.broadcast_to(0.5 * jnp.eye(2), (4, 2, 2)),
    )
    rp = dataclasses.replace(rp, initial_state_prior={**rp.initial_state_prior, "x": four})
    cs, cp = merge(
        {"a": driver, "b": rs},
        {"a": dp, "b": rp},
        [Link("a.d", "b.u_opt")],
    )
    before = rp.transition["x"].innovations
    after = cp.transition["b.x"].innovations
    assert after.parent_shape == before.parent_shape
    np.testing.assert_array_equal(after.loc, before.loc)
    q, _ = cs.infer_states(cp, {"a.seen": jnp.asarray(0), "b.fix": _obs(cs)["fix"]})
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(q))


# ------------------------------------------------------ the composite driver's continuous action


def float_composite():
    """One component with a single continuous control node, and a mind that emits a vector."""
    system = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("push",),
        state_names=("x",),
        control_names=("v",),
        modality_names=("fix",),
        policy_len=1,
    )
    params = SystemParams(
        likelihood={
            "fix": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.3 * jnp.eye(2))
        },
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(2),
                G=jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 2)), chol=0.1 * jnp.eye(2)[None], parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(2), chol=0.5 * jnp.eye(2))},
        control={"v": Box(-jnp.ones(2), jnp.ones(2))},
    )
    return merge({"c": system}, {"c": params}, [])


def _mind(width):
    """A step choosing ``width`` halves on the continuous action node ``model.c.push``."""

    def step(system, key, obs, carry, params):
        if obs is None:
            return {}, carry, params
        return {"model.c.push": jnp.full((width,), 0.5, jnp.float32)}, carry, params

    return step


def _x(trajectory):
    return np.asarray([c["world.c.x"] for c in trajectory["world"].carries])


@pytest.mark.parametrize("order", ["simultaneous", "sequential"])
def test_a_continuous_action_runs_against_the_mirror(order):
    """The merge coupled to its mirror, the action node continuous: recorded per round at its
    width and realised as chosen. Under ``"simultaneous"`` the run is the composite form's
    (``tests/data/migration_driver.npz``, ``float``) under its key layout, states included; under
    ``"sequential"`` the composite form stepped the component's factors at its turn with the
    realisation key, which one world leaf stepping with its own key does not reproduce, so the
    actions are compared and the fixture holds no states or observations for that order."""
    from tests.test_run_parity import _run_old_keys

    system, params = with_mirror(*float_composite())
    steps = {"model.c": _mind(2), "world": sample()}
    traj = _run_old_keys(1, system, params, 3, KEY, steps, tick_order=order, record_carries=True)
    assert traj["model.c"].intended["push"].shape == (3, 2)
    np.testing.assert_allclose(traj["model.c"].emitted["push"], 0.5)
    fixture = np.load(pathlib.Path(__file__).parent / "data" / "migration_driver.npz")
    np.testing.assert_array_equal(
        traj["model.c"].emitted["push"], fixture[f"float/{order}/realised/c.push"]
    )
    np.testing.assert_array_equal(
        traj["model.c"].intended["push"], fixture[f"float/{order}/actions/c.push"]
    )
    if order == "simultaneous":
        np.testing.assert_array_equal(_x(traj), fixture[f"float/{order}/states/c.x"])
        np.testing.assert_array_equal(
            traj["model.c"].received["fix"], fixture[f"float/{order}/observations/c.fix"]
        )


def test_a_wrong_width_continuous_action_is_refused():
    system, params = with_mirror(*float_composite())
    with pytest.raises(ValueError, match="a continuous action node takes a float vector"):
        run(system, params, 1, KEY, {"model.c": _mind(3), "world": sample()})
