"""A categorical field over a continuous state: a tabular factor whose parent is the cell a
continuous factor's belief falls in.

Pinned by: the cell mass read (one-hot, spread, mixture, differentiable); a toy rover whose terrain
is drawn from the learnable row of the cell it stands in (the transition prior, the count update,
the novelty term, the description length, the compiled faces); and the refusals.

Design: docs/design/terrain-field.md.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Grid, cell_mass
from hybrid_aif.composition.merge import merge, subsystem
from hybrid_aif.controls import Box, ControlStep
from hybrid_aif.description import build, describe
from hybrid_aif.description.mdl import description_length, tabular_family_cost
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
    Gaussian,
    MixtureGaussian,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import registry
from hybrid_aif.planning.enumerate import evaluate_policies
from hybrid_aif.system import System, SystemParams

GRID = Grid(shape=(3, 3), low=-3.0, high=3.0, axes=(0, 1))
CELLS = 9


def _gauss(mean, sd) -> Gaussian:
    return Gaussian(loc=jnp.asarray(mean, jnp.float32), chol=sd * jnp.eye(len(mean)))


# ------------------------------------------------------------------------- the cell mass read


def test_a_narrow_gaussian_inside_one_cell_is_one_hot_and_a_wide_one_spreads():
    mass = cell_mass(_gauss([-2.0, 2.0], 0.05), GRID)
    assert mass.shape == (CELLS,)
    #  row-major over (3, 3): the cell (0, 2) is entry 0 * 3 + 2
    np.testing.assert_allclose(np.asarray(mass), np.eye(CELLS)[2], atol=1e-6)
    wide = cell_mass(_gauss([0.0, 0.0], 3.0), GRID)
    assert float(wide.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(wide.min()) > 0.01 and float(wide.max()) < 0.5
    #  the tails fall into the first and last cell: a belief far outside is still one-hot
    far = cell_mass(_gauss([-40.0, 40.0], 0.1), GRID)
    np.testing.assert_allclose(np.asarray(far), np.eye(CELLS)[2], atol=1e-6)


def test_a_mixture_is_the_weight_sum_of_its_components_masses():
    mixture = MixtureGaussian(
        log_weights=jnp.log(jnp.array([0.7, 0.3])),
        means=jnp.array([[-2.0, -2.0], [2.0, 2.0]]),
        chols=jnp.broadcast_to(0.05 * jnp.eye(2), (2, 2, 2)),
    )
    mass = np.asarray(cell_mass(mixture, GRID))
    assert mass[0] == pytest.approx(0.7, abs=1e-5) and mass[8] == pytest.approx(0.3, abs=1e-5)
    assert mass.sum() == pytest.approx(1.0, abs=1e-6)


def test_the_cell_mass_is_differentiable_in_the_mean_and_jits():
    def mass_in_cell(loc):
        return cell_mass(Gaussian(loc=loc, chol=0.5 * jnp.eye(2)), GRID)[0]

    loc = jnp.array([-1.2, -1.0])
    grad = jax.grad(mass_in_cell)(loc)
    assert bool(jnp.all(grad < 0.0))  # moving toward the cell's neighbours loses mass
    np.testing.assert_allclose(
        np.asarray(jax.jit(mass_in_cell)(loc)), np.asarray(mass_in_cell(loc)), rtol=1e-6
    )
    point = jax.grad(mass_in_cell)(jnp.array([-2.0, -2.0]))  # a point belief: finite, not nan
    assert bool(jnp.all(jnp.isfinite(point)))


# ------------------------------------------------------------------------------- a toy rover


def _rover(policy_len=2, planner="gradient", epistemic=("transition:terrain",), cells=GRID):
    """A 2-D position mixture switched by three terrains and moved by a velocity control; the
    terrain is drawn from a learnable row of the cell the position falls in; a tabular modality
    reads the terrain and a Gaussian fix reads the position."""
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((1,), (0, 2), ()),
        controls=(2,),
        action_names=("a",),
        modality_names=("tex", "fix"),
        state_names=("terrain", "x"),
        control_names=("u",),
        cells={"x": cells},
        policy_len=policy_len,
        epistemic=epistemic,
        mixture_k=3,
        mixture_schedule="mode_aligned",
        planner=planner,
    )
    innovations = ConditionalGaussian(
        loc=jnp.zeros((3, 2)), chol=jnp.broadcast_to(0.05 * jnp.eye(2), (3, 2, 2)), parent_ndims=1
    )
    params = SystemParams(
        likelihood={
            "tex": ConditionalCategorical(
                log_table=jnp.log(0.9 * jnp.eye(3) + 0.05), parent_ndims=1
            ),
            "fix": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.3 * jnp.eye(2)),
        },
        transition={
            "terrain": ConditionalDirichlet(concentration=_field(), parent_ndims=1),
            "x": LinearGaussianTransition(
                F=jnp.eye(2), innovations=innovations, G=1.5 * jnp.eye(2)
            ),
        },
        initial_state_prior={
            "terrain": Categorical(logits=jnp.zeros(3)),
            "x": MixtureGaussian(
                log_weights=jnp.log(jnp.ones(3) / 3),
                means=jnp.tile(jnp.array([-2.0, -2.0]), (3, 1)),
                chols=jnp.broadcast_to(0.1 * jnp.eye(2), (3, 2, 2)),
            ),
        },
        control={"u": Box(-jnp.ones(2), jnp.ones(2))},
    )
    return system, params


def _field() -> jnp.ndarray:
    """A field whose cell 0 (the corner the rover starts in) is well visited and prefers terrain 0,
    cell 8 prefers terrain 2, the rest uniform."""
    conc = jnp.ones((3, CELLS))
    conc = conc.at[:, 0].set(jnp.array([30.0, 5.0, 5.0]))
    return conc.at[:, 8].set(jnp.array([1.0, 1.0, 6.0]))


OBS = {"tex": jnp.array(1), "fix": jnp.array([-2.0, -2.0])}
ACTION = {"a": jnp.array([0.5, 0.0])}


def _at(position, sd=0.05) -> MixtureGaussian:
    return MixtureGaussian(
        log_weights=jnp.log(jnp.ones(3) / 3),
        means=jnp.tile(jnp.asarray(position, jnp.float32), (3, 1)),
        chols=jnp.broadcast_to(sd * jnp.eye(2), (3, 2, 2)),
    )


def test_the_terrain_prior_is_the_cells_entry_of_the_field_where_the_position_was():
    system, params = _rover()
    conc = np.asarray(params.transition["terrain"].concentration)
    beliefs = {"terrain": Categorical(logits=jnp.zeros(3)), "x": _at([-2.0, -2.0])}
    #  the move carries the position to (-0.5, -0.5), cell 4; the prior reads the pre-transition
    #  cell (0), the lag of one step the design documents
    prior = system.update_empirical_prior(params, beliefs, {"a": jnp.array([1.0, 1.0])})
    np.testing.assert_allclose(
        np.asarray(prior["terrain"].probs), conc[:, 0] / conc[:, 0].sum(), atol=1e-5
    )
    assert not np.allclose(np.asarray(prior["terrain"].probs), conc[:, 4] / conc[:, 4].sum())
    assert float(prior["x"].means[0, 0]) == pytest.approx(-0.5, abs=1e-4)
    #  a position in the far corner reads that cell's entry
    prior = system.update_empirical_prior(
        params, {**beliefs, "x": _at([2.0, 2.0])}, {"a": jnp.zeros(2)}
    )
    np.testing.assert_allclose(
        np.asarray(prior["terrain"].probs), conc[:, 8] / conc[:, 8].sum(), atol=1e-5
    )


def test_learning_counts_the_terrain_into_the_cells_of_the_previous_position():
    system, params = _rover()
    before = np.asarray(params.transition["terrain"].concentration)
    previous = {"terrain": Categorical(logits=jnp.zeros(3)), "x": _at([-2.0, -2.0])}
    current, _ = system.infer_states(params, OBS, system.initial_priors(params))
    current = {**current, "x": _at([2.0, 2.0])}  # the position now sits in cell 8
    for _ in range(4):
        params = system.learn(params, OBS, current, prev_beliefs=previous, action=ACTION, lr=1.0)
    gained = np.asarray(params.transition["terrain"].concentration) - before
    assert gained[:, 0].sum() == pytest.approx(4.0, abs=1e-4)  # counted where the rover was
    assert gained[1, 0] > 3.0  # the observed terrain
    np.testing.assert_allclose(gained[:, 1:], 0.0, atol=1e-6)  # nothing at cell 8 or elsewhere


def _plan(system, params, moves):
    ctx = system.planner_context(params, system.initial_priors(params))
    step = ControlStep(entries=(jnp.asarray(moves, jnp.float32)[None],))
    tree = evaluate_policies(
        step, ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn
    )
    return float(tree["infogain"]["transition:terrain"].lower[0])


def test_novelty_is_positive_and_higher_for_a_plan_into_an_unvisited_cell():
    system, params = _rover()
    stay = _plan(system, params, [[0.0, 0.0], [0.0, 0.0]])
    move = _plan(system, params, [[1.0, 0.0], [1.0, 0.0]])  # leaves the visited corner cell
    assert stay > 0.0
    assert move > stay
    #  the epistemic selector gates it: without the field's variable there is no leaf
    off, off_params = _rover(epistemic=())
    ctx = off.planner_context(off_params, off.initial_priors(off_params))
    step = ControlStep(entries=(jnp.zeros((1, 2, 2)),))
    tree = evaluate_policies(
        step, ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn
    )
    assert "infogain" not in tree or "transition:terrain" not in tree["infogain"]


def test_the_planner_runs_through_the_field():
    system, params = _rover()
    beliefs, _ = system.infer_states(params, OBS, system.initial_priors(params))
    q, tree = system.infer_policies(params, beliefs, metrics=True)
    assert "transition:terrain" in tree["infogain"]


def test_description_length_prices_the_field_as_one_table_row_per_cell():
    system, params = _rover()
    dl = description_length(system, 100.0, params=params)
    record = dl.per_address["transition:terrain"]
    assert record.family == pytest.approx(CELLS * tabular_family_cost(3, 100.0))
    assert record.prior is not None and "Dirichlet" in record.note
    #  and the system round-trips through its description document
    doc = describe(system, params)[0]
    assert doc["variables"]["x"]["cells"]["shape"] == [3, 3]
    rebuilt = build(doc)[0]
    assert dict(rebuilt.cells) == dict(system.cells)


def test_the_compiled_faces_trace_once_with_a_field_and_match_the_eager_faces():
    system, params = _rover()
    compiled = system.compiled()
    priors_c = priors_e = system.initial_priors(params)
    params_c = params_e = params
    prev_c = prev_e = None
    for tick in range(3):
        key = jax.random.PRNGKey(tick)
        beliefs_c, _ = compiled.infer_states(params_c, OBS, priors_c)
        beliefs_e, _ = system.infer_states(params_e, OBS, priors_e)
        action_c, _, _ = compiled.act(key, params_c, beliefs_c, metrics=False)
        action_e, _, _ = system.act(key, params_e, beliefs_e, metrics=False)
        compiled.infer_policies(params_c, beliefs_c, metrics=False)
        if prev_c is not None:
            params_c = compiled.learn(
                params_c, OBS, beliefs_c, prev_beliefs=prev_c[0], action=prev_c[1], lr=0.5
            )
            params_e = system.learn(
                params_e, OBS, beliefs_e, prev_beliefs=prev_e[0], action=prev_e[1], lr=0.5
            )
        priors_c = compiled.update_empirical_prior(params_c, beliefs_c, action_c)
        priors_e = system.update_empirical_prior(params_e, beliefs_e, action_c)
        prev_c, prev_e = (beliefs_c, action_c), (beliefs_e, action_c)
    assert compiled.trace_counts == {
        "infer_states": 1,
        "act": 1,
        "infer_policies": 1,
        "update_empirical_prior": 1,
        "learn": 1,
    }
    for c, e in zip(
        jax.tree_util.tree_leaves((params_c, priors_c)),
        jax.tree_util.tree_leaves((params_e, priors_e)),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(c), np.asarray(e), atol=2e-4, rtol=1e-3)


# ---------------------------------------------------------------------------------- refusals


def test_a_tabular_factor_naming_a_continuous_parent_needs_a_grid():
    system, params = _rover()
    system = dataclasses.replace(system, cells={})
    with pytest.raises(ValueError, match=r"factor 'terrain'.*continuous factor 'x'.*cells="):
        system.infer_states(params, OBS, system.initial_priors(params))


def test_a_grid_declared_for_a_categorical_factor_is_refused():
    system, params = _rover()
    system = dataclasses.replace(system, cells={"x": GRID, "terrain": Grid(shape=3, low=0, high=1)})
    with pytest.raises(ValueError, match="grid for 'terrain', which is a categorical factor"):
        system.infer_states(params, OBS, system.initial_priors(params))


def test_a_table_whose_cell_axis_has_the_wrong_length_is_refused_with_the_cell_count():
    system, params = _rover()
    bad = dataclasses.replace(
        params,
        transition={
            **params.transition,
            "terrain": ConditionalDirichlet(concentration=jnp.ones((3, 8)), parent_ndims=1),
        },
    )
    with pytest.raises(ValueError, match=r"has 9 cells \(shape \(3, 3\)\)"):
        system.infer_states(bad, OBS, system.initial_priors(bad))


def test_a_prior_parent_that_is_a_continuous_factor_is_refused_as_non_categorical():
    #  x is declared first, so the ordering rule passes and the family rule is what fires
    system = System(
        B_deps=((), (0,)),
        state_names=("x", "terrain"),
        D_deps=((), (0,)),
        cells={"x": Grid(shape=3, low=-3.0, high=3.0, axes=(0,))},
        policy_len=1,
    )
    params = SystemParams(
        likelihood={},
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=0.1 * jnp.ones((1, 1, 1)), parent_ndims=1
                ),
            ),
            "terrain": ConditionalDirichlet(concentration=jnp.ones((3, 3)), parent_ndims=1),
        },
        initial_state_prior={
            "x": _gauss([0.0], 0.1),
            "terrain": ConditionalCategorical(log_table=jnp.zeros((3, 3)), parent_ndims=1),
        },
    )
    with pytest.raises(ValueError, match="D_deps for factor 'terrain' names non-categorical"):
        system._validate(params)


def test_the_forward_process_refuses_a_system_with_cells():
    system, params = _rover()
    with pytest.raises(NotImplementedError, match="as_env on a System with cells"):
        system.as_env(params)


def test_a_grid_on_an_all_tabular_system_is_refused_by_every_face():
    system = System(
        A_deps=((0,),),
        B_deps=((0,), (0,)),
        state_names=("s", "t"),
        cells={"s": Grid(shape=4, low=0.0, high=1.0)},
        policy_len=1,
    )
    params = SystemParams(
        likelihood={
            "modality_0": ConditionalCategorical(
                log_table=jnp.log(0.9 * jnp.eye(3) + 0.033), parent_ndims=1
            )
        },
        transition={
            "s": ConditionalCategorical(log_table=jnp.zeros((3, 3)), parent_ndims=1),
            "t": ConditionalCategorical(log_table=jnp.zeros((2, 3)), parent_ndims=1),
        },
        initial_state_prior={
            "s": Categorical(logits=jnp.zeros(3)),
            "t": Categorical(logits=jnp.zeros(2)),
        },
    )
    priors = system.initial_priors(params)
    sentence = "which is a categorical factor"
    with pytest.raises(ValueError, match=sentence):
        system.infer_states(params, {"modality_0": jnp.array(1)}, priors)
    with pytest.raises(ValueError, match=sentence):
        system.update_empirical_prior(params, priors, {})
    with pytest.raises(ValueError, match=sentence):
        system.as_env(params)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(shape=np.int64(3), low=0, high=1),
        dict(shape=(3, 3), low=np.array([0.0, 0.0]), high=1.0),
        dict(shape=(3, 3), low=jnp.array([0.0, 0.0]), high=jnp.asarray(1.0)),
        dict(shape=[3], low=[0], high=[1], axes=0),
        dict(shape=(3, 3), low=0, high=1, axes=np.array([1, 0])),
    ],
)
def test_a_grid_takes_numpy_and_jax_numbers_and_stays_hashable(kwargs):
    grid = Grid(**kwargs)
    assert hash(grid) == hash(Grid(**kwargs))
    assert all(type(n) is int for n in grid.shape) and all(type(v) is float for v in grid.low)
    assert Grid(shape=(3,), low=0, high=1) == Grid(shape=3, low=0.0, high=1.0, axes=(0,))


def test_cells_accepts_none_and_refuses_a_repeated_name():
    system, params = _rover()
    assert dataclasses.replace(system, cells=None).cells == ()
    with pytest.raises(ValueError, match=r"cells names \['x'\] twice"):
        dataclasses.replace(system, cells=(("x", GRID), ("x", GRID)))


def test_an_amortized_system_refuses_cells():
    from hybrid_aif.reference.raif.world import tiny_rssm

    system, params = tiny_rssm(2, 1, key=jax.random.PRNGKey(0), policy_len=2)
    system = dataclasses.replace(system, cells={system.node_names[0]: GRID})
    with pytest.raises(NotImplementedError, match="cells on an amortized"):
        system._validate(params)


def test_a_registered_estimator_reading_the_previous_step_gets_the_belief_not_the_cells():
    seen = []

    def prepare(st):
        def step(ctx):
            seen.append(type(ctx.qs_prev[1]).__name__)
            return {"cell_prev_probe": Interval.exact(jnp.zeros(()))}

        return step

    if "cell_prev_probe" not in registry.ROWS:
        registry.register_row(
            registry.Estimator(
                "cell_prev_probe", registry.SINGLE, prepare, term="cell_prev_probe", reads_prev=True
            )
        )
    system, params = _rover()
    system = dataclasses.replace(system, terms=("cell_prev_probe",))
    ctx = system.planner_context(params, system.initial_priors(params))
    step = ControlStep(entries=(jnp.zeros((1, 2, 2)),))
    tree = evaluate_policies(
        step, ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn
    )
    assert set(seen) == {"MixtureGaussian"}
    assert (
        tree["infogain"]["transition:terrain"].lower[0] > 0.0
    )  # the novelty leaf still reads cells


def test_a_merged_field_and_its_subsystem_view_keep_the_grid_and_plan():
    tab = System(
        A_deps=((0,),),
        B_deps=((),),
        state_names=("s",),
        modality_names=("o",),
        policy_len=2,
        planner="gradient",
        mixture_k=3,
        mixture_schedule="mode_aligned",
    )
    tab_params = SystemParams(
        likelihood={
            "o": ConditionalCategorical(log_table=jnp.log(0.8 * jnp.eye(2) + 0.1), parent_ndims=1)
        },
        transition={"s": ConditionalCategorical(log_table=jnp.zeros((2, 1)), parent_ndims=1)},
        initial_state_prior={"s": Categorical(logits=jnp.zeros(2))},
    )
    system, params = _rover()
    composite, cparams = merge({"rover": system, "tab": tab}, {"rover": params, "tab": tab_params})
    assert dict(composite.cells) == {"rover.x": GRID}
    obs = {"rover.tex": OBS["tex"], "rover.fix": OBS["fix"], "tab.o": jnp.array(0)}
    beliefs, _ = composite.infer_states(cparams, obs, composite.initial_priors(cparams))
    q, tree = composite.infer_policies(cparams, beliefs, metrics=True)
    assert "transition:rover.terrain" in tree["infogain"]
    view, vparams = subsystem(composite, cparams, "rover")
    assert dict(view.cells) == {"rover.x": GRID}
    vbeliefs, _ = view.infer_states(
        vparams,
        {k: v for k, v in obs.items() if k.startswith("rover.")},
        view.initial_priors(vparams),
    )
    view.infer_policies(vparams, vbeliefs, metrics=False)


def test_the_description_refusals_say_what_to_declare():
    from hybrid_aif.description import DescriptionError

    system, params = _rover()
    doc, _ = describe(system, params)
    bare = json.loads(json.dumps(doc))
    del bare["variables"]["x"]["cells"]
    with pytest.raises(
        DescriptionError, match="give 'x' a cells grid to read the cell it falls in"
    ):
        build(bare)
    wrong = json.loads(json.dumps(doc))
    wrong["variables"]["terrain"]["cells"] = doc["variables"]["x"]["cells"]
    with pytest.raises(DescriptionError, match="cells discretises a real latent"):
        build(wrong)
