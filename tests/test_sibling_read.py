"""The sibling-belief read (design: nesting-and-coupling.md N3): a modality of one component
observes an emission of another, the source's belief through a kernel, realised; ``merge`` carries
the pair, ``subsystem`` leaves the modality an observation, and ``run`` supplies it each round."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    Gaussian,
    Grid,
    LinearGaussianObservation,
    LinearGaussianTransition,
    MixtureGaussian,
    Parameter,
    System,
    SystemParams,
    build,
    cell_mass,
    describe,
)
from hybrid_aif.composition import (
    Identify,
    Link,
    MindCarry,
    agent_step,
    merge,
    nested,
    run,
    subsystem,
)
from hybrid_aif.distributions import ConditionalGaussian
from hybrid_aif.reads import Emission
from hybrid_aif.system.emissions import identity, indicator

GRID = Grid(shape=(2, 2), low=0.0, high=2.0)
CELLS = GRID.n_cells
READ = "mission.cell_obs"
SOURCE = "rover.x"
EMITTED = "rover.cell_obs"  # the emission the belief read declares on the source's component


def _rover(
    shared: bool = False,
    loc=(1.0, 1.0),
    chol0: float = 0.6,
    innovation: float = 0.4,
    noise: float = 0.3,
):
    """A position factor ``x`` (a 2-D random walk, observed with noise); ``shared`` adds a
    categorical ``map`` factor reading the parameter variable ``field``."""
    names = ("x", "map") if shared else ("x",)
    system = System(
        A_deps=((0,),),
        B_deps=((),) * len(names),
        factor_names=names,
        modality_names=("obs",),
        epistemic=False,
        policy_len=1,
    )
    transition = {
        "x": LinearGaussianTransition(
            F=jnp.eye(2),
            innovations=ConditionalGaussian(
                loc=jnp.zeros((1, 2)),
                chol=jnp.broadcast_to(innovation * jnp.eye(2), (1, 2, 2)),
                parent_ndims=1,
            ),
        ),
    }
    initial = {"x": Gaussian(loc=jnp.asarray(loc), chol=chol0 * jnp.eye(2))}
    params = dict(
        likelihood={
            "obs": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=noise * jnp.eye(2))
        },
    )
    if shared:
        transition["map"] = Parameter("field")
        initial["map"] = Categorical(logits=jnp.zeros(2))
        params["parameters"] = {"field": _field()}
    return system, SystemParams(transition=transition, initial_state_prior=initial, **params)


def _field():
    return ConditionalDirichlet(concentration=jnp.ones((2, 1)), parent_ndims=1)


def _scheduler(shared: bool = False):
    """A categorical ``visit`` factor over the grid's cells, observed through ``cell_obs``."""
    names = ("visit", "map") if shared else ("visit",)
    system = System(
        A_deps=((0,),),
        B_deps=((),) * len(names),
        factor_names=names,
        modality_names=("cell_obs",),
        epistemic=False,
        policy_len=1,
    )
    table = jnp.log(jnp.eye(CELLS) * 0.85 + 0.05)
    transition = {"visit": ConditionalCategorical(jnp.log(jnp.full((CELLS, 1), 1 / CELLS)), 1)}
    initial = {"visit": Categorical(logits=jnp.zeros(CELLS))}
    extra = {}
    if shared:
        transition["map"] = Parameter("field")
        initial["map"] = Categorical(logits=jnp.zeros(2))
        extra["parameters"] = {"field": _field()}
    return system, SystemParams(
        likelihood={"cell_obs": ConditionalCategorical(table, 1)},
        transition=transition,
        initial_state_prior=initial,
        **extra,
    )


def _composite(
    kernel=GRID, realise="mode", shared: bool = False, bindings=(), rover: dict | None = None
):
    (sr, pr), (ss, ps) = _rover(shared, **(rover or {})), _scheduler(shared)
    return merge(
        {"rover": sr, "mission": ss},
        {"rover": pr, "mission": ps},
        [Link(SOURCE, READ, realise=realise, kernel=kernel), *bindings],
    )


def _recorded(step, log, mind: str):
    def wrapped(key, obs, carry, params):
        log.append((mind, dict(obs), carry))
        return step(key, obs, carry, params)

    return wrapped


def _posterior(view, entry, mind_prefix: str = "rover"):
    """The posterior ``view`` held at a step: its logged ``(obs, carry in)`` filtered again."""
    obs, carry = entry
    return view[0].infer_states(view[1], obs, priors=carry.beliefs)[0]


def _run(composite, params, ticks: int, key=0):
    """Run the two perceivers; the log holds ``(mind, obs seen, carry in)`` per step. Returns the
    trajectory, one record per round (the ``obs`` the mission saw, both carries in, and the
    rover's posterior at the round) and the mission's view."""
    log: list = []
    rover = subsystem(composite, params, "rover")
    mission = subsystem(composite, params, "mission")
    minds = {
        "rover": _recorded(agent_step(rover[0]), log, "rover"),
        "mission": _recorded(agent_step(mission[0]), log, "mission"),
    }
    trajectory = run(composite, params, minds, ticks, jax.random.PRNGKey(key))
    rounds = []
    for t in range(ticks):
        entries = {m: (o, c) for m, o, c in log[2 * t : 2 * t + 2]}
        rounds.append(
            {
                "obs": entries["mission"][0],
                "rover": entries["rover"][1],
                "mission": entries["mission"][1],
                "posterior": _posterior(rover, entries["rover"])[SOURCE],
            }
        )
    return trajectory, rounds, mission


def _mass(round_):
    """The cell mass of the rover's posterior at ``round_``."""
    return np.asarray(cell_mass(round_["posterior"], GRID))


# ------------------------------------------------------------------ the three realisations


def test_mode_datum_is_the_argmax_of_the_cell_mass_of_the_carried_belief():
    comp, params = _composite(realise="mode")
    _, rounds, _ = _run(comp, params, 8)
    assert rounds[0]["obs"][READ] is None  # the source has no carry yet: absent
    for before, r in zip(rounds[:-1], rounds[1:], strict=True):
        datum = r["obs"][READ]
        assert datum.dtype == jnp.int32
        assert int(datum) == int(np.argmax(_mass(before)))  # the posterior of the round before


def test_sample_frequency_matches_the_cell_mass_and_mode_would_not():
    """A rover held at a cell edge (an uninformative observation, a tiny innovation) puts about
    half its mass on each of two cells: a draw's frequency matches the mass within the binomial
    standard error, and the same statistic under ``mode`` misses the bound by about twice its
    width at this seed and length."""
    hold = dict(loc=(1.0, 0.5), chol0=0.05, innovation=1e-3, noise=1e3)
    comp, params = _composite(realise="sample", rover=hold)
    ticks = 600
    _, rounds, _ = _run(comp, params, ticks)
    masses = np.array([_mass(r) for r in rounds[:-1]])
    mean_mass = masses.mean(axis=0)
    assert abs(mean_mass[0] - 0.5) < 0.05 and abs(mean_mass[2] - 0.5) < 0.05
    drawn = np.array([int(r["obs"][READ]) for r in rounds[1:]])
    freq = np.bincount(drawn, minlength=CELLS) / (ticks - 1)
    tol = 4 * np.sqrt(0.25 / (ticks - 1))  # four binomial standard errors at p = 1/2
    assert np.abs(freq - mean_mass).max() < tol
    mode_freq = np.bincount(masses.argmax(axis=1), minlength=CELLS) / (ticks - 1)
    assert np.abs(mode_freq - mean_mass).max() > tol


def test_many_draws_of_a_fixed_categorical_belief_have_its_frequencies():
    belief = Categorical.from_probs(jnp.asarray([0.2, 0.5, 0.3]))
    n = 4000
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    drawn = jax.vmap(lambda k: identity(belief, "sample", k, "e"))(keys)
    freq = np.bincount(np.asarray(drawn), minlength=3) / n
    tol = 4 * np.sqrt(0.25 / n)
    np.testing.assert_allclose(freq, [0.2, 0.5, 0.3], atol=tol)
    mode = identity(belief, "mode", None, "e")
    assert int(mode) == 1 and abs(1.0 - 0.5) > tol


def test_the_soft_observation_is_within_the_jensen_bound_of_the_exact_treatment():
    """The note's acceptance: the log expected likelihood less the expected log-likelihood (the
    filter's soft observation) is non-negative, zero for a delta, and at most the log ratio of the
    largest to the smallest likelihood over the datum's support."""
    table = np.asarray(jax.nn.softmax(jnp.log(jnp.eye(CELLS) * 0.85 + 0.05), axis=0), np.float64)
    table = table / table.sum(axis=0)
    belief = Gaussian(loc=jnp.asarray([1.0, 0.7]), chol=0.4 * jnp.eye(2))
    mass = np.asarray(indicator(belief, GRID, "stats", None, "e"), np.float64)
    support = mass > 1e-12
    gap = np.log(mass @ table) - mass @ np.log(table)  # per hidden state, over the datum's cells
    bound = np.log(table[support].max(axis=0) / table[support].min(axis=0))
    assert (gap >= -1e-9).all() and (gap <= bound + 1e-9).all() and gap.max() > 1e-3
    delta = np.eye(CELLS)[1]
    np.testing.assert_allclose(np.log(delta @ table) - delta @ np.log(table), 0.0, atol=1e-9)


def test_sample_draws_are_keyed_by_the_round_and_reproducible():
    comp, params = _composite(realise="sample")
    _, a, _ = _run(comp, params, 12, key=3)
    _, b, _ = _run(comp, params, 12, key=3)
    assert [r["obs"][READ] is None or int(r["obs"][READ]) for r in a] == [
        r["obs"][READ] is None or int(r["obs"][READ]) for r in b
    ]


def test_stats_datum_is_the_cell_mass_and_the_posterior_is_the_soft_observation_one():
    comp, params = _composite(realise="stats")
    trajectory, rounds, mission = _run(comp, params, 6)
    view, view_params = mission
    log_a = jax.nn.log_softmax(view_params.likelihood[READ].log_table, axis=0)
    for before, r in zip(rounds[:-1], rounds[1:], strict=True):
        datum = r["obs"][READ]
        np.testing.assert_allclose(np.asarray(datum), _mass(before), rtol=1e-5, atol=1e-7)
        prior = r["mission"].beliefs["mission.visit"]
        # the filter's soft observation: the expected log-likelihood under the vector
        want = jax.nn.softmax(prior.logits + datum @ log_a)
        got, _ = view.infer_states(view_params, {READ: datum}, priors=r["mission"].beliefs)
        np.testing.assert_allclose(
            np.asarray(got["mission.visit"].probs), np.asarray(want), rtol=1e-5, atol=1e-7
        )
    # the mind's own carry is what the recorded observations give, round by round
    step, carry = agent_step(view), MindCarry()
    for t, r in enumerate(rounds):
        _, carry, _ = step(jax.random.PRNGKey(t), r["obs"], carry, view_params)
    np.testing.assert_allclose(
        np.asarray(carry.beliefs["mission.visit"].probs),
        np.asarray(trajectory.carries["mission"].beliefs["mission.visit"].probs),
        rtol=1e-5,
        atol=1e-7,
    )


def test_a_read_is_one_tick_behind_under_sequential_order_too():
    comp, params = _composite(realise="mode")
    log: list = []
    rover, mission = subsystem(comp, params, "rover"), subsystem(comp, params, "mission")
    minds = {
        "rover": _recorded(agent_step(rover[0]), log, "rover"),
        "mission": _recorded(agent_step(mission[0]), log, "mission"),
    }
    run(comp, params, minds, 5, jax.random.PRNGKey(1), tick_order="sequential")
    for t in range(1, 5):
        # the rover responds first; the mission reads the posterior of the round before
        rover_before = _posterior(rover, log[2 * (t - 1)][1:])[SOURCE]
        datum = log[2 * t + 1][1][READ]
        assert int(datum) == int(np.argmax(np.asarray(cell_mass(rover_before, GRID))))


def test_the_pushed_belief_may_be_a_mixture_and_the_identity_kernel_reads_the_mean():
    mixture = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.25, 0.75])),
        means=jnp.asarray([[0.5, 0.5], [1.5, 0.5]]),
        chols=jnp.broadcast_to(0.05 * jnp.eye(2), (2, 2, 2)),
    )
    mass = np.asarray(cell_mass(mixture, GRID))
    np.testing.assert_allclose(mass, [0.25, 0.0, 0.75, 0.0], atol=1e-3)
    assert int(indicator(mixture, GRID, "mode", None, "e")) == 2
    np.testing.assert_allclose(
        np.asarray(indicator(mixture, GRID, "stats", None, "e")), mass, atol=1e-6
    )
    gaussian = Gaussian(loc=jnp.asarray([1.0, -1.0]), chol=0.1 * jnp.eye(2))
    np.testing.assert_allclose(np.asarray(identity(gaussian, "mode", None, "e")), [1.0, -1.0])
    draw = identity(gaussian, "sample", jax.random.PRNGKey(0), "e")
    assert draw.shape == (2,) and float(jnp.abs(draw - gaussian.loc).max()) < 1.0
    with pytest.raises(ValueError, match="deferred"):
        identity(gaussian, "stats", None, "e")
    with pytest.raises(ValueError, match="needs a key"):
        identity(gaussian, "sample", None, "e")


# ----------------------------------------------------------------------- the merge


def test_a_sibling_read_is_an_emission_and_an_observation_on_the_composite():
    comp, params = _composite(realise="mode")
    assert comp.observes == ((READ, EMITTED),)
    assert dict(comp.emissions)[EMITTED] == Emission((SOURCE,), "mode", GRID)
    table = dict(comp.subsystems)
    assert table["mission"].reads == ((READ, SOURCE, GRID, "mode"),)
    assert table["rover"].reads == () and table["rover"].emissions == (EMITTED,)
    # the reader's view leaves the modality an ordinary observation; the source's view emits
    view, _ = subsystem(comp, params, "mission")
    assert view.observes == () and view.emissions == ()
    source, _ = subsystem(comp, params, "rover")
    assert source.names.emissions == (EMITTED,) and source.observes == ()
    assert comp.ports(params).outs[EMITTED] == ("level", CELLS)
    assert READ not in comp.ports(params).ins


def test_the_identity_kernel_declares_no_static_kernel_on_the_emission():
    shared_r, shared_rp = _rover(shared=True)
    ss, ps = _scheduler()
    ps = dataclasses.replace(
        ps,
        likelihood={
            "cell_obs": ConditionalCategorical(jnp.log(jnp.full((2, 4), 1 / 2)), 1),
        },
    )
    comp, _ = merge(
        {"rover": shared_r, "mission": ss},
        {"rover": shared_rp, "mission": ps},
        [Link("rover.map", READ, realise="mode")],
    )
    assert dict(comp.emissions)[EMITTED] == Emission(("rover.map",), "mode", "identity")


def test_a_name_the_source_component_uses_is_refused_naming_the_explicit_form():
    (sr, pr), (ss, ps) = _rover(), _scheduler()
    sr = dataclasses.replace(
        sr, emissions={"cell_obs": Emission(("x",), "mode", kernel="identity")}
    )
    with pytest.raises(
        ValueError, match=r"named 'rover.cell_obs'.*another source, kernel or realisation"
    ):
        merge(
            {"rover": sr, "mission": ss},
            {"rover": pr, "mission": ps},
            [Link(SOURCE, READ, realise="mode", kernel=GRID)],
        )


def test_nested_is_true_for_a_component_bound_only_by_a_read_and_false_once_a_variable_is_shared():
    comp, _ = _composite()
    assert nested(comp, "mission") and nested(comp, "rover")
    assert dict(comp.subsystems)["mission"].nested
    shared, _ = _composite(shared=True, bindings=[Identify("rover.field", "mission.field")])
    assert not nested(shared, "mission") and not nested(shared, "rover")
    # no binding crosses the boundary: not nested
    (sr, pr), (ss, ps) = _rover(), _scheduler()
    apart, _ = merge({"rover": sr, "mission": ss}, {"rover": pr, "mission": ps})
    assert not nested(apart, "mission")


def test_merge_carries_a_components_own_observes_under_its_prefix():
    from tests.test_structure_learning import _own_read_system

    system, params = _own_read_system()
    other, other_params = _scheduler()
    comp, cparams = merge({"a": system, "mission": other}, {"a": params, "mission": other_params})
    assert comp.observes == (("a.prev_x", "a.x_mean"), ("a.prev_type", "a.kind_vector"))
    assert comp.names.emissions == ("a.x_mean", "a.kind_vector")
    view, _ = subsystem(comp, cparams, "a")
    assert view.observes == comp.observes
    # the composite reads its own datum as the component does
    obs = {"obs_x": jnp.asarray([1.2, -0.8]), "obs_kind": jnp.asarray(2, jnp.int32)}
    first, _ = system.infer_states(params, obs, None)
    datum = system.emit(jax.random.PRNGKey(0), params, first)
    second, _ = system.infer_states(params, {**obs, **datum}, None)
    c_obs = {f"a.{k}": v for k, v in {**obs, **datum}.items()}
    c_obs["mission.cell_obs"] = jnp.asarray(0, jnp.int32)
    got, _ = comp.infer_states(cparams, c_obs, None)
    np.testing.assert_allclose(
        np.asarray(got["a.z"].probs), np.asarray(second["z"].probs), rtol=1e-5
    )


def test_a_sibling_read_in_one_component_is_refused_naming_system_observes():
    (sr, pr), (ss, ps) = _rover(), _scheduler()
    with pytest.raises(ValueError, match=r"both in component 'rover'.*System\.observes"):
        merge(
            {"rover": sr, "mission": ss},
            {"rover": pr, "mission": ps},
            [Link(SOURCE, "rover.obs", realise="mode", kernel=GRID)],
        )


def test_a_level_count_that_is_not_the_source_s_or_the_grid_s_is_refused():
    with pytest.raises(ValueError, match=r"Link.*4 values \(the grid's cells\).*3 levels"):
        (sr, pr), (ss, ps) = _rover(), _scheduler()
        ps = dataclasses.replace(
            ps,
            likelihood={
                "cell_obs": ConditionalCategorical(jnp.log(jnp.full((3, CELLS), 1 / 3)), 1)
            },
        )
        merge(
            {"rover": sr, "mission": ss},
            {"rover": pr, "mission": ps},
            [Link(SOURCE, READ, realise="mode", kernel=GRID)],
        )
    # a categorical source under the identity kernel: the modality's levels are the source's
    shared_r, shared_rp = _rover(shared=True)
    ss, ps = _scheduler()
    with pytest.raises(ValueError, match=r"2 values \(the source's levels\).*4 levels"):
        merge(
            {"rover": shared_r, "mission": ss},
            {"rover": shared_rp, "mission": ps},
            [Link("rover.map", READ, realise="mode")],
        )


def test_a_categorical_modality_reading_a_continuous_source_needs_a_grid():
    with pytest.raises(ValueError, match=r"needs an indicator kernel.*Grid"):
        _composite(kernel="identity", realise="mode")


def test_stats_of_a_gaussian_source_is_refused_naming_the_deferral():
    with pytest.raises(ValueError, match="deferred"):
        _composite(kernel="identity", realise="stats")


def test_the_other_refusals():
    with pytest.raises(ValueError, match="the source is unknown"):
        (sr, pr), (ss, ps) = _rover(), _scheduler()
        merge(
            {"rover": sr, "mission": ss},
            {"rover": pr, "mission": ps},
            [Link("rover.nothing", READ, realise="mode", kernel=GRID)],
        )
    with pytest.raises(ValueError, match="already observes a variable"):
        _composite(bindings=[Link(SOURCE, READ, realise="stats", kernel=GRID)])
    with pytest.raises(ValueError, match="realise must be one of"):
        _composite(realise="legacy")


# ----------------------------------------------------------------------- the description


def test_a_read_round_trips_through_the_description_document():
    comp, cparams = _composite(realise="sample")
    document, arrays = describe(comp, cparams, arrays="inline")
    assert document["factors"][READ]["likelihood"]["observes"] == EMITTED
    assert document["factors"][EMITTED]["actuator"] == {
        "parents": [SOURCE],
        "options": {
            "kernel": "indicator",
            "grid": {"shape": [2, 2], "low": [0.0, 0.0], "high": [2.0, 2.0], "axes": [0, 1]},
        },
    }
    rebuilt, rebuilt_params = build(document, arrays=arrays)
    assert rebuilt.observes == comp.observes and rebuilt.emissions == comp.emissions
    assert dict(rebuilt.subsystems)["mission"].reads == dict(comp.subsystems)["mission"].reads
    # the document shows the boundary: a read crosses it as a blanket, a shared variable does not
    assert dict(rebuilt.subsystems)["mission"].nested and dict(rebuilt.subsystems)["rover"].nested
    shared, shared_params = _composite(
        shared=True, bindings=[Identify("rover.field", "mission.field")]
    )
    document2, arrays2 = describe(shared, shared_params, arrays="inline")
    assert not dict(build(document2, arrays=arrays2)[0].subsystems)["mission"].nested
    for a, b in zip(
        jax.tree_util.tree_leaves(cparams),
        jax.tree_util.tree_leaves(rebuilt_params),
        strict=True,
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # the rebuilt composite runs the read the same way
    _, rounds, _ = _run(rebuilt, rebuilt_params, 3)
    assert rounds[0]["obs"][READ] is None and rounds[2]["obs"][READ] is not None


def test_an_own_read_in_one_system_round_trips():
    system, params = _scheduler()
    system = dataclasses.replace(
        system,
        A_deps=((0,), (1,)),
        B_deps=((), ()),
        factor_names=("visit", "kind"),
        modality_names=("cell_obs", "prev_kind"),
        emissions={"kind_vector": Emission(("kind",), "stats", kernel="identity")},
        observes={"prev_kind": "kind_vector"},
    )
    params = dataclasses.replace(
        params,
        likelihood={
            **params.likelihood,
            "prev_kind": ConditionalCategorical(jnp.log(jnp.eye(3) * 0.8 + 0.1), 1),
        },
        transition={
            **params.transition,
            "kind": ConditionalCategorical(jnp.log(jnp.full((3, 1), 1 / 3)), 1),
        },
        initial_state_prior={
            **params.initial_state_prior,
            "kind": Categorical(logits=jnp.zeros(3)),
        },
    )
    document, arrays = describe(system, params, arrays="inline")
    assert document["factors"]["prev_kind"]["likelihood"]["observes"] == "kind_vector"
    assert document["factors"]["kind_vector"]["actuator"]["options"] == {
        "kernel": "identity",
        "realise": "stats",
    }
    rebuilt, _ = build(document, arrays=arrays)
    assert rebuilt == system


# ------------------------------------------- every read of a minded component is supplied


def test_a_components_own_emissions_are_supplied_from_its_own_posterior_under_run():
    from tests.test_structure_learning import _own_read_system

    system, params = _own_read_system()
    other, other_params = _scheduler()
    comp, cparams = merge({"a": system, "mission": other}, {"a": params, "mission": other_params})
    log: list = []
    view = subsystem(comp, cparams, "a")
    minds = {
        "a": _recorded(agent_step(view[0]), log, "a"),
        "mission": _recorded(agent_step(subsystem(comp, cparams, "mission")[0]), log, "mission"),
    }
    run(comp, cparams, minds, 5, jax.random.PRNGKey(0))
    seen = [(obs, carry) for m, obs, carry in log if m == "a"]
    # the view's modalities take their data keyed by the variables they observe
    assert seen[0][0]["a.x_mean"] is None and seen[0][0]["a.kind_vector"] is None
    assert "a.prev_x" not in seen[0][0]
    for t, (obs, _) in enumerate(seen[1:], start=1):  # the posterior of the round before
        before = _posterior(view, seen[t - 1])
        np.testing.assert_allclose(
            np.asarray(obs["a.x_mean"]), np.asarray(before["a.x"].mean()), rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(obs["a.kind_vector"]), np.asarray(before["a.kind"].probs), rtol=1e-6
        )


def test_a_sibling_read_inside_an_inner_composite_is_supplied_when_it_is_one_mind():
    inner, inner_params = _composite(realise="mode")
    other, other_params = _scheduler()
    outer, params = merge(
        {"tier": inner, "other": other}, {"tier": inner_params, "other": other_params}
    )
    table = dict(outer.subsystems)
    assert table["tier.mission"].reads == (
        (READ.replace("mission", "tier.mission"), "tier.rover.x", GRID, "mode"),
    )
    assert table["tier.mission"].boundary == "blanket" and table["tier"].reads == ()
    log: list = []
    view = subsystem(outer, params, "tier")
    minds = {
        "tier": _recorded(agent_step(view[0]), log, "tier"),
        "other": _recorded(agent_step(subsystem(outer, params, "other")[0]), log, "other"),
    }
    run(outer, params, minds, 6, jax.random.PRNGKey(0))
    seen = [(obs, carry) for m, obs, carry in log if m == "tier"]
    assert seen[0][0]["tier.rover.cell_obs"] is None
    for t, (obs, _) in enumerate(seen[1:], start=1):
        belief = _posterior(view, seen[t - 1])["tier.rover.x"]
        assert int(obs["tier.rover.cell_obs"]) == int(
            np.argmax(np.asarray(cell_mass(belief, GRID)))
        )


def test_a_held_mind_s_emissions_are_read_through_its_wrapper():
    from hybrid_aif.composition import HoldCarry
    from hybrid_aif.composition.driver import _emitted

    assert _emitted(HoldCarry(MindCarry(emitted={"x": 1}), 2, {})) == {"x": 1}
    assert _emitted(MindCarry()) is None


def test_the_compiled_face_takes_the_supplied_datum():
    from hybrid_aif import Emission as _Emission
    from tests.test_structure_learning import _own_read_system

    system, params = _own_read_system(
        emissions={
            "x_mean": _Emission(("x",), "mode", kernel="identity"),
            "kind_draw": _Emission(("kind",), "sample", kernel="identity"),
        },
        observes={"prev_x": "x_mean", "prev_type": "kind_draw"},
    )
    obs = {"obs_x": jnp.asarray([1.0, -1.0]), "obs_kind": jnp.asarray(2, jnp.int32)}
    first, _ = system.infer_states(params, obs, None)
    datum = system.emit(jax.random.PRNGKey(4), params, first)
    eager, _ = system.infer_states(params, {**obs, **datum}, None)
    fast, _ = system.compiled().infer_states(params, {**obs, **datum}, None)
    np.testing.assert_allclose(
        np.asarray(eager["z"].logits), np.asarray(fast["z"].logits), rtol=1e-5, atol=1e-6
    )


# ------------------------------------------------------------------ more of the merge


def test_merge_carries_own_observes_beside_a_sibling_read_and_refuses_a_second_read():
    from tests.test_structure_learning import _own_read_system

    system, params = _own_read_system()
    mission, mission_params = _scheduler()
    binding = Link("a.x", READ, realise="mode", kernel=GRID)
    comp, _ = merge(
        {"a": system, "mission": mission}, {"a": params, "mission": mission_params}, [binding]
    )
    assert comp.observes == (
        ("a.prev_x", "a.x_mean"),
        ("a.prev_type", "a.kind_vector"),
        (READ, "a.cell_obs"),
    )
    assert dict(comp.emissions)["a.cell_obs"] == Emission(("a.x",), "mode", GRID)
    with pytest.raises(ValueError, match="already observes a variable"):
        merge(
            {"a": system, "mission": mission},
            {"a": params, "mission": mission_params},
            [Link("mission.visit", "a.prev_x", realise="mode")],
        )


def test_an_own_observation_of_an_action_node_follows_its_node_and_is_refused_when_it_leaves():
    from hybrid_aif.composition import Clamp, Link
    from tests.test_merge import agent_and_world

    (agent, agent_params), (world, world_params) = agent_and_world()
    reading = dataclasses.replace(agent, observes={"location_obs": "a_location"})
    parts = ({"agent": reading, "world": world}, {"agent": agent_params, "world": world_params})
    comp, _ = merge(*parts)
    assert comp.observes == (("agent.location_obs", "agent.a_location"),)
    with pytest.raises(ValueError, match="a binding removed"):
        merge(*parts, [Clamp("agent.u_location", 1)])
    drive = Link("world.a_location", "agent.u_location")
    with pytest.raises(ValueError, match="another component's"):
        merge(*parts, [drive])


def test_an_observation_of_an_action_node_round_trips_through_the_document():
    from tests.test_merge import agent_and_world

    (agent, agent_params), _ = agent_and_world()
    system = dataclasses.replace(agent, observes={"location_obs": "a_location"})
    document, arrays = describe(system, agent_params, arrays="inline")
    assert document["factors"]["location_obs"]["likelihood"]["observes"] == "a_location"
    rebuilt, _ = build(document, arrays=arrays)
    assert rebuilt.observes == system.observes


def test_the_message_names_the_inner_merge_for_a_read_between_inner_components():
    inner, inner_params = _composite(realise="mode")
    other, other_params = _scheduler()
    with pytest.raises(ValueError, match="declare the link in the inner merge"):
        merge(
            {"tier": inner, "other": other},
            {"tier": inner_params, "other": other_params},
            [Link("tier.rover.x", "tier.mission.cell_obs", realise="mode", kernel=GRID)],
        )


# ------------------------------------------------------------ nested, under every binding


def test_the_boundary_under_each_binding_kind_from_merge_and_from_the_document():
    from hybrid_aif.composition import Clamp, Link
    from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world

    target = ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])),
        parent_ndims=1,
    )
    relay = Link("agent.location", "world.u_location")
    control = Link("agent.a_location", "world.u_location")
    cases = {
        "relay": ([relay], ("shared", "shared")),
        "control node": ([control], ("blanket", "blanket")),
        "structural link": ([OBSERVATION_BINDINGS[0]], ("shared", "shared")),
        "clamp": ([Clamp("world.u_location", 1)], ("none", "none")),
        "terminal link": (
            [Link(("agent.location",), "terminal:world.context")],
            ("shared", "shared"),
        ),
        "control node and link": ([control, *OBSERVATION_BINDINGS], ("shared", "shared")),
    }
    assert Link  # the link is among the bindings above
    for label, (bindings, expected) in cases.items():
        (agent, agent_params), (world, world_params) = agent_and_world()
        if "terminal" in label:
            world_params = dataclasses.replace(world_params, terminal_targets={"context": target})
        comp, cparams = merge(
            {"agent": agent, "world": world},
            {"agent": agent_params, "world": world_params},
            bindings,
        )
        made = dict(comp.subsystems)
        assert (made["agent"].boundary, made["world"].boundary) == expected, label
        document, arrays = describe(comp, cparams, arrays="inline")
        rebuilt = dict(build(document, arrays=arrays)[0].subsystems)
        assert (rebuilt["agent"].boundary, rebuilt["world"].boundary) == expected, label
        assert made["agent"].nested == (expected[0] == "blanket"), label
    shared, _ = _composite(shared=True, bindings=[Identify("rover.field", "mission.field")])
    assert {dict(shared.subsystems)[c].boundary for c in ("rover", "mission")} == {"shared"}


# ------------------------------------------------------------------------------- the pricing


def test_a_read_is_counted_from_an_episodes_second_tick_in_the_description_length():
    from hybrid_aif.description.mdl import _trace_resolution
    from tests.test_structure_learning import _own_read_system

    ticks = 6
    valid = np.ones((1, ticks), bool)
    system, _ = _own_read_system()
    n = _trace_resolution(system, valid)
    assert n["likelihood:prev_type"] == ticks - 1 and n["likelihood:obs_x"] == ticks
    comp, _ = _composite()
    n = _trace_resolution(comp, valid)
    assert n[f"likelihood:{READ}"] == ticks - 1 and n["likelihood:rover.obs"] == ticks
