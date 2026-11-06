"""The sibling-belief read (design: nesting-and-coupling.md N3): a modality of one component
takes its datum from another component's belief, through a kernel, realised; ``merge`` carries
it, ``subsystem`` leaves the modality an observation, and ``run`` supplies it each round."""

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
    Read,
    System,
    SystemParams,
    build,
    cell_mass,
    describe,
)
from hybrid_aif.composition import (
    MindCarry,
    SharedParameter,
    SiblingRead,
    agent_respond,
    merge,
    nested,
    run,
    subsystem,
)
from hybrid_aif.distributions import ConditionalGaussian
from hybrid_aif.reads import modality_key, read_datum

GRID = Grid(shape=(2, 2), low=0.0, high=2.0)
CELLS = GRID.n_cells
READ = "mission.cell_obs"
SOURCE = "rover.x"


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


def _scheduler(shared: bool = False, lagged=()):
    """A categorical ``visit`` factor over the grid's cells, observed through ``cell_obs``."""
    names = ("visit", "map") if shared else ("visit",)
    system = System(
        A_deps=((0,),),
        B_deps=((),) * len(names),
        factor_names=names,
        modality_names=("cell_obs",),
        epistemic=False,
        policy_len=1,
        lagged=lagged,
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
        [SiblingRead(READ, SOURCE, kernel, realise), *bindings],
    )


def _recorded(respond, log, mind: str):
    def wrapped(key, obs, carry):
        log.append((mind, dict(obs), carry))
        return respond(key, obs, carry)

    return wrapped


def _run(composite, params, ticks: int, key=0):
    """Run the two perceivers; the log holds ``(mind, obs seen, carry in)`` per respond."""
    log: list = []
    rover = subsystem(composite, params, "rover")
    mission = subsystem(composite, params, "mission")
    minds = {
        "rover": _recorded(agent_respond(*rover), log, "rover"),
        "mission": _recorded(agent_respond(*mission), log, "mission"),
    }
    trajectory = run(composite, params, minds, ticks, jax.random.PRNGKey(key))
    rounds = [
        {
            "obs": next(o for m, o, _ in log[2 * t : 2 * t + 2] if m == "mission"),
            "rover": next(c for m, _, c in log[2 * t : 2 * t + 2] if m == "rover"),
            "mission": next(c for m, _, c in log[2 * t : 2 * t + 2] if m == "mission"),
        }
        for t in range(ticks)
    ]
    return trajectory, rounds, mission


def _mass(round_):
    return np.asarray(cell_mass(round_["rover"].posterior[SOURCE], GRID))


# ------------------------------------------------------------------ the three realisations


def test_mode_datum_is_the_argmax_of_the_cell_mass_of_the_carried_belief():
    comp, params = _composite(realise="mode")
    _, rounds, _ = _run(comp, params, 8)
    assert rounds[0]["obs"][READ] is None  # the source has no carry yet: absent
    for r in rounds[1:]:
        datum = r["obs"][READ]
        assert datum.dtype == jnp.int32
        assert int(datum) == int(np.argmax(_mass(r)))


def test_sample_frequency_matches_the_cell_mass_and_mode_would_not():
    """A rover held at a cell edge (an uninformative observation, a tiny innovation) puts about
    half its mass on each of two cells: a draw's frequency matches the mass within the binomial
    standard error, and the same statistic under ``mode`` misses the bound by about twice its
    width at this seed and length."""
    hold = dict(loc=(1.0, 0.5), chol0=0.05, innovation=1e-3, noise=1e3)
    comp, params = _composite(realise="sample", rover=hold)
    ticks = 600
    _, rounds, _ = _run(comp, params, ticks)
    masses = np.array([_mass(r) for r in rounds[1:]])
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
    drawn = jax.vmap(lambda k: read_datum(belief, "identity", "sample", k))(keys)
    freq = np.bincount(np.asarray(drawn), minlength=3) / n
    tol = 4 * np.sqrt(0.25 / n)
    np.testing.assert_allclose(freq, [0.2, 0.5, 0.3], atol=tol)
    mode = read_datum(belief, "identity", "mode")
    assert int(mode) == 1 and abs(1.0 - 0.5) > tol


def test_the_soft_observation_is_within_the_jensen_bound_of_the_exact_treatment():
    """The note's acceptance: the log expected likelihood less the expected log-likelihood (the
    filter's soft observation) is non-negative, zero for a delta, and at most the log ratio of the
    largest to the smallest likelihood over the datum's support."""
    table = np.asarray(jax.nn.softmax(jnp.log(jnp.eye(CELLS) * 0.85 + 0.05), axis=0), np.float64)
    table = table / table.sum(axis=0)
    belief = Gaussian(loc=jnp.asarray([1.0, 0.7]), chol=0.4 * jnp.eye(2))
    mass = np.asarray(read_datum(belief, GRID, "stats"), np.float64)
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
    for r in rounds[1:]:
        datum = r["obs"][READ]
        np.testing.assert_allclose(np.asarray(datum), _mass(r), rtol=1e-5, atol=1e-7)
        prior = r["mission"].beliefs["mission.visit"]
        # the filter's soft observation: the expected log-likelihood under the vector
        want = jax.nn.softmax(prior.logits + datum @ log_a)
        got, _ = view.infer_states(view_params, {READ: datum}, priors=r["mission"].beliefs)
        np.testing.assert_allclose(
            np.asarray(got["mission.visit"].probs), np.asarray(want), rtol=1e-5, atol=1e-7
        )
    # the mind's own carry is what the recorded observations give, round by round
    respond, carry = agent_respond(view, view_params), MindCarry()
    for t, r in enumerate(rounds):
        _, carry = respond(jax.random.PRNGKey(t), r["obs"], carry)
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
        "rover": _recorded(agent_respond(*rover), log, "rover"),
        "mission": _recorded(agent_respond(*mission), log, "mission"),
    }
    run(comp, params, minds, 5, jax.random.PRNGKey(1), tick_order="sequential")
    for t in range(1, 5):
        rover_in = log[2 * t][2].posterior[SOURCE]  # the rover responds first, from last round's
        datum = log[2 * t + 1][1][READ]
        assert int(datum) == int(np.argmax(np.asarray(cell_mass(rover_in, GRID))))


def test_the_pushed_belief_may_be_a_mixture_and_the_identity_kernel_reads_the_mean():
    mixture = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.25, 0.75])),
        means=jnp.asarray([[0.5, 0.5], [1.5, 0.5]]),
        chols=jnp.broadcast_to(0.05 * jnp.eye(2), (2, 2, 2)),
    )
    mass = np.asarray(cell_mass(mixture, GRID))
    np.testing.assert_allclose(mass, [0.25, 0.0, 0.75, 0.0], atol=1e-3)
    assert int(read_datum(mixture, GRID, "mode")) == 2
    np.testing.assert_allclose(np.asarray(read_datum(mixture, GRID, "stats")), mass, atol=1e-6)
    gaussian = Gaussian(loc=jnp.asarray([1.0, -1.0]), chol=0.1 * jnp.eye(2))
    np.testing.assert_allclose(np.asarray(read_datum(gaussian, "identity", "mode")), [1.0, -1.0])
    draw = read_datum(gaussian, "identity", "sample", modality_key(jax.random.PRNGKey(0), 0))
    assert draw.shape == (2,) and float(jnp.abs(draw - gaussian.loc).max()) < 1.0
    with pytest.raises(ValueError, match="deferred"):
        read_datum(gaussian, "identity", "stats")
    with pytest.raises(ValueError, match="needs a key"):
        read_datum(gaussian, "identity", "sample")


# ----------------------------------------------------------------------- the merge


def test_a_sibling_read_is_a_read_on_the_composite_and_a_record_on_the_reader():
    comp, params = _composite(realise="mode")
    assert comp.lagged == (Read(READ, SOURCE, GRID, "mode"),)
    table = dict(comp.subsystems)
    assert table["mission"].reads == ((READ, SOURCE, GRID, "mode"),)
    assert table["rover"].reads == ()
    # the reader's view leaves the modality an ordinary observation
    view, _ = subsystem(comp, params, "mission")
    assert view.lagged == ()


def test_nested_is_true_for_a_component_bound_only_by_a_read_and_false_once_a_variable_is_shared():
    comp, _ = _composite()
    assert nested(comp, "mission") and nested(comp, "rover")
    assert dict(comp.subsystems)["mission"].nested
    shared, _ = _composite(shared=True, bindings=[SharedParameter("rover.field", "mission.field")])
    assert not nested(shared, "mission") and not nested(shared, "rover")
    # no binding crosses the boundary: not nested
    (sr, pr), (ss, ps) = _rover(), _scheduler()
    apart, _ = merge({"rover": sr, "mission": ss}, {"rover": pr, "mission": ps})
    assert not nested(apart, "mission")


def test_merge_carries_a_components_own_lagged_pairs_under_its_prefix():
    from tests.test_structure_learning import _lagged_system

    system, params = _lagged_system()
    other, other_params = _scheduler()
    comp, cparams = merge({"a": system, "mission": other}, {"a": params, "mission": other_params})
    assert comp.lagged == (("a.prev_x", "a.x"), ("a.prev_type", "a.kind"))
    view, _ = subsystem(comp, cparams, "a")
    assert view.lagged == comp.lagged
    # the composite reads its own carry as the component does
    obs = {"obs_x": jnp.asarray([1.2, -0.8]), "obs_kind": jnp.asarray(2, jnp.int32)}
    first, _ = system.infer_states(params, {**obs, "prev_x": None, "prev_type": None}, None)
    second, _ = system.infer_states(
        params, {**obs, "prev_x": None, "prev_type": None}, None, previous=first
    )
    c_obs = {f"a.{k}": v for k, v in {**obs, "prev_x": None, "prev_type": None}.items()}
    c_obs["mission.cell_obs"] = jnp.asarray(0, jnp.int32)
    prev = {f"a.{k}": v for k, v in first.items()}
    got, _ = comp.infer_states(cparams, c_obs, None, previous=prev)
    np.testing.assert_allclose(
        np.asarray(got["a.z"].probs), np.asarray(second["z"].probs), rtol=1e-5
    )


def test_a_sibling_read_in_one_component_is_refused_naming_system_lagged():
    (sr, pr), (ss, ps) = _rover(), _scheduler()
    with pytest.raises(ValueError, match=r"both in component 'rover'.*System\.lagged"):
        merge(
            {"rover": sr, "mission": ss},
            {"rover": pr, "mission": ps},
            [SiblingRead("rover.obs", SOURCE, GRID, "mode")],
        )


def test_a_level_count_that_is_not_the_source_s_or_the_grid_s_is_refused():
    with pytest.raises(ValueError, match=r"SiblingRead.*4 values \(the grid's cells\).*3 levels"):
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
            [SiblingRead(READ, SOURCE, GRID, "mode")],
        )
    # a categorical source under the identity kernel: the modality's levels are the source's
    shared_r, shared_rp = _rover(shared=True)
    ss, ps = _scheduler()
    with pytest.raises(ValueError, match=r"2 values \(the source's levels\).*4 levels"):
        merge(
            {"rover": shared_r, "mission": ss},
            {"rover": shared_rp, "mission": ps},
            [SiblingRead(READ, "rover.map", "identity", "mode")],
        )


def test_a_categorical_modality_reading_a_continuous_source_needs_a_grid():
    with pytest.raises(ValueError, match=r"needs an indicator kernel.*Grid"):
        _composite(kernel="identity", realise="mode")


def test_stats_of_a_gaussian_source_is_refused_naming_the_deferral():
    with pytest.raises(ValueError, match="deferred"):
        _composite(kernel="identity", realise="stats")


def test_the_other_refusals():
    with pytest.raises(ValueError, match="not a state factor"):
        (sr, pr), (ss, ps) = _rover(), _scheduler()
        merge(
            {"rover": sr, "mission": ss},
            {"rover": pr, "mission": ps},
            [SiblingRead(READ, "rover.nothing", GRID, "mode")],
        )
    with pytest.raises(ValueError, match="already has a read"):
        _composite(bindings=[SiblingRead(READ, SOURCE, GRID, "stats")])
    with pytest.raises(ValueError, match="realise must be one of"):
        _composite(realise="legacy")


# ----------------------------------------------------------------------- the description


def test_a_read_round_trips_through_the_description_document():
    comp, cparams = _composite(realise="sample")
    document, arrays = describe(comp, cparams, arrays="inline")
    options = document["factors"][READ]["likelihood"]["options"]
    assert options["datum"] == {
        "source": f"{SOURCE}[t-1]",
        "kernel": "indicator",
        "grid": {"shape": [2, 2], "low": [0.0, 0.0], "high": [2.0, 2.0], "axes": [0, 1]},
        "realise": "sample",
    }
    rebuilt, rebuilt_params = build(document, arrays=arrays)
    assert rebuilt.lagged == comp.lagged
    assert dict(rebuilt.subsystems)["mission"].reads == dict(comp.subsystems)["mission"].reads
    # the document shows the boundary: a read crosses it as a blanket, a shared variable does not
    assert dict(rebuilt.subsystems)["mission"].nested and dict(rebuilt.subsystems)["rover"].nested
    shared, shared_params = _composite(
        shared=True, bindings=[SharedParameter("rover.field", "mission.field")]
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


def test_a_read_in_one_system_round_trips_and_a_pair_keeps_its_spelling():
    system, params = _scheduler()
    system = dataclasses.replace(
        system,
        A_deps=((0,), (1,)),
        B_deps=((), ()),
        factor_names=("visit", "kind"),
        modality_names=("cell_obs", "prev_kind"),
        lagged=(("prev_kind", "kind"),),
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
    assert document["factors"]["prev_kind"]["likelihood"]["options"] == {"datum": "kind[t-1]"}
    rebuilt, _ = build(document, arrays=arrays)
    assert rebuilt == system


# ------------------------------------------- every read of a minded component is supplied


def _posteriors_seen(log, mind):
    return [(obs, carry) for m, obs, carry in log if m == mind]


def test_a_components_own_lagged_pairs_are_supplied_from_its_own_posterior_under_run():
    from tests.test_structure_learning import _lagged_system

    system, params = _lagged_system()
    other, other_params = _scheduler()
    comp, cparams = merge({"a": system, "mission": other}, {"a": params, "mission": other_params})
    log: list = []
    minds = {
        n: _recorded(agent_respond(*subsystem(comp, cparams, n)), log, n) for n in ("a", "mission")
    }
    run(comp, cparams, minds, 5, jax.random.PRNGKey(0))
    seen = _posteriors_seen(log, "a")
    assert seen[0][0]["a.prev_x"] is None and seen[0][0]["a.prev_type"] is None
    for obs, carry in seen[1:]:  # the posterior of the round before, not the world's sample
        np.testing.assert_allclose(
            np.asarray(obs["a.prev_x"]), np.asarray(carry.posterior["a.x"].mean()), rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(obs["a.prev_type"]), np.asarray(carry.posterior["a.kind"].probs), rtol=1e-6
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
    minds = {
        n: _recorded(agent_respond(*subsystem(outer, params, n)), log, n) for n in ("tier", "other")
    }
    run(outer, params, minds, 6, jax.random.PRNGKey(0))
    seen = _posteriors_seen(log, "tier")
    assert seen[0][0]["tier.mission.cell_obs"] is None
    for obs, carry in seen[1:]:
        belief = carry.posterior["tier.rover.x"]
        assert int(obs["tier.mission.cell_obs"]) == int(
            np.argmax(np.asarray(cell_mass(belief, GRID)))
        )


def test_a_held_mind_s_posterior_is_read_through_its_wrapper():
    from hybrid_aif.composition import HoldCarry
    from hybrid_aif.composition.driver import _posterior

    assert _posterior(HoldCarry(MindCarry(posterior={"x": 1}), 2, {})) == {"x": 1}
    assert _posterior(MindCarry()) is None


def test_the_compiled_face_takes_the_read_key():
    from hybrid_aif.reads import Read as _Read
    from tests.test_structure_learning import _lagged_system

    system, params = _lagged_system(lagged=(_Read("prev_type", "kind", "identity", "sample"),))
    obs = {
        "obs_x": jnp.asarray([1.0, -1.0]),
        "obs_kind": jnp.asarray(2, jnp.int32),
        "prev_x": jnp.zeros(2),
        "prev_type": None,
    }
    first, _ = system.infer_states(params, obs, None)
    key = jax.random.PRNGKey(4)
    eager, _ = system.infer_states(params, obs, None, previous=first, read_key=key)
    fast, _ = system.compiled().infer_states(params, obs, None, previous=first, read_key=key)
    np.testing.assert_allclose(
        np.asarray(eager["z"].logits), np.asarray(fast["z"].logits), rtol=1e-5, atol=1e-6
    )


# ------------------------------------------------------------------ more of the merge


def test_merge_carries_own_pairs_beside_a_sibling_read_and_refuses_a_second_read():
    from tests.test_structure_learning import _lagged_system

    system, params = _lagged_system()
    mission, mission_params = _scheduler()
    binding = SiblingRead(READ, "a.x", GRID, "mode")
    comp, _ = merge(
        {"a": system, "mission": mission}, {"a": params, "mission": mission_params}, [binding]
    )
    assert comp.lagged == (("a.prev_x", "a.x"), ("a.prev_type", "a.kind"), Read(*binding))
    with pytest.raises(ValueError, match="already has a read"):
        merge(
            {"a": system, "mission": mission},
            {"a": params, "mission": mission_params},
            [SiblingRead("a.prev_x", "mission.visit", "identity", "mode")],
        )


def test_an_own_pair_on_an_action_node_follows_its_node_and_is_refused_when_it_leaves():
    from hybrid_aif.composition import ActionControl, BoundaryPrior
    from tests.test_merge import agent_and_world

    (agent, agent_params), (world, world_params) = agent_and_world()
    reading = dataclasses.replace(agent, lagged=(("location_obs", "a_location"),))
    parts = ({"agent": reading, "world": world}, {"agent": agent_params, "world": world_params})
    comp, _ = merge(*parts)
    assert comp.lagged == (("agent.location_obs", "agent.a_location"),)
    with pytest.raises(ValueError, match="a binding removed"):
        merge(*parts, [BoundaryPrior("agent.u_location", 1)])
    drive = ActionControl(driver="world.u_location", control="agent.u_location")
    with pytest.raises(ValueError, match="another component's"):
        merge(*parts, [drive])


def test_a_read_of_an_action_node_round_trips_through_the_document():
    from hybrid_aif.reads import Read as _Read
    from tests.test_merge import agent_and_world

    (agent, agent_params), _ = agent_and_world()
    system = dataclasses.replace(
        agent, lagged=(_Read("location_obs", "a_location", "identity", "mode"),)
    )
    document, arrays = describe(system, agent_params, arrays="inline")
    options = document["factors"]["location_obs"]["likelihood"]["options"]
    assert options["datum"] == {
        "source": "a_location[t-1]",
        "kernel": "identity",
        "realise": "mode",
    }
    rebuilt, _ = build(document, arrays=arrays)
    assert rebuilt.lagged == system.lagged


def test_a_user_read_cannot_declare_the_pair_s_realisation_and_the_message_names_the_inner_merge():
    with pytest.raises(ValueError, match="realise must be one of"):
        Read("m", "s", "identity", "legacy")
    inner, inner_params = _composite(realise="mode")
    other, other_params = _scheduler()
    with pytest.raises(ValueError, match="declare the SiblingRead in the inner merge"):
        merge(
            {"tier": inner, "other": other},
            {"tier": inner_params, "other": other_params},
            [SiblingRead("tier.mission.cell_obs", "tier.rover.x", GRID, "mode")],
        )


# ------------------------------------------------------------ nested, under every binding


def test_the_boundary_under_each_binding_kind_from_merge_and_from_the_document():
    from hybrid_aif.composition import ActionControl, BoundaryPrior, ObservationLink
    from hybrid_aif.composition.merge import TargetEmission
    from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world

    target = ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])),
        parent_ndims=1,
    )
    relay = ActionControl(driver="agent.location", control="world.u_location")
    control = ActionControl(driver="agent.u_location", control="world.u_location")
    cases = {
        "relay": ([relay], ("shared", "shared")),
        "control node": ([control], ("blanket", "blanket")),
        "observation link": ([OBSERVATION_BINDINGS[0]], ("shared", "shared")),
        "boundary prior": ([BoundaryPrior("world.u_location", 1)], ("none", "none")),
        "target emission": (
            [TargetEmission(target="world.context", parents=("agent.location",))],
            ("shared", "shared"),
        ),
        "control node and link": ([control, *OBSERVATION_BINDINGS], ("shared", "shared")),
    }
    assert ObservationLink  # the link is among the bindings above
    for label, (bindings, expected) in cases.items():
        (agent, agent_params), (world, world_params) = agent_and_world()
        if "target" in label:
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
    shared, _ = _composite(shared=True, bindings=[SharedParameter("rover.field", "mission.field")])
    assert {dict(shared.subsystems)[c].boundary for c in ("rover", "mission")} == {"shared"}


# ------------------------------------------------------------------------------- the pricing


def test_a_read_is_counted_from_an_episodes_second_tick_in_the_description_length():
    from hybrid_aif.description.mdl import _trace_resolution
    from tests.test_structure_learning import _lagged_system

    ticks = 6
    valid = np.ones((1, ticks), bool)
    system, _ = _lagged_system(lagged=(Read("prev_type", "kind", "identity", "sample"),))
    n = _trace_resolution(system, valid)
    assert n["likelihood:prev_type"] == ticks - 1 and n["likelihood:obs_x"] == ticks
    comp, _ = _composite()
    n = _trace_resolution(comp, valid)
    assert n[f"likelihood:{READ}"] == ticks - 1 and n["likelihood:rover.obs"] == ticks
