"""``System.observes``: a slot (a modality or a control node) observes a variable the System
generates (an emission or an action node), and the variable set is derived through the map.

Its readers are the ports (a slot the map names is no in-port), the keyed faces (a datum is keyed
by the variable and fans out to every slot observing it), the description (an ``observes`` entry on
the likelihood record, a record of the static kernel on the emission) and the description length
(a slot the map names is no variable of its own). An emission may carry the identity or a grid as
its static kernel, ``mirror`` drops what only the System itself observes, and ``replay`` realises
and records the emissions.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hybrid_aif
import hybrid_aif.reads as reads
from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    Emission,
    Gaussian,
    Grid,
    LinearGaussianObservation,
    LinearGaussianTransition,
    System,
    SystemParams,
    build,
    describe,
    replay,
)
from hybrid_aif.cells import cell_index, cell_mass
from hybrid_aif.controls import FLOAT, LEVEL
from hybrid_aif.description.mdl import description_length, log_star
from hybrid_aif.distributions import ConditionalGaussian
from hybrid_aif.system.mirror import mirror
from tests.test_action_nodes import pusher
from tests.test_own_read_equivalence import GRID, model

KEY = jax.random.PRNGKey(0)


def _flat(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


# ------------------------------------------------------------------------------ the declaration


def test_emit_and_the_fixed_lag_read_are_gone():
    for name in ("Emit", "Read", "read_datum", "LEGACY", "READ_TAG", "modality_key", "as_read"):
        assert not hasattr(reads, name), name
        assert not hasattr(hybrid_aif, name), name
    assert reads.EMIT_TAG != 0 and hybrid_aif.Emission is reads.Emission
    assert not hasattr(System, "lagged") and "lagged" not in System.__dataclass_fields__


def test_an_emission_is_a_parents_realise_kernel_triple_with_no_kernel_by_default():
    e = Emission(("x",))
    assert e == Emission(("x",), "sample", None)
    assert Emission(("x",), "mode", GRID).kernel is GRID


def test_observes_is_normalised_from_a_mapping_and_validated_structurally():
    system, _ = model()
    assert system.observes == (
        ("read_x", "x_mean"),
        ("read_kind", "kind_vector"),
        ("cell_mode", "cell_largest"),
        ("cell_stats", "cell_mass"),
    )
    with pytest.raises(TypeError, match="observes maps"):
        dataclasses.replace(system, observes={"read_x": 3})
    with pytest.raises(ValueError, match="neither a modality"):
        dataclasses.replace(system, observes={"nowhere": "x_mean"})
    with pytest.raises(ValueError, match="twice"):
        dataclasses.replace(system, observes=(("read_x", "x_mean"), ("read_x", "cell_mass")))
    # a state factor is no variable a slot may observe: only what the System generates is
    with pytest.raises(ValueError, match="does not generate"):
        dataclasses.replace(system, observes={"read_x": "x"})
    with pytest.raises(ValueError, match="does not generate"):
        dataclasses.replace(system, observes={"read_x": "obs_kind"})


def test_the_port_types_of_a_slot_and_its_variable_must_agree():
    system, params = model()
    system._validate(params)
    for slot, variable in (
        ("read_x", "kind_vector"),  # a float slot reading a level port
        ("read_kind", "cell_largest"),  # three levels reading four cells
        ("cell_mode", "x_mean"),
    ):
        wrong = dataclasses.replace(system, observes={**dict(system.observes), slot: variable})
        with pytest.raises(ValueError, match="port types must be equal"):
            wrong._validate(params)


def test_a_control_node_slot_is_driven_and_has_no_action_node():
    system, params = pusher()
    emissions = {"x_out": Emission(("x",), "mode", kernel="identity")}
    with pytest.raises(ValueError, match="its action node 'push' is named"):
        dataclasses.replace(system, emissions=emissions, observes={"u_push": "x_out"})
    driven = dataclasses.replace(
        system, emissions=emissions, observes={"u_push": "x_out"}, action_names=(None,)
    )
    assert "u_push" not in driven.ports(params).ins and "push" not in driven.ports(params).outs
    with pytest.raises(ValueError, match=r"control node\(s\) \['u_push'\] are driven"):
        driven.policies(params)
    with pytest.raises(ValueError, match="neither a modality"):
        dataclasses.replace(system, emissions={"e": ("x",)}, observes={"x": "e"})


# ----------------------------------------------------------------------------------- ports


def test_ports_list_in_ports_the_map_does_not_name_and_type_every_out_port():
    system, params = model()
    ports = system.ports(params)
    assert ports.ins == {"obs_x": (FLOAT, 2), "obs_kind": (LEVEL, 3)}
    assert ports.outs == {
        "x_mean": (FLOAT, 2),
        "kind_vector": (LEVEL, 3),
        "cell_largest": (LEVEL, GRID.n_cells),
        "cell_mass": (LEVEL, GRID.n_cells),
    }
    plain = dataclasses.replace(system, observes={})
    assert set(plain.ports(params).ins) == set(system.modality_names)


def test_an_action_node_is_a_variable_a_modality_observes():
    system, params = pusher()
    system = dataclasses.replace(
        system,
        A_deps=((0,), (0,)),
        modality_names=("see", "prev_push"),
        observes={"prev_push": "push"},
    )
    params = dataclasses.replace(
        params,
        likelihood={
            **params.likelihood,
            "prev_push": ConditionalCategorical(jnp.log(0.7 * jnp.eye(3) + 0.1), 1),
        },
    )
    system._validate(params)
    assert system.ports(params).ins == {"see": (LEVEL, 3), "u_push": (LEVEL, 3)}
    assert system.ports(params).outs["push"] == (LEVEL, 3)


# ------------------------------------------------------------------------------- keyed faces


def _two_slot_system():
    """``x`` a random walk with two modalities, ``a`` and ``b``, that observe the one emission
    ``mean`` of it (and ``direct``, the same system with both observing the datum by name)."""
    base = dict(
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        factor_names=("x",),
        epistemic=False,
    )
    both = System(
        A_deps=((0,), (0,)),
        modality_names=("a", "b"),
        emissions={"mean": Emission(("x",), "mode", kernel="identity")},
        observes={"a": "mean", "b": "mean"},
        **base,
    )
    direct = System(A_deps=((0,), (0,)), modality_names=("a", "b"), **base)
    params = SystemParams(
        likelihood={
            "a": LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=0.2 * jnp.eye(1)),
            "b": LinearGaussianObservation(
                H=2 * jnp.eye(1), c=jnp.zeros(1), noise=0.4 * jnp.eye(1)
            ),
        },
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.full((1, 1, 1), 0.5), parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1))},
    )
    return both, direct, params


def test_a_datum_keyed_by_its_variable_feeds_every_slot_observing_it():
    both, direct, params = _two_slot_system()
    datum = jnp.asarray([0.8])
    got, info = both.infer_states(params, {"mean": datum}, detail=True)
    want, want_info = direct.infer_states(params, {"a": datum, "b": datum}, detail=True)
    for a, b in zip(_flat(got), _flat(want), strict=True):
        np.testing.assert_array_equal(a, b)
    assert set(info.log_evidence) == {"a", "b"}
    for name in ("a", "b"):
        np.testing.assert_array_equal(
            np.asarray(info.log_evidence[name]), np.asarray(want_info.log_evidence[name])
        )
    # the learning face fans out the same way
    np.testing.assert_array_equal(
        *[
            np.asarray(s.learn(params, o, q, prev_beliefs=q).likelihood["a"].c)
            for s, o, q in ((both, {"mean": datum}, got), (direct, {"a": datum, "b": datum}, want))
        ]
    )


def test_a_key_naming_a_renamed_slot_is_refused_naming_the_variable():
    both, _, params = _two_slot_system()
    with pytest.raises(
        ValueError, match=r"names 'a', a modality that observes the variable 'mean'"
    ):
        both.infer_states(params, {"a": jnp.asarray([0.8])})
    with pytest.raises(ValueError, match="have unknown key"):
        both.infer_states(params, {"mean": jnp.asarray([0.8]), "elsewhere": 1})
    with pytest.raises(ValueError, match=r"present names 'b'.*key its datum by 'mean'"):
        both.infer_states(params, {"mean": jnp.asarray([0.8])}, present={"b": True})


def test_the_datum_of_a_generated_variable_is_optional_and_a_missing_one_is_absent():
    both, _, params = _two_slot_system()
    prior = both.infer_states(params, {"mean": None})[0]["x"]
    left_out = both.infer_states(params, {})[0]["x"]
    flagged = both.infer_states(params, {"mean": jnp.asarray([5.0])}, present={"mean": False})[0][
        "x"
    ]
    for q in (left_out, flagged):
        np.testing.assert_array_equal(np.asarray(q.loc), np.asarray(prior.loc))
    # a plain modality's datum is still required
    system, _ = model()
    with pytest.raises(ValueError, match=r"observations are missing \['obs_kind'\]"):
        system.infer_states(model()[1], {"obs_x": jnp.zeros(2)})


# --------------------------------------------------------------------------------- describe


def test_describe_writes_an_observes_entry_only_where_the_variable_differs_from_the_slot():
    system, params = model()
    document, arrays = describe(system, params, arrays="inline")
    factors = document["factors"]
    for slot, variable in system.observes:
        assert factors[slot]["likelihood"]["observes"] == variable
    assert "observes" not in factors["obs_x"]["likelihood"]
    assert "observes" not in factors["obs_kind"]["likelihood"]
    # the emission is a variable with a record of its own, naming no family and no parameters
    assert factors["cell_largest"]["actuator"] == {
        "parents": ["x"],
        "options": {
            "kernel": "indicator",
            "grid": {"shape": [2, 2], "low": [-2.0, -2.0], "high": [2.0, 2.0], "axes": [0, 1]},
            "realise": "mode",
        },
    }
    assert factors["kind_vector"]["actuator"] == {
        "parents": ["kind"],
        "options": {"kernel": "identity", "realise": "stats"},
    }
    assert document["variables"]["cell_mass"] == {"domain": "categorical", "levels": 4}
    assert document["variables"]["x_mean"]["dim"] == 2
    rebuilt, rebuilt_params = build(document, arrays=arrays)
    assert rebuilt == system
    assert rebuilt_params.emission == {}
    for a, b in zip(_flat(rebuilt_params), _flat(params), strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_document_refuses_a_malformed_observes_or_static_kernel():
    from hybrid_aif import DescriptionError

    system, params = model()
    document, arrays = describe(system, params, arrays="inline")

    def broken(path, value):
        copy = jax.tree_util.tree_map(lambda x: x, document)  # a deep structural copy
        target = copy
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return copy

    with pytest.raises(ValueError, match="which this System does not generate"):
        build(broken(("factors", "read_x", "likelihood", "observes"), "x"), arrays=arrays)
    with pytest.raises(DescriptionError, match="observes"):
        build(broken(("factors", "read_x", "likelihood", "observes"), 3), arrays=arrays)
    with pytest.raises(DescriptionError, match="'identity' or 'indicator'"):
        build(
            broken(("factors", "x_mean", "actuator", "options"), {"kernel": "median"}),
            arrays=arrays,
        )
    with pytest.raises(DescriptionError, match="an indicator kernel has a grid"):
        build(
            broken(("factors", "cell_largest", "actuator", "options"), {"kernel": "indicator"}),
            arrays=arrays,
        )


# --------------------------------------------------------------------------------------- MDL


def test_a_slot_the_map_names_is_no_variable_of_its_own_in_the_description_length():
    system, params = model()
    dl = description_length(system, 100.0, params=params)
    f, u = len(system.factor_names), 0
    named = {slot for slot, _ in system.observes}
    modalities = len(system.modality_names) - len(named)
    emissions = len(system.emissions)
    assert dl.per_address["graph"].graph == pytest.approx(
        log_star(f) + log_star(modalities + 1) + log_star(u + 1) + log_star(emissions + 1)
    )
    plain = dataclasses.replace(system, observes={})
    base = description_length(plain, 100.0, params=params).per_address["graph"].graph
    assert base == pytest.approx(
        log_star(f)
        + log_star(len(system.modality_names) + 1)
        + log_star(u + 1)
        + log_star(emissions + 1)
    )
    assert dl.per_address["graph"].graph < base  # four slots fewer, the same emissions
    # a static kernel is structure only: its parents and the size of what it emits
    for e, _ in system.emissions:
        link = dl.per_address[f"emission:{e}"]
        assert link.family is None and link.prior is None and link.graph > 0.0
        assert (f"emission:{e}", "family") not in dl.unscored
    assert dl.per_address["emission:x_mean"].note == "the identity"
    assert dl.per_address["emission:cell_mass"].note == "an indicator kernel over a grid"
    # each modality still has its own likelihood record, the table that explains its datum
    assert all(f"likelihood:{m}" in dl.per_address for m in system.modality_names)


# ------------------------------------------------------------------- the static emission kernel


def test_the_identity_realises_a_levels_vector_or_location_from_a_point_or_a_belief():
    system, params = model()
    belief = Gaussian(loc=jnp.asarray([0.4, -1.0]), chol=0.3 * jnp.eye(2))
    kind = Categorical.from_probs(jnp.asarray([0.2, 0.5, 0.3]))
    values = {"x": belief, "kind": kind}
    out = system.emit(KEY, params, values)
    np.testing.assert_array_equal(np.asarray(out["x_mean"]), np.asarray(belief.mean()))
    np.testing.assert_array_equal(np.asarray(out["kind_vector"]), np.asarray(kind.probs))
    # from a point a categorical factor's stats are the one-hot and a Gaussian's mode its value
    point = system.emit(KEY, params, {"x": jnp.asarray([0.4, -1.0]), "kind": jnp.asarray(1)})
    np.testing.assert_allclose(np.asarray(point["x_mean"]), [0.4, -1.0], rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(point["kind_vector"]), [0.0, 1.0, 0.0])
    # a sample of a categorical belief takes the key; a mode does not
    sampled = dataclasses.replace(
        system,
        emissions={
            **dict(system.emissions),
            "kind_vector": Emission(("kind",), "sample", kernel="identity"),
        },
    )
    draws = {
        int(sampled.emit(jax.random.PRNGKey(i), params, values)["kind_vector"]) for i in range(30)
    }
    assert draws == {0, 1, 2}


def test_a_grid_is_an_indicator_a_cell_lookup_from_a_point_and_cell_mass_from_a_belief():
    system, params = model()
    belief = Gaussian(loc=jnp.asarray([0.9, -0.2]), chol=0.4 * jnp.eye(2))
    point = jnp.asarray([0.9, -0.2])
    kind = Categorical.from_probs(jnp.asarray([0.2, 0.5, 0.3]))
    from_belief = system.emit(KEY, params, {"x": belief, "kind": kind})
    from_point = system.emit(KEY, params, {"x": point, "kind": jnp.asarray(0)})
    mass = cell_mass(belief, GRID)
    np.testing.assert_array_equal(np.asarray(from_belief["cell_mass"]), np.asarray(mass))
    assert int(from_belief["cell_largest"]) == int(jnp.argmax(mass))
    assert int(from_point["cell_largest"]) == int(cell_index(point, GRID)) == 2
    np.testing.assert_array_equal(np.asarray(from_point["cell_mass"]), np.eye(4)[2])
    # the cells follow the grid's order, the outer cells absorbing the tails
    for p, cell in (([-9.0, -9.0], 0), ([-9.0, 9.0], 1), ([9.0, -9.0], 2), ([9.0, 9.0], 3)):
        assert int(cell_index(jnp.asarray(p), GRID)) == cell


def test_a_static_kernel_is_checked_against_its_parent():
    system, params = model()
    over_categorical = dataclasses.replace(
        system,
        emissions={**dict(system.emissions), "cell_largest": Emission(("kind",), "mode", GRID)},
    )
    with pytest.raises(ValueError, match="categorical factor 'kind' through an indicator"):
        over_categorical._validate(params)
    far = Grid(shape=(2, 2), low=-2.0, high=2.0, axes=(0, 5))
    off_axis = dataclasses.replace(
        system, emissions={**dict(system.emissions), "cell_largest": Emission(("x",), "mode", far)}
    )
    with pytest.raises(ValueError, match="every axis must be a coordinate"):
        off_axis._validate(params)
    two_parents = dataclasses.replace(
        system, emissions={**dict(system.emissions), "x_mean": Emission(("x", "kind"), "mode")}
    )
    with pytest.raises(ValueError, match="one kernel per emission"):
        two_parents._validate(params)
    with pytest.raises(ValueError, match="a static kernel .* reads one state factor"):
        dataclasses.replace(
            system, emissions={"c": Emission(("x", "kind"), "mode", GRID)}, observes={}
        )
    with pytest.raises(ValueError, match="a static kernel is 'identity' or a Grid"):
        dataclasses.replace(
            system, emissions={"c": Emission(("x",), "mode", "median")}, observes={}
        )
    with pytest.raises(ValueError, match="holds a kernel for"):
        system._validate(
            dataclasses.replace(
                params,
                emission={
                    "cell_largest": ConditionalCategorical(jnp.log(jnp.full((4, 2), 0.25)), 1)
                },
            )
        )


# --------------------------------------------------------------------------------- mirror


def test_the_mirror_drops_an_emission_only_its_own_system_observes_with_its_entry():
    system, params = model()
    world, wp = mirror(system, params)
    assert world.observes == () and world.emissions == tuple(
        (m, Emission(tuple(_parents(system, i)))) for i, m in enumerate(system.modality_names)
    )
    assert world.names.emissions == system.modality_names
    assert set(wp.emission) == set(system.modality_names)
    # as_env is the mirror run forward, keyed by the System's own modalities
    env, ep = system.as_env(params)
    obs, state = env.reset(KEY, ep)
    assert set(obs) == set(system.modality_names)
    # every emission of the System is dropped, observed by the System or by nobody
    extra = dataclasses.replace(
        system,
        emissions={**dict(system.emissions), "export": Emission(("x",), "mode", kernel="identity")},
    )
    exported, _ = mirror(extra, params)
    assert exported.names.emissions == system.modality_names and exported.observes == ()
    extra.as_env(params)


def _parents(system, m):
    return [system.factor_names[g] for g in system.A_deps[m]]


# --------------------------------------------------------------------------------- replay


def test_replay_records_the_emissions_it_realises_and_feeds_the_previous_tick_back():
    system, params = model()
    rng = np.random.default_rng(1)
    T = 6
    xs = jnp.asarray(np.cumsum(rng.normal(size=(T, 2)), axis=0), jnp.float32)
    kinds = jnp.asarray(rng.integers(0, 3, size=T), jnp.int32)
    out = replay(system, params, {"obs_x": xs, "obs_kind": kinds}, {}, metrics=False)
    assert set(out.emitted) == set(dict(system.emissions))
    assert out.emitted["cell_mass"].shape == (T, 4) and out.emitted["x_mean"].shape == (T, 2)
    np.testing.assert_allclose(np.asarray(out.emitted["cell_mass"]).sum(-1), 1.0, rtol=1e-5)
    # tick t's inference read tick t - 1's emissions, tick 0 had none
    priors = {n: jax.tree_util.tree_map(lambda a: a[3], b) for n, b in out.priors.items()}
    datum = {k: v[2] for k, v in out.emitted.items()}
    want, _ = system.infer_states(params, {"obs_x": xs[3], "obs_kind": kinds[3], **datum}, priors)
    np.testing.assert_array_equal(np.asarray(out.beliefs["x"].loc[3]), np.asarray(want["x"].loc))
    # a System that declares no emission records none
    _, direct, direct_params = _two_slot_system()
    trace = jnp.asarray(rng.normal(size=(T, 1)), jnp.float32)
    assert replay(direct, direct_params, {"a": trace, "b": trace}, {}, metrics=False).emitted == {}


def test_replay_feeds_the_previous_action_to_a_modality_that_observes_an_action_node():
    system, params = pusher()
    system = dataclasses.replace(
        system,
        A_deps=((0,), (0,)),
        modality_names=("see", "prev_push"),
        observes={"prev_push": "push"},
    )
    params = dataclasses.replace(
        params,
        likelihood={
            **params.likelihood,
            "prev_push": ConditionalCategorical(jnp.log(0.7 * jnp.eye(3) + 0.1), 1),
        },
    )
    seen = jnp.asarray([0, 1, 2, 1, 0], jnp.int32)
    pushes = jnp.asarray([1, 2, 0, 0, 1], jnp.int32)
    out = replay(system, params, {"see": seen}, {"push": pushes}, metrics=False)
    # tick 2 observed the push taken at tick 1, as a hand call does
    priors = {n: jax.tree_util.tree_map(lambda a: a[2], b) for n, b in out.priors.items()}
    want, _ = system.infer_states(params, {"see": seen[2], "push": pushes[1]}, priors)
    np.testing.assert_allclose(
        np.asarray(out.beliefs["x"].probs[2]), np.asarray(want["x"].probs), rtol=1e-6
    )
    first, _ = system.infer_states(params, {"see": seen[0]})
    np.testing.assert_allclose(
        np.asarray(out.beliefs["x"].probs[0]), np.asarray(first["x"].probs), rtol=1e-6
    )


def test_a_composite_carries_the_entries_and_its_ports_leave_the_named_slots_out():
    system, params = model()
    # the view of a one-component composite keeps the System's own emission and entries
    from hybrid_aif.composition import merge, subsystem

    comp, cparams = merge({"a": system}, {"a": params})
    view, _ = subsystem(comp, cparams, "a")
    assert view.observes == comp.observes
    assert comp.ports(cparams).ins == {"a.obs_x": (FLOAT, 2), "a.obs_kind": (LEVEL, 3)}


# ----------------------------------------------------------------------- the fix pass


def test_a_missing_params_kernel_is_refused_and_the_identity_is_explicit():
    system, params = model()
    nokernel = dataclasses.replace(
        system, emissions={**dict(system.emissions), "x_mean": Emission(("x",), "mode")}
    )
    with pytest.raises(ValueError, match="one kernel per emission"):
        nokernel._validate(params)
    with pytest.raises(ValueError, match="one kernel per emission"):
        nokernel.emit(KEY, params, {"x": jnp.zeros(2), "kind": jnp.asarray(0)})
    with pytest.raises(ValueError, match="'identity' or a Grid"):
        dataclasses.replace(system, emissions={"c": Emission(("x",), "mode", "median")})


def test_two_links_with_one_source_kernel_and_realisation_share_the_emission():
    from hybrid_aif.composition import Link, merge, subsystem
    from tests.test_sibling_read import GRID as G
    from tests.test_sibling_read import _rover, _scheduler

    (sr, pr), (m1, p1), (m2, p2) = _rover(), _scheduler(), _scheduler()
    parts = ({"rover": sr, "m1": m1, "m2": m2}, {"rover": pr, "m1": p1, "m2": p2})
    both = [
        Link("rover.x", "m1.cell_obs", realise="mode", kernel=G),
        Link("rover.x", "m2.cell_obs", realise="mode", kernel=G),
    ]
    comp, cp = merge(*parts, both)
    assert [n for n, _ in comp.emissions] == ["rover.cell_obs"]
    assert comp.observes == (("m1.cell_obs", "rover.cell_obs"), ("m2.cell_obs", "rover.cell_obs"))
    assert comp.ports(cp).ins == {"rover.obs": (FLOAT, 2)}
    # a different realisation under the same name is refused
    with pytest.raises(ValueError, match="another source, kernel or realisation"):
        merge(
            *parts,
            [both[0], Link("rover.x", "m2.cell_obs", realise="stats", kernel=G)],
        )
    # the source component's view mirrors and runs as an environment
    view, vp = subsystem(comp, cp, "rover")
    assert view.names.emissions == ("rover.cell_obs",)
    world, _ = mirror(view, vp)
    assert world.names.emissions == ("rover.obs",) and world.observes == ()
    view.as_env(vp)


def test_infer_policies_takes_present_keyed_by_variable_as_the_other_faces_do():
    both, _, params = _two_slot_system()
    beliefs = both.initial_priors(params)
    from hybrid_aif.system.faces import plan as _plan

    absent = _plan._present_in(
        both, {"mean": False}, 2, "infer_policies", both._params_in(params, "t")
    )
    assert absent is not None
    with pytest.raises(
        ValueError, match=r"names 'a', a modality that observes the variable 'mean'"
    ):
        both.infer_policies(params, beliefs, present={"a": True})


def test_replay_refuses_every_sampled_emission_without_a_key():
    system, params = model()
    sampled = dataclasses.replace(
        system,
        emissions={
            **dict(system.emissions),
            "kind_vector": Emission(("kind",), "sample", "identity"),
        },
    )
    xs, kinds = jnp.zeros((3, 2)), jnp.zeros((3,), jnp.int32)
    datum = {"kind_vector": jnp.zeros((3, 3))}  # the trace supplies the datum, so it is not fed
    with pytest.raises(ValueError, match="emission 'kind_vector' is realised by 'sample'"):
        replay(sampled, params, {"obs_x": xs, "obs_kind": kinds, **datum}, {}, metrics=False)


def test_emit_refuses_a_grid_over_an_axis_the_factor_does_not_have():
    system, params = model()
    far = Grid(shape=(2, 2), low=-2.0, high=2.0, axes=(0, 5))
    off = dataclasses.replace(
        system, emissions={**dict(system.emissions), "cell_largest": Emission(("x",), "mode", far)}
    )
    with pytest.raises(ValueError, match="every axis must be a coordinate"):
        off.emit(KEY, params, {"x": jnp.zeros(2), "kind": jnp.asarray(0)})


def test_the_document_lists_the_observed_variable_once_and_no_observing_slot():
    both, _, params = _two_slot_system()
    document, arrays = describe(both, params, arrays="inline")
    assert set(document["variables"]) & {"a", "b", "x", "mean"} == {"x", "mean"}
    assert document["factors"]["a"]["likelihood"]["observes"] == "mean"
    assert document["relations"]["observed"] == ["a", "b"]
    rebuilt, _ = build(document, arrays=arrays)
    assert rebuilt == both
    system, mparams = model()
    document, arrays = describe(system, mparams, arrays="inline")
    assert not {"read_x", "read_kind", "cell_mode", "cell_stats"} & set(document["variables"])
    assert build(document, arrays=arrays)[0] == system
