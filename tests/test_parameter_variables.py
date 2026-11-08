"""Named parameter variables: a table held once under a name, read by several factors.

Pinned by: the explicit form with one reader against the implicit form (inference, planning,
learning), the resolved positional view and the keyed params every face returns, learning that sums
the readers' statistics into the one table, the rate keys, novelty on a referencing reader, the
description length (a variable priced once under ``parameter:<name>``), the description document and
its round trip, ``merge`` with ``Identify`` and the views' write-back, and the refusals.

Design: docs/design/shared-parameters.md (the 2026-11-05 amendment); the first customer is the
two-tier field in docs/design/terrain-field.md.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Parameter
from hybrid_aif.composition import Identify, merge, subsystem, write_back
from hybrid_aif.description import build, describe
from hybrid_aif.description.mdl import description_length
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    MLPConditional,
)
from hybrid_aif.system import System, SystemParams
from tests.test_terrain_field import GRID, OBS, _field, _rover

LEVELS = 3


def _table(conc: float = 2.0) -> ConditionalDirichlet:
    """A learnable ``(3, 3, 1)`` transition table: near-diagonal counts, no control axis."""
    counts = jnp.full((LEVELS, LEVELS, 1), conc) + jnp.eye(LEVELS)[..., None]
    return ConditionalDirichlet(concentration=counts, parent_ndims=2)


def _emission() -> ConditionalCategorical:
    return ConditionalCategorical(log_table=jnp.log(0.8 * jnp.eye(LEVELS) + 0.1), parent_ndims=1)


def _dirichlet_emission(conc: float = 2.0) -> ConditionalDirichlet:
    return ConditionalDirichlet(
        concentration=jnp.full((LEVELS, LEVELS), conc) + jnp.eye(LEVELS), parent_ndims=1
    )


def _system(**kwargs) -> System:
    return System(
        A_deps=((0,), (1,)),
        B_deps=((0,), (1,)),
        controlled=(False, False),
        policy_len=2,
        modality_names=("oa", "ob"),
        factor_names=("a", "b"),
        **kwargs,
    )


def _prior() -> dict:
    return {"a": Categorical(logits=jnp.zeros(LEVELS)), "b": Categorical(logits=jnp.zeros(LEVELS))}


def _implicit() -> SystemParams:
    """Two factors with a table each: the model written without a name."""
    return SystemParams(
        likelihood={"oa": _emission(), "ob": _emission()},
        transition={"a": _table(), "b": _table()},
        initial_state_prior=_prior(),
    )


def _explicit(readers: tuple[str, ...] = ("a", "b")) -> SystemParams:
    """The same model with the table held once, as ``T``, and read by ``readers``."""
    return SystemParams(
        likelihood={"oa": _emission(), "ob": _emission()},
        transition={f: Parameter("T") if f in readers else _table() for f in ("a", "b")},
        initial_state_prior=_prior(),
        parameters={"T": _table()},
    )


OBS_AB = {"oa": jnp.asarray(1), "ob": jnp.asarray(2)}


def _tick(system: System, params: SystemParams):
    beliefs, _ = system.infer_states(params, OBS_AB)
    return beliefs


def _learn(system, params, beliefs, **kwargs):
    return system.learn(params, OBS_AB, beliefs, prev_beliefs=beliefs, **kwargs)


def _conc(decl) -> np.ndarray:
    return np.asarray(decl.concentration)


# ------------------------------------------------------------- one home, one reader


def test_one_explicit_home_and_one_reader_match_the_implicit_form():
    system = _system()
    implicit = _implicit()
    explicit = _explicit(readers=("a",))
    beliefs_i, beliefs_e = _tick(system, implicit), _tick(system, explicit)
    for name in beliefs_i:
        np.testing.assert_allclose(beliefs_i[name].probs, beliefs_e[name].probs, atol=1e-6)
    q_i, _ = system.infer_policies(implicit, beliefs_i)
    q_e, _ = system.infer_policies(explicit, beliefs_e)
    np.testing.assert_allclose(np.asarray(q_i.probs), np.asarray(q_e.probs), atol=1e-6)
    learned_i = _learn(system, implicit, beliefs_i)
    learned_e = _learn(system, explicit, beliefs_e)
    np.testing.assert_allclose(
        _conc(learned_i.transition["a"]), _conc(learned_e.parameters["T"]), atol=1e-6
    )
    np.testing.assert_allclose(
        _conc(learned_i.transition["b"]), _conc(learned_e.transition["b"]), atol=1e-6
    )


# -------------------------------------------------------------- two readers, one home


def test_the_resolved_view_holds_the_home_at_every_readers_position():
    system, params = _system(), _explicit()
    view = system._p(params)
    assert view.references == (("transition", 0, "T"), ("transition", 1, "T"))
    for position in (0, 1):
        np.testing.assert_array_equal(
            _conc(view.transition[position]), _conc(params.parameters["T"])
        )


def test_the_faces_return_references_and_one_home_never_copies():
    system, params = _system(), _explicit()
    learned = _learn(system, params, _tick(system, params))
    assert isinstance(learned.transition["a"], Parameter)
    assert isinstance(learned.transition["b"], Parameter)
    assert learned.references == ()
    assert set(learned.parameters) == {"T"}
    assert jax.tree_util.tree_leaves(learned.transition) == []
    assert not np.array_equal(_conc(learned.parameters["T"]), _conc(params.parameters["T"]))


def test_a_reference_is_a_pytree_node_with_no_array_leaves():
    assert jax.tree_util.tree_leaves(Parameter("T")) == []
    mapped = jax.tree_util.tree_map(lambda x: x + 1, {"a": Parameter("T")})
    assert mapped["a"].name == "T"
    assert Parameter("T") == Parameter("T") and Parameter("T") != Parameter("U")


def test_the_params_a_face_takes_may_be_jitted_with_references():
    system, params = _system(), _explicit()
    beliefs = _tick(system, params)
    compiled = system.compiled()
    learned = compiled.learn(params, OBS_AB, beliefs, prev_beliefs=beliefs)
    eager = _learn(system, params, beliefs)
    np.testing.assert_allclose(
        _conc(learned.parameters["T"]), _conc(eager.parameters["T"]), atol=1e-6
    )
    assert isinstance(learned.transition["a"], Parameter)


# ------------------------------------------------------------------------ learning


def test_the_home_absorbs_the_sum_of_its_readers_statistics_once():
    system, params = _system(), _explicit()
    beliefs = _tick(system, params)
    # each reader's own update, from the implicit model with the other frozen
    deltas = []
    for reader, other in (("a", "b"), ("b", "a")):
        learned = _learn(system, _implicit(), beliefs, lr_by_slot={f"transition:{other}": 0.0})
        deltas.append(_conc(learned.transition[reader]) - _conc(_table()))
    assert all(np.abs(d).sum() > 0 for d in deltas)
    learned = _learn(system, params, beliefs)
    np.testing.assert_allclose(
        _conc(learned.parameters["T"]), _conc(_table()) + deltas[0] + deltas[1], atol=1e-6
    )


def test_a_home_at_rate_zero_is_untouched_and_the_rate_is_the_homes():
    system, params = _system(), _explicit()
    beliefs = _tick(system, params)
    frozen = _learn(system, params, beliefs, lr_by_slot={"parameter:T": 0.0})
    np.testing.assert_array_equal(_conc(frozen.parameters["T"]), _conc(_table()))
    half = _learn(system, params, beliefs, lr_by_slot={"parameter:T": 0.5})
    full = _learn(system, params, beliefs)
    gained_half = _conc(half.parameters["T"]) - _conc(_table())
    gained_full = _conc(full.parameters["T"]) - _conc(_table())
    np.testing.assert_allclose(gained_half, 0.5 * gained_full, atol=1e-6)
    wild = _learn(system, params, beliefs, lr_by_slot={"*:*": 0.0})
    np.testing.assert_array_equal(_conc(wild.parameters["T"]), _conc(_table()))


def test_a_traced_rate_on_the_home_learns_under_jit():
    system, params = _system(), _explicit()
    beliefs = _tick(system, params)

    @jax.jit
    def step(rate):
        return _learn(system, params, beliefs, lr_by_slot={"parameter:T": rate}).parameters["T"]

    np.testing.assert_array_equal(_conc(step(jnp.float32(0.0))), _conc(_table()))
    assert np.abs(_conc(step(jnp.float32(1.0))) - _conc(_table())).sum() > 0


def test_a_readers_own_rate_key_takes_only_a_static_zero_and_names_the_variables_key():
    system, params = _system(), _explicit()
    beliefs = _tick(system, params)
    with pytest.raises(ValueError, match=r"transition:a.*parameter:T"):
        _learn(system, params, beliefs, lr_by_slot={"transition:a": 0.5})
    with pytest.raises(ValueError, match=r"parameter:T"):
        _learn(system, params, beliefs, decay_by_slot={"transition:b": 1.0})
    with pytest.raises(ValueError, match=r"transition:a.*nonzero.*parameter:T"):
        _learn(system, params, beliefs, lr_by_slot={"transition:*": 0.5})
    with pytest.raises(ValueError, match="unknown parameter key"):
        _learn(system, params, beliefs, lr_by_slot={"parameter:U": 1.0})
    # a fully wildcarded nonzero rate reaches the variable through parameter:*, not the readers
    learned = _learn(system, params, beliefs, lr_by_slot={"*:*": 0.5})
    gained = _conc(learned.parameters["T"]) - _conc(_table())
    full = _conc(_learn(system, params, beliefs).parameters["T"]) - _conc(_table())
    np.testing.assert_allclose(gained, 0.5 * full, atol=1e-6)


def test_a_static_zero_on_one_reader_keeps_its_statistic_out_of_the_variable():
    system, params = _system(), _explicit()
    beliefs = _tick(system, params)
    only_b = _learn(system, params, beliefs, lr_by_slot={"transition:a": 0.0})
    only_a = _learn(system, params, beliefs, lr_by_slot={"transition:b": 0.0})
    both = _learn(system, params, beliefs)
    start = _conc(_table())
    np.testing.assert_allclose(
        _conc(only_a.parameters["T"]) + _conc(only_b.parameters["T"]) - start,
        _conc(both.parameters["T"]),
        atol=1e-6,
    )
    none = _learn(system, params, beliefs, lr_by_slot={"transition:*": 0.0})
    np.testing.assert_array_equal(_conc(none.parameters["T"]), start)
    compiled = system.compiled().learn(
        params, OBS_AB, beliefs, prev_beliefs=beliefs, lr_by_slot={"transition:a": 0.0}
    )
    np.testing.assert_allclose(
        _conc(compiled.parameters["T"]), _conc(only_b.parameters["T"]), atol=1e-6
    )


def test_a_scheduler_that_observes_nothing_of_the_field_can_be_kept_out_of_its_counts():
    composite, params = _field_composite()
    obs = {"rover.tex": OBS["tex"], "rover.fix": OBS["fix"], "mission.zone": jnp.array(0)}
    beliefs, _ = composite.infer_states(params, obs, composite.initial_priors(params))
    act = {"rover.a": jnp.array([0.5, 0.0])}
    start = _conc(params.parameters["rover.field"])

    def gain(**kw):
        out = composite.learn(params, obs, beliefs, prev_beliefs=beliefs, action=act, **kw)
        return _conc(out.parameters["rover.field"]) - start

    both = gain()
    rover_only = gain(lr_by_slot={"transition:mission.map": 0.0})
    scheduler_only = gain(lr_by_slot={"transition:rover.terrain": 0.0})
    assert both.sum() == pytest.approx(2.0, abs=1e-4)
    assert rover_only.sum() == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(rover_only + scheduler_only, both, atol=1e-5)


def test_likelihood_readers_sum_into_one_table_and_an_absent_modality_adds_nothing():
    system = _system()
    params = SystemParams(
        likelihood={"oa": Parameter("A"), "ob": Parameter("A")},
        transition={"a": _table(), "b": _table()},
        initial_state_prior=_prior(),
        parameters={"A": _dirichlet_emission()},
    )
    beliefs = _tick(system, params)
    both = _learn(system, params, beliefs)
    # a modality statically absent teaches nothing: only "ob" counts
    positional = system._p(params)
    obs = system.to_positional("observations", OBS_AB)
    prev = system.to_positional("beliefs", beliefs)
    only_ob = system._learn(
        positional,
        obs,
        prev,
        prev_beliefs=prev,
        lr_by_slot={"transition:*": 0.0},
        present=(False, True),
    )
    base = _conc(_dirichlet_emission())
    add_both = _conc(both.parameters["A"]) - base
    add_ob = _conc(only_ob.parameters["A"]) - base
    assert add_both.sum() == pytest.approx(2.0, abs=1e-5)  # one unit of mass per reader
    assert add_ob.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.all(add_both - add_ob >= -1e-6)


def test_a_point_table_home_reads_and_learns_nothing():
    system = _system()
    point = ConditionalCategorical(
        log_table=jnp.log(jnp.full((LEVELS, LEVELS, 1), 1.0 / LEVELS)), parent_ndims=2
    )
    params = dataclasses.replace(_explicit(), parameters={"T": point})
    beliefs = _tick(system, params)
    learned = _learn(system, params, beliefs)
    np.testing.assert_array_equal(
        np.asarray(learned.parameters["T"].log_table), np.asarray(point.log_table)
    )


def test_a_variable_nothing_reads_is_allowed_and_untouched():
    system = _system()
    params = dataclasses.replace(_implicit(), parameters={"spare": _table()})
    learned = _learn(system, params, _tick(system, params))
    np.testing.assert_array_equal(_conc(learned.parameters["spare"]), _conc(_table()))


def test_the_carry_takes_a_learned_home_whole():
    system, fresh = _system(), _explicit()
    learned = _learn(system, fresh, _tick(system, fresh))
    carried = system.carry(fresh, learned)
    np.testing.assert_array_equal(_conc(carried.parameters["T"]), _conc(learned.parameters["T"]))
    assert isinstance(carried.transition["a"], Parameter)
    only = system.carry(fresh, learned, slots=["parameter:T"])
    np.testing.assert_array_equal(_conc(only.parameters["T"]), _conc(learned.parameters["T"]))


def test_the_readers_of_a_home_share_one_draw_in_an_environment():
    system, params = _system(), _explicit()
    _, env_params = system.as_env(params, param_key=jax.random.PRNGKey(3))
    np.testing.assert_array_equal(
        np.asarray(env_params.transition["a"]), np.asarray(env_params.transition["b"])
    )
    _, own = system.as_env(_implicit(), param_key=jax.random.PRNGKey(3))
    assert not np.array_equal(np.asarray(own.transition["a"]), np.asarray(own.transition["b"]))


# ------------------------------------------------------------------------- novelty


def test_novelty_on_a_referencing_reader_equals_the_implicit_forms():
    both = ("transition:a", "transition:b")
    system = _system(epistemic=both)
    implicit, explicit = _implicit(), _explicit()
    beliefs = _tick(system, implicit)
    _, tree_i = system.infer_policies(implicit, beliefs, metrics=True)
    _, tree_e = system.infer_policies(explicit, beliefs, metrics=True)
    for key in both:
        gain_i = tree_i["infogain"][key]
        gain_e = tree_e["infogain"][key]
        np.testing.assert_allclose(np.asarray(gain_i.lower), np.asarray(gain_e.lower), atol=1e-6)
    assert float(np.asarray(tree_e["infogain"]["transition:a"].lower).sum()) > 0.0


# ------------------------------------------------------------------ description length


_ADDRESSES = (
    ("transition", ("a", "b")),
    ("initial_state_prior", ("a", "b")),
    ("likelihood", ("oa", "ob")),
)


def test_the_home_is_priced_once_and_a_reader_pays_its_structure_only():
    system = _system()
    n = 100.0
    implicit = description_length(system, n, params=_implicit())
    one = description_length(system, n, params=_explicit(readers=("a",)))
    held = {
        **{f"{slot}:{name}": n for slot, names in _ADDRESSES for name in names},
        "parameter:T": n,  # the variable's own resolution: the readers' sum is the default
    }
    two = description_length(system, held, params=_explicit())
    assert "parameter:T" in two.per_address and "parameter:T" in one.per_address
    assert sum(1 for key in two.per_address if key == "parameter:T") == 1
    for key in ("transition:a", "transition:b"):
        link = two.per_address[key]
        assert link.family is None and link.prior is None and link.posterior is None
        assert link.graph > 0.0
    # one reader of T and one own table is the implicit model, to the nat
    assert one.total == pytest.approx(implicit.total, abs=1e-9)
    # a second reader adds its structure and no second table
    second = two.per_address["transition:b"]
    own = one.per_address["transition:b"]
    assert two.total == pytest.approx(
        one.total - (own.structure + own.parameters) + second.graph, abs=1e-6
    )
    assert ("transition:a", "family") not in two.unscored
    assert ("parameter:T", "posterior") in two.unscored  # no prior= given: the link is unscored


def test_a_homes_resolution_is_the_sum_of_its_readers():
    system = _system()
    explicit = _explicit()
    by_reader = description_length(
        system, {"transition:a": 30.0, "transition:b": 70.0}, params=explicit
    )
    assert by_reader.per_address["parameter:T"].n == pytest.approx(100.0)
    named = description_length(system, {"transition:a": 30.0, "parameter:T": 55.0}, params=explicit)
    assert named.per_address["parameter:T"].n == pytest.approx(55.0)


def test_the_priors_reference_mirrors_the_posteriors():
    system = _system()
    prior = _explicit()
    beliefs = _tick(system, prior)
    posterior = _learn(system, prior, beliefs)
    priced = description_length(system, 100.0, params=posterior, prior=prior)
    assert priced.per_address["parameter:T"].posterior > 0.0
    assert priced.per_address["transition:a"].posterior is None


# ----------------------------------------------------------- the description document


def test_the_document_records_each_home_once_and_each_reader_by_name():
    system, params = _system(), _explicit()
    doc, arrays = describe(system, params)
    assert list(doc["parameters"]) == ["T"]
    assert doc["parameters"]["T"]["family"] == "ConditionalCategorical"
    for reader in ("a", "b"):
        record = doc["factors"][reader]["transition"]
        assert record["parameter"] == "T" and "family" not in record
        assert record["parents"] == [reader]
    assert sum(1 for v in doc["variables"] if v.endswith("concentration")) == 1


def test_an_implicit_models_document_has_no_parameters_entry():
    doc, _ = describe(_system(), _implicit())
    assert "parameters" not in doc
    assert all("parameter" not in block.get("transition", {}) for block in doc["factors"].values())


def test_a_document_with_references_builds_and_describes_identically():
    system, params = _system(), _explicit()
    doc, arrays = describe(system, params)
    rebuilt_system, rebuilt = build(doc, arrays=arrays)
    assert isinstance(rebuilt.transition["a"], Parameter)
    assert rebuilt.transition["a"].name == "T" and set(rebuilt.parameters) == {"T"}
    np.testing.assert_allclose(
        _conc(rebuilt.parameters["T"]), _conc(params.parameters["T"]), atol=1e-6
    )
    again, _ = describe(rebuilt_system, rebuilt)
    assert json.dumps(again, sort_keys=True) == json.dumps(doc, sort_keys=True)


def test_a_novelty_term_on_each_reader_round_trips_as_one_term_per_reader():
    system = _system(epistemic=("transition:a", "transition:b"))
    doc, arrays = describe(system, _explicit())
    assert [t["on"] for t in doc["objective"]["terms"]] == ["P_T", "P_T"]
    rebuilt_system, _ = build(doc, arrays=arrays)
    assert rebuilt_system.epistemic == ("transition:a", "transition:b")


def test_a_variable_no_record_reads_round_trips():
    system = _system()
    params = dataclasses.replace(_implicit(), parameters={"spare": _table()})
    doc, arrays = describe(system, params)
    _, rebuilt = build(doc, arrays=arrays)
    assert set(rebuilt.parameters) == {"spare"}
    np.testing.assert_allclose(_conc(rebuilt.parameters["spare"]), _conc(_table()), atol=1e-6)


def test_the_schema_accepts_a_reference_and_refuses_a_record_with_both():
    jsonschema = pytest.importorskip("jsonschema")
    from hybrid_aif.description import schema

    doc, _ = describe(_system(), _explicit())
    validator = jsonschema.Draft202012Validator(schema())
    validator.validate(doc)
    both = json.loads(json.dumps(doc))
    both["factors"]["a"]["transition"]["family"] = "ConditionalCategorical"
    assert list(validator.iter_errors(both))


def test_a_document_that_names_an_undeclared_variable_is_refused():
    from hybrid_aif.description import DescriptionError

    doc, arrays = describe(_system(), _explicit())
    broken = json.loads(json.dumps(doc))
    broken["factors"]["a"]["transition"]["parameter"] = "missing"
    with pytest.raises(DescriptionError, match="missing.*do not declare"):
        build(broken, arrays=arrays)


# ------------------------------------------------------------ the field's two readers


def _field_home() -> ConditionalDirichlet:
    return ConditionalDirichlet(concentration=_field(), parent_ndims=1)


def _rover_reading_the_field(cells: int | None = None):
    """The terrain-field rover with its terrain table held as the variable ``field``."""
    system, params = _rover()
    home = params.transition["terrain"] if cells is None else _field_of(cells)
    params = dataclasses.replace(
        params,
        transition={**params.transition, "terrain": Parameter("field")},
        parameters={"field": home},
    )
    return system, params


def _field_of(cells: int) -> ConditionalDirichlet:
    return ConditionalDirichlet(concentration=jnp.ones((LEVELS, cells)), parent_ndims=1)


def _mission(sites: int = GRID.n_cells):
    """A scheduler whose ``map`` factor reads the field through a categorical ``visit`` parent
    with one level per cell."""
    system = System(
        A_deps=((0,),),
        B_deps=((), (0,)),
        factor_names=("visit", "map"),
        modality_names=("zone",),
        policy_len=2,
        planner="gradient",
        mixture_k=3,
        mixture_schedule="mode_aligned",
    )
    params = SystemParams(
        likelihood={
            "zone": ConditionalCategorical(
                log_table=jnp.log(jnp.full((2, sites), 0.5)), parent_ndims=1
            )
        },
        transition={
            "visit": ConditionalCategorical(
                log_table=jnp.log(jnp.full((sites, 1), 1.0 / sites)), parent_ndims=1
            ),
            "map": Parameter("field"),
        },
        initial_state_prior={
            "visit": Categorical(logits=jnp.zeros(sites)),
            "map": Categorical(logits=jnp.zeros(LEVELS)),
        },
        parameters={"field": _field_home()},
    )
    return system, params


def _field_composite():
    rover, rover_params = _rover_reading_the_field()
    mission, mission_params = _mission()
    return merge(
        {"rover": rover, "mission": mission},
        {"rover": rover_params, "mission": mission_params},
        [Identify("rover.field", "mission.field")],
    )


def test_a_gridded_reader_and_a_categorical_reader_share_one_field():
    rover, rover_params = _rover_reading_the_field()
    mission, mission_params = _mission()
    rover._validate(rover_params)
    mission._validate(mission_params)
    assert GRID.n_cells == 9 and _conc(mission_params.parameters["field"]).shape == (3, 9)


def test_a_cell_count_that_does_not_match_the_grid_refuses_and_names_both():
    rover, rover_params = _rover_reading_the_field(cells=4)
    with pytest.raises(
        ValueError, match=r"4 entries.*9 cells.*'terrain' reads parameter .*'field'"
    ):
        rover._validate(rover_params)


def test_a_reader_whose_parents_do_not_match_the_homes_axes_names_reader_and_home():
    system = System(
        A_deps=((0,),),
        B_deps=((0, 1), (1,)),
        controlled=(False, False),
        factor_names=("a", "b"),
        modality_names=("oa",),
        policy_len=1,
    )
    one_parent = ConditionalDirichlet(concentration=jnp.ones((LEVELS, LEVELS)), parent_ndims=1)
    params = SystemParams(
        likelihood={"oa": _emission()},
        transition={"a": Parameter("T"), "b": _table()},
        initial_state_prior=_prior(),
        parameters={"T": one_parent},
    )
    with pytest.raises(
        ValueError, match=r"factor 'a'.*B_deps declares 2.*reads parameter variable 'T'"
    ):
        system._validate(params)


def test_the_composite_keeps_one_field_that_both_members_read():
    composite, params = _field_composite()
    assert list(params.parameters) == ["rover.field"]
    assert isinstance(params.transition["rover.terrain"], Parameter)
    assert params.transition["rover.terrain"].name == "rover.field"
    assert params.transition["mission.map"].name == "rover.field"
    view = composite._p(params)
    assert len(view.references) == 2 and {h for _, _, h in view.references} == {"rover.field"}
    doc, _ = describe(composite, params)
    assert list(doc["parameters"]) == ["rover.field"]
    readers = [
        block["transition"]["parameter"]
        for block in doc["factors"].values()
        if "parameter" in block.get("transition", {})
    ]
    assert sorted(readers) == ["rover.field", "rover.field"]


def test_a_composite_with_a_shared_variable_describes_and_builds_with_one_home():
    composite, params = _field_composite()
    doc, arrays = describe(composite, params)
    rebuilt_system, rebuilt = build(doc, arrays=arrays)
    assert list(rebuilt.parameters) == ["rover.field"]
    assert rebuilt.transition["rover.terrain"].name == "rover.field"
    assert rebuilt.transition["mission.map"].name == "rover.field"
    np.testing.assert_allclose(
        _conc(rebuilt.parameters["rover.field"]).reshape(LEVELS, -1),
        _conc(params.parameters["rover.field"]),
        atol=1e-6,
    )
    assert rebuilt_system.subsystems is not None
    view, view_params = subsystem(rebuilt_system, rebuilt, "mission")
    assert list(view_params.parameters) == ["rover.field"]


def test_the_composites_novelty_on_the_scheduler_reader_reads_the_shared_field():
    rover, rover_params = _rover_reading_the_field()
    mission, mission_params = _mission()
    mission = dataclasses.replace(mission, epistemic=("transition:map",))
    composite, params = merge(
        {"rover": rover, "mission": mission},
        {"rover": rover_params, "mission": mission_params},
        [Identify("rover.field", "mission.field")],
    )
    obs = {
        "rover.tex": OBS["tex"],
        "rover.fix": OBS["fix"],
        "mission.zone": jnp.array(0),
    }
    beliefs, _ = composite.infer_states(params, obs, composite.initial_priors(params))
    _, tree = composite.infer_policies(params, beliefs, metrics=True)
    assert "transition:mission.map" in tree["infogain"]
    assert "transition:rover.terrain" in tree["infogain"]


def _view_tick(view, view_params, obs, action=None, **rates):
    beliefs, _ = view.infer_states(view_params, obs, view.initial_priors(view_params))
    return view.learn(view_params, obs, beliefs, prev_beliefs=beliefs, action=action, **rates)


ROVER_OBS = {"rover.tex": OBS["tex"], "rover.fix": OBS["fix"]}
ROVER_ACTION = {"rover.a": jnp.array([0.5, 0.0])}


def test_learning_in_one_view_and_writing_back_updates_the_field_the_other_view_reads():
    composite, params = _field_composite()
    rover_view, rover_params = subsystem(composite, params, "rover")
    _, mission_params = subsystem(composite, params, "mission")
    assert list(rover_params.parameters) == list(mission_params.parameters) == ["rover.field"]
    assert rover_params.transition["rover.terrain"].name == "rover.field"
    start = _conc(params.parameters["rover.field"])
    learned = _view_tick(
        rover_view,
        rover_params,
        ROVER_OBS,
        ROVER_ACTION,
        lr_by_slot={"*:*": 0.0, "parameter:rover.field": 1.0},
    )
    moved = _conc(learned.parameters["rover.field"])
    assert np.abs(moved - start).sum() > 0.0
    written = write_back(composite, params, "rover", rover_params, learned)
    np.testing.assert_allclose(_conc(written.parameters["rover.field"]), moved, atol=1e-6)
    assert isinstance(written.transition["mission.map"], Parameter)
    _, remission = subsystem(composite, written, "mission")
    np.testing.assert_allclose(_conc(remission.parameters["rover.field"]), moved, atol=1e-6)
    np.testing.assert_array_equal(_conc(mission_params.parameters["rover.field"]), start)


def test_a_view_taken_before_another_write_back_is_refused_not_merged():
    composite, params = _field_composite()
    rover_view, rover_params = subsystem(composite, params, "rover")
    mission_view, mission_params = subsystem(composite, params, "mission")
    rates = {"*:*": 0.0, "parameter:rover.field": 1.0}
    rover_after = _view_tick(rover_view, rover_params, ROVER_OBS, ROVER_ACTION, lr_by_slot=rates)
    mission_after = _view_tick(
        mission_view, mission_params, {"mission.zone": jnp.array(0)}, lr_by_slot=rates
    )
    once = write_back(composite, params, "rover", rover_params, rover_after)
    with pytest.raises(ValueError, match=r"Take the view again.*learning on the composite"):
        write_back(composite, once, "mission", mission_params, mission_after)


def test_views_taken_in_turn_write_back_one_after_the_other():
    composite, params = _field_composite()
    start = _conc(params.parameters["rover.field"])
    rates = {"*:*": 0.0, "parameter:rover.field": 1.0}
    rover_view, rover_params = subsystem(composite, params, "rover")
    rover_after = _view_tick(rover_view, rover_params, ROVER_OBS, ROVER_ACTION, lr_by_slot=rates)
    params = write_back(composite, params, "rover", rover_params, rover_after)
    gained_rover = _conc(rover_after.parameters["rover.field"]) - start
    mission_view, mission_params = subsystem(composite, params, "mission")
    mission_after = _view_tick(
        mission_view, mission_params, {"mission.zone": jnp.array(0)}, lr_by_slot=rates
    )
    params = write_back(composite, params, "mission", mission_params, mission_after)
    gained_mission = _conc(mission_after.parameters["rover.field"]) - (start + gained_rover)
    assert gained_rover.sum() > 0.5 and gained_mission.sum() > 0.5
    np.testing.assert_allclose(
        _conc(params.parameters["rover.field"]), start + gained_rover + gained_mission, atol=1e-5
    )


def test_a_view_learning_under_decay_writes_back_what_the_composite_would_learn():
    composite, params = _field_composite()
    anchored = ConditionalDirichlet(
        concentration=_field(), parent_ndims=1, alpha0=jnp.ones_like(_field())
    )
    params = dataclasses.replace(params, parameters={"rover.field": anchored})
    obs = {"rover.tex": OBS["tex"], "rover.fix": OBS["fix"], "mission.zone": jnp.array(0)}
    beliefs, _ = composite.infer_states(params, obs, composite.initial_priors(params))
    rates = {"*:*": 0.0, "parameter:rover.field": 1.0}
    reference = composite.learn(
        params,
        obs,
        beliefs,
        prev_beliefs=beliefs,
        action=ROVER_ACTION,
        lr_by_slot={"*:*": 0.0, "parameter:rover.field": 1.0, "transition:mission.map": 0.0},
        decay=0.5,
    )
    rover_view, rover_params = subsystem(composite, params, "rover")
    vbeliefs, _ = rover_view.infer_states(
        rover_params, ROVER_OBS, rover_view.initial_priors(rover_params)
    )
    after = rover_view.learn(
        rover_params,
        ROVER_OBS,
        vbeliefs,
        prev_beliefs=vbeliefs,
        action=ROVER_ACTION,
        lr_by_slot=rates,
        decay=0.5,
    )
    written = write_back(composite, params, "rover", rover_params, after)
    np.testing.assert_allclose(
        _conc(written.parameters["rover.field"]),
        _conc(reference.parameters["rover.field"]),
        atol=1e-4,
    )


def test_write_back_refuses_a_missing_variable_and_a_non_positive_concentration():
    composite, params = _field_composite()
    view, view_params = subsystem(composite, params, "mission")
    bare = dataclasses.replace(view_params, parameters={})
    with pytest.raises(ValueError, match=r"held parameter variable 'rover.field'"):
        write_back(composite, params, "mission", view_params, bare)
    broken = dataclasses.replace(
        view_params,
        parameters={
            "rover.field": ConditionalDirichlet(
                concentration=_conc(params.parameters["rover.field"]) * 0.0 - 1.0,
                parent_ndims=1,
            )
        },
    )
    with pytest.raises(ValueError, match="not positive"):
        write_back(composite, params, "mission", view_params, broken)


def test_write_back_replaces_a_point_table_only_when_the_view_changed_it():
    composite, params = _field_composite()
    point = ConditionalCategorical(log_table=jnp.zeros((LEVELS, GRID.n_cells)), parent_ndims=1)
    changed = ConditionalCategorical(log_table=jnp.ones((LEVELS, GRID.n_cells)), parent_ndims=1)
    held = dataclasses.replace(params, parameters={"rover.field": point})
    before = dataclasses.replace(held, parameters={"rover.field": point})
    same = write_back(composite, held, "rover", before, before)
    np.testing.assert_array_equal(
        np.asarray(same.parameters["rover.field"].log_table), np.asarray(point.log_table)
    )
    after = dataclasses.replace(held, parameters={"rover.field": changed})
    moved = write_back(composite, held, "rover", before, after)
    np.testing.assert_array_equal(
        np.asarray(moved.parameters["rover.field"].log_table), np.asarray(changed.log_table)
    )


def test_write_back_refuses_a_view_taken_from_a_table_of_another_shape():
    composite, params = _field_composite()
    stale = dataclasses.replace(params, parameters={"rover.field": _field_of(5)})
    with pytest.raises(ValueError, match=r"take the view again"):
        write_back(composite, params, "rover", stale, stale)


def test_identifying_variables_of_different_shape_or_family_refuses_naming_both():
    rover, rover_params = _rover_reading_the_field()
    mission, mission_params = _mission()
    taller = ConditionalDirichlet(concentration=jnp.ones((4, 9)), parent_ndims=1)
    bad = dataclasses.replace(mission_params, parameters={"field": taller})
    with pytest.raises(ValueError, match=r"rover\.field.*\(3, 9\).*mission\.field.*\(4, 9\)"):
        merge(
            {"rover": rover, "mission": mission},
            {"rover": rover_params, "mission": bad},
            [Identify("rover.field", "mission.field")],
        )
    point = ConditionalCategorical(log_table=jnp.zeros((3, 9)), parent_ndims=1)
    other = dataclasses.replace(mission_params, parameters={"field": point})
    with pytest.raises(ValueError, match=r"ConditionalDirichlet.*ConditionalCategorical"):
        merge(
            {"rover": rover, "mission": mission},
            {"rover": rover_params, "mission": other},
            [Identify("rover.field", "mission.field")],
        )
    with pytest.raises(ValueError, match=r"'mission\.nothing'.*not a parameter variable"):
        merge(
            {"rover": rover, "mission": mission},
            {"rover": rover_params, "mission": mission_params},
            [Identify("rover.field", "mission.nothing")],
        )


def test_without_the_binding_the_composite_keeps_two_fields():
    rover, rover_params = _rover_reading_the_field()
    mission, mission_params = _mission()
    _, params = merge(
        {"rover": rover, "mission": mission},
        {"rover": rover_params, "mission": mission_params},
    )
    assert sorted(params.parameters) == ["mission.field", "rover.field"]


# ------------------------------------------------------------------------- refusals


def test_a_reference_to_a_variable_that_is_not_declared_names_reader_and_variable():
    system = _system()
    params = dataclasses.replace(_explicit(), parameters={})
    with pytest.raises(ValueError, match=r"factor 'a' reads parameter variable 'T'"):
        system.infer_states(params, OBS_AB)


def test_a_reference_to_a_gradient_home_is_not_built_yet():
    system = _system()
    net = MLPConditional.from_shapes((LEVELS, LEVELS), LEVELS, key=jax.random.PRNGKey(0))
    params = dataclasses.replace(_explicit(), parameters={"T": net})
    with pytest.raises(NotImplementedError, match="tabular declarations.*gradient"):
        system.infer_states(params, OBS_AB)


def test_a_variable_that_is_itself_a_reference_is_refused():
    system = _system()
    params = dataclasses.replace(_explicit(), parameters={"T": Parameter("U")})
    with pytest.raises(ValueError, match="itself a reference"):
        system.infer_states(params, OBS_AB)


def test_one_declaration_object_under_two_keys_still_raises_and_points_at_the_variable():
    shared = _table()
    with pytest.raises(
        ValueError,
        match=r"transition\['a'\] and transition\['b'\] hold one declaration object.*Parameter",
    ):
        SystemParams(
            likelihood={"oa": _emission(), "ob": _emission()},
            transition={"a": shared, "b": shared},
            initial_state_prior=_prior(),
        )


def test_a_parameter_needs_a_name():
    with pytest.raises(ValueError, match="non-empty name"):
        Parameter("")


def test_a_growable_factor_cannot_read_a_variable():
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        factor_names=("s",),
        modality_names=("o",),
        policy_len=1,
    )
    growable = ConditionalDirichlet(
        concentration=jnp.ones((LEVELS, 1)), parent_ndims=1, growth=jnp.asarray(0.5)
    )
    params = SystemParams(
        likelihood={"o": _emission()},
        transition={"s": Parameter("T")},
        initial_state_prior={"s": Categorical(logits=jnp.zeros(LEVELS))},
        parameters={"T": growable},
    )
    from hybrid_aif.learning.structure import growable_factors

    with pytest.raises(NotImplementedError, match="named parameter variable"):
        growable_factors(system, system._p(params))


def test_a_binding_that_rewrites_a_readers_control_axis_is_refused_not_applied():
    from hybrid_aif.composition import Clamp

    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        factor_names=("s",),
        modality_names=("o",),
        policy_len=1,
    )
    table = ConditionalDirichlet(concentration=jnp.ones((LEVELS, LEVELS, 2)), parent_ndims=2)
    params = SystemParams(
        likelihood={"o": _emission()},
        transition={"s": Parameter("T")},
        initial_state_prior={"s": Categorical(logits=jnp.zeros(LEVELS))},
        parameters={"T": table},
    )
    with pytest.raises(NotImplementedError, match=r"control axis of factor 'm.s'.*'m.T'"):
        merge({"m": system}, {"m": params}, [Clamp("m.u_s", 0)])


# ------------------------------------------------------------------- the fix-pass defects


def _shared_across_kinds(order=("u", "v")):
    """A table read by a transition of one member and a likelihood of another."""

    def table():
        return ConditionalDirichlet(
            concentration=jnp.full((LEVELS, LEVELS), 2.0) + jnp.eye(LEVELS), parent_ndims=1
        )

    reader_t = System(
        A_deps=((0,),),
        B_deps=((0,), (0,)),
        controlled=(False, False),
        policy_len=1,
        factor_names=("s", "t"),
        modality_names=("os",),
    )
    params_t = SystemParams(
        likelihood={"os": _emission()},
        transition={"s": _table(), "t": Parameter("H")},
        initial_state_prior=_prior_of("s", "t"),
        parameters={"H": table()},
    )
    reader_o = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(False,),
        policy_len=1,
        factor_names=("z",),
        modality_names=("o",),
    )
    params_o = SystemParams(
        likelihood={"o": Parameter("H")},
        transition={"z": _table()},
        initial_state_prior=_prior_of("z"),
        parameters={"H": table()},
    )
    members = {"u": (reader_t, params_t), "v": (reader_o, params_o)}
    ordered = {k: members[k] for k in order}
    return merge(
        {k: v[0] for k, v in ordered.items()},
        {k: v[1] for k, v in ordered.items()},
        [Identify(f"{order[0]}.H", f"{order[1]}.H")],
    )


def _prior_of(*names):
    return {n: Categorical(logits=jnp.zeros(LEVELS)) for n in names}


@pytest.mark.parametrize("order", [("u", "v"), ("v", "u")])
def test_a_variable_read_by_a_transition_and_a_likelihood_rebuilds_from_its_own_record(order):
    composite, params = _shared_across_kinds(order)
    obs = {"u.os": jnp.asarray(1), "v.o": jnp.asarray(2)}
    beliefs, _ = composite.infer_states(params, obs, composite.initial_priors(params))
    doc, arrays = describe(composite, params)
    rebuilt_system, rebuilt = build(doc, arrays=arrays)
    home = f"{order[0]}.H"
    assert _conc(rebuilt.parameters[home]).shape == (LEVELS, LEVELS)
    again, _ = rebuilt_system.infer_states(rebuilt, obs, rebuilt_system.initial_priors(rebuilt))
    for name in beliefs:
        np.testing.assert_allclose(beliefs[name].probs, again[name].probs, atol=1e-6)


def test_a_parents_level_count_is_checked_against_every_tabular_table():
    mission, params = _mission(sites=1)
    with pytest.raises(
        ValueError,
        match=r"factor 'map'.*'visit' has 1 levels.*9 entries|"
        r"factor 'map'.*9 entries.*'visit' has 1 levels",
    ):
        mission._validate(params)
    own = dataclasses.replace(
        params, transition={**params.transition, "map": _field_home()}, parameters={}
    )
    with pytest.raises(ValueError, match=r"factor 'map'.*'visit' has 1 levels"):
        mission._validate(own)
    system = _system()
    wrong = dataclasses.replace(
        _implicit(),
        likelihood={
            "oa": _dirichlet_emission(),
            "ob": ConditionalCategorical(
                log_table=jnp.log(jnp.full((LEVELS, 5), 1.0 / LEVELS)), parent_ndims=1
            ),
        },
    )
    with pytest.raises(ValueError, match=r"modality 'ob'.*5 entries.*'b' has 3 levels"):
        system._validate(wrong)
    big = dataclasses.replace(params, parameters={"field": _field_of(4)})
    mission4, _ = _mission(sites=4)
    with pytest.raises(ValueError, match=r"factor 'map'.*4 entries"):
        mission4._validate(dataclasses.replace(big, likelihood=params.likelihood))


def test_a_variables_price_does_not_depend_on_the_order_of_its_readers():
    dead = jnp.full((LEVELS, LEVELS, 1), 2.0).at[2].set(0.0)
    c_table = ConditionalDirichlet(concentration=dead, parent_ndims=2)
    c_prior = Categorical(logits=jnp.log(jnp.array([0.5, 0.5, 0.0])))
    totals = []
    for names in (("x", "y", "c"), ("y", "x", "c")):
        deps = {"x": (2,), "c": (2,), "y": (names.index("y"),)}
        system = System(
            A_deps=((0,), (1,), (2,)),
            B_deps=tuple(deps[n] for n in names),
            controlled=(False,) * 3,
            policy_len=1,
            factor_names=names,
            modality_names=("o0", "o1", "o2"),
        )
        params = SystemParams(
            likelihood={m: _emission() for m in ("o0", "o1", "o2")},
            transition={"x": Parameter("T"), "y": Parameter("T"), "c": c_table},
            initial_state_prior={"x": _prior()["a"], "y": _prior()["a"], "c": c_prior},
            parameters={"T": _table()},
        )
        link = description_length(system, 10.0, params=params).per_address["parameter:T"]
        totals.append((link.family, link.prior))
    assert totals[0] == pytest.approx(totals[1])


def test_carry_between_a_reference_and_a_table_raises_naming_the_slot():
    system = _system()
    explicit = _explicit()
    learned_explicit = _learn(system, explicit, _tick(system, explicit))
    with pytest.raises(
        ValueError, match=r"transition:a reads its own table.*parameter variable 'T'"
    ):
        system.carry(_implicit(), learned_explicit)
    learned_implicit = _learn(system, _implicit(), _tick(system, _implicit()))
    with pytest.raises(ValueError, match=r"transition:a reads parameter variable 'T'.*own table"):
        system.carry(explicit, learned_implicit)


def test_one_declaration_object_may_not_sit_under_two_names_across_containers():
    table = _table()
    with pytest.raises(ValueError, match=r"transition\['a'\] and parameters\['T'\] hold one"):
        SystemParams(
            likelihood={"oa": _emission(), "ob": _emission()},
            transition={"a": table, "b": Parameter("T")},
            initial_state_prior=_prior(),
            parameters={"T": table},
        )
    with pytest.raises(ValueError, match=r"parameters\['T'\] and parameters\['U'\] hold one"):
        SystemParams(
            likelihood={"oa": _emission(), "ob": _emission()},
            transition={"a": Parameter("T"), "b": Parameter("U")},
            initial_state_prior=_prior(),
            parameters={"T": table, "U": table},
        )
    shared = _dirichlet_emission()
    with pytest.raises(ValueError, match=r"likelihood\['oa'\] and parameters\['A'\] hold one"):
        SystemParams(
            likelihood={"oa": shared, "ob": Parameter("A")},
            transition={"a": _table(), "b": _table()},
            initial_state_prior=_prior(),
            parameters={"A": shared},
        )


def test_a_posterior_is_priced_against_a_prior_of_the_same_form_only():
    system = _system()
    explicit, implicit = _explicit(readers=("a",)), _implicit()
    learned = _learn(system, explicit, _tick(system, explicit))
    with pytest.raises(ValueError, match=r"transition:a reads parameter variable 'T'.*own table"):
        description_length(system, 10.0, params=learned, prior=implicit)
    learned_implicit = _learn(system, implicit, _tick(system, implicit))
    with pytest.raises(ValueError, match=r"transition:a reads its own table.*parameter variable"):
        description_length(system, 10.0, params=learned_implicit, prior=explicit)


def test_a_composite_namespaces_a_parameter_rate_key_like_any_other():
    from hybrid_aif.composition import namespaced_key

    assert namespaced_key("rover", "parameter:field") == "parameter:rover.field"
    composite, params = _field_composite()
    beliefs, _ = composite.infer_states(
        params,
        {"rover.tex": OBS["tex"], "rover.fix": OBS["fix"], "mission.zone": jnp.array(0)},
        composite.initial_priors(params),
    )
    key = namespaced_key("rover", "parameter:field")
    out = composite.learn(
        params,
        {"rover.tex": OBS["tex"], "rover.fix": OBS["fix"], "mission.zone": jnp.array(0)},
        beliefs,
        prev_beliefs=beliefs,
        action={"rover.a": jnp.array([0.5, 0.0])},
        lr_by_slot={key: 0.0},
    )
    np.testing.assert_array_equal(
        _conc(out.parameters["rover.field"]), _conc(params.parameters["rover.field"])
    )


def test_the_positional_form_refuses_an_edit_made_at_a_readers_position():
    import equinox as eqx

    system, params = _system(), _explicit()
    positional = system.to_positional("params", params)
    back = system.from_positional("params", positional)
    assert isinstance(back.transition["a"], Parameter)
    edited = eqx.tree_at(
        lambda p: p.transition[0].concentration,
        positional,
        positional.transition[0].concentration * 2,
    )
    with pytest.raises(
        ValueError, match=r"every reader\s+position.*parameters\['T'\] in the keyed form"
    ):
        system.from_positional("params", edited)
    moved = eqx.tree_at(
        lambda p: p.parameters["T"].concentration,
        positional,
        positional.parameters["T"].concentration * 2,
    )
    with pytest.raises(
        ValueError, match=r"every reader\s+position.*parameters\['T'\] in the keyed form"
    ):
        system.from_positional("params", moved)


@pytest.mark.parametrize("compiled", [False, True])
def test_a_literal_key_outranks_a_pattern_for_the_readers_it_would_exclude(compiled):
    system, params = _system(), _explicit()
    beliefs = _tick(system, params)
    face = system.compiled().learn if compiled else system.learn

    def gain(rates):
        out = face(params, OBS_AB, beliefs, prev_beliefs=beliefs, lr_by_slot=rates)
        return _conc(out.parameters["T"]) - _conc(_table())

    full = gain({})
    only_b = gain({"transition:a": 0.0})
    assert full.sum() > 0 and 0 < only_b.sum() < full.sum()
    np.testing.assert_allclose(gain({"*:*": 0.0, "parameter:T": 1.0}), full, atol=1e-6)
    np.testing.assert_allclose(gain({"transition:*": 0.0, "parameter:T": 1.0}), full, atol=1e-6)
    np.testing.assert_array_equal(gain({"*:*": 0.0}), np.zeros_like(full))
    np.testing.assert_allclose(gain({"parameter:T": 1.0, "transition:a": 0.0}), only_b, atol=1e-6)
    np.testing.assert_allclose(
        gain({"*:*": 0.0, "parameter:T": 1.0, "transition:a": 0.0}), only_b, atol=1e-6
    )
