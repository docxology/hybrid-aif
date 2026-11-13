"""A slow table's reading pooled over an option level: the novelty term scaled by a declared count.

Pinned by: the default (all-ones counts, or none) leaving every term unchanged, a count of n scaling
the table's novelty leaf and nothing else, the count following the level chosen at each step, a
table reached through a parameter variable, the validation refusals, ``merge`` and the views, and
the description document's round trip.

Design: docs/design/step-durations.md T10.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Parameter
from hybrid_aif.composition import Clamp, Link, joint_planner, merge, subsystem
from hybrid_aif.description import build, describe
from hybrid_aif.distributions import Categorical, ConditionalCategorical, ConditionalDirichlet
from hybrid_aif.system import System, SystemParams
from tests.test_terrain_field import _rover

KEY = "transition:s"


def _model(readings=(), counts=None, shared=False, epistemic=(KEY,)):
    """One factor ``s`` read through a controlled Dirichlet transition (two options), one
    modality; the table may be held as a parameter variable."""
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=2,
        state_names=("s",),
        modality_names=("o",),
        epistemic=epistemic,
        readings=readings,
    )
    table = ConditionalDirichlet(concentration=jnp.full((3, 3, 2), 2.0), parent_ndims=2)
    params = SystemParams(
        likelihood={
            "o": ConditionalCategorical(log_table=jnp.log(0.8 * jnp.eye(3) + 0.1), parent_ndims=1)
        },
        transition={"s": Parameter("T") if shared else table},
        initial_state_prior={"s": Categorical(logits=jnp.zeros(3))},
        parameters={"T": table} if shared else {},
        readings={} if counts is None else {"s": jnp.asarray(counts)},
    )
    return system, params


def _plan(system, params):
    beliefs, _ = system.infer_states(params, {"o": jnp.asarray(0)})
    return system.infer_policies(params, beliefs, metrics=True)


def _leaf(tree):
    return np.asarray(tree["infogain"][KEY].lower)


def _policies(system, params):
    return np.asarray(system.policies(params))[..., 0]  # (P, T)


def test_counts_of_one_or_no_declaration_leave_every_term_unchanged():
    plain = _plan(*_model())
    ones = _plan(*_model((("s", "u_s"),), [1.0, 1.0]))
    np.testing.assert_allclose(np.asarray(plain[0].probs), np.asarray(ones[0].probs), atol=1e-7)
    for term in plain[1]:
        for key in plain[1][term]:
            np.testing.assert_allclose(
                np.asarray(plain[1][term][key].lower),
                np.asarray(ones[1][term][key].lower),
                atol=1e-7,
            )


def test_a_count_scales_the_novelty_leaf_of_its_table_and_no_other_term():
    system, params = _model((("s", "u_s"),), [4.0, 4.0])
    _, base = _plan(*_model())
    _, scaled = _plan(system, params)
    np.testing.assert_allclose(_leaf(scaled), 4.0 * _leaf(base), rtol=1e-6)
    for term in base:
        for key in base[term]:
            if (term, key) != ("infogain", KEY):
                np.testing.assert_allclose(
                    np.asarray(scaled[term][key].lower),
                    np.asarray(base[term][key].lower),
                    atol=1e-7,
                )


def test_the_count_follows_the_level_chosen_at_each_step():
    system, params = _model((("s", "u_s"),), [3.0, 5.0])
    _, base = _plan(*_model())
    _, scaled = _plan(system, params)
    levels = _policies(system, params)
    per_step = np.asarray(_leaf(base)) / levels.shape[1]  # equal steps under a flat belief
    expected = np.array([sum([3.0, 5.0][u] for u in policy) for policy in levels]) * per_step
    np.testing.assert_allclose(_leaf(scaled), expected, rtol=1e-5)
    assert len(set(np.round(_leaf(scaled), 5))) == 3  # [0,0], the mixed pair, [1,1]


def test_a_table_reached_through_a_parameter_variable_is_scaled_the_same():
    own = _plan(*_model((("s", "u_s"),), [3.0, 5.0]))
    shared = _plan(*_model((("s", "u_s"),), [3.0, 5.0], shared=True))
    np.testing.assert_allclose(_leaf(own[1]), _leaf(shared[1]), rtol=1e-6)


def test_a_likelihood_table_is_scaled_by_the_count_of_its_level():
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=1,
        state_names=("s",),
        modality_names=("o",),
        epistemic=("likelihood:o",),
        readings=(("o", "u_s"),),
    )
    base = SystemParams(
        likelihood={"o": ConditionalDirichlet(concentration=jnp.full((3, 3), 2.0), parent_ndims=1)},
        transition={
            "s": ConditionalCategorical(
                log_table=jnp.log(jnp.full((3, 3, 2), 1.0 / 3.0)), parent_ndims=2
            )
        },
        initial_state_prior={"s": Categorical(logits=jnp.zeros(3))},
    )
    params = dataclasses.replace(base, readings={"o": jnp.asarray([2.0, 7.0])})
    beliefs, _ = system.infer_states(params, {"o": jnp.asarray(0)})
    _, tree = system.infer_policies(params, beliefs, metrics=True)
    unscaled = dataclasses.replace(system, readings=())
    _, ref = unscaled.infer_policies(base, beliefs, metrics=True)
    scale = np.array([2.0, 7.0])[np.asarray(system.policies(params))[:, 0, 0]]
    np.testing.assert_allclose(
        np.asarray(tree["infogain"]["likelihood:o"].lower),
        scale * np.asarray(ref["infogain"]["likelihood:o"].lower),
        rtol=1e-5,
    )


def test_the_planner_accepts_the_declaration_under_jit():
    system, params = _model((("s", "u_s"),), [3.0, 5.0])
    beliefs, _ = system.infer_states(params, {"o": jnp.asarray(0)})
    eager, _ = system.infer_policies(params, beliefs, metrics=True)
    compiled, _ = system.compiled().infer_policies(params, beliefs, metrics=True)
    np.testing.assert_allclose(np.asarray(eager.probs), np.asarray(compiled.probs), atol=1e-5)


# ----------------------------------------------------------------------- refusals


def test_a_declaration_naming_an_unknown_table_or_node_or_twice_is_refused():
    with pytest.raises(ValueError, match="neither a state factor"):
        _model((("nope", "u_s"),), None)
    with pytest.raises(ValueError, match="not a control node"):
        _model((("s", "u_nope"),), None)
    with pytest.raises(ValueError, match="twice"):
        _model((("s", "u_s"), ("s", "u_s")), [1.0, 1.0])


def test_a_table_that_is_not_learnable_is_refused():
    system, params = _model((("s", "u_s"),), [1.0, 1.0], epistemic=False)
    point = ConditionalCategorical(
        log_table=jnp.log(jnp.full((3, 3, 2), 1.0 / 3.0)), parent_ndims=2
    )
    with pytest.raises(ValueError, match="learnable table"):
        system._validate(dataclasses.replace(params, transition={"s": point}))


def test_counts_of_the_wrong_length_or_not_positive_or_missing_are_refused():
    system, params = _model((("s", "u_s"),), [1.0, 1.0])
    with pytest.raises(ValueError, match=r"has shape \(3,\).*2 level"):
        system._validate(dataclasses.replace(params, readings={"s": jnp.ones(3)}))
    with pytest.raises(ValueError, match="positive counts"):
        system._validate(dataclasses.replace(params, readings={"s": jnp.asarray([1.0, 0.0])}))
    with pytest.raises(ValueError, match="needs its counts"):
        system._validate(dataclasses.replace(params, readings={}))
    with pytest.raises(ValueError, match="names a declared table"):
        _, plain = _model()
        system._validate(dataclasses.replace(plain, readings={"s": jnp.ones(2), "x": jnp.ones(2)}))


def test_a_reading_over_a_continuous_control_node_is_refused():
    system, params = _rover()
    system = dataclasses.replace(system, readings=(("terrain", "u"),))
    params = dataclasses.replace(params, readings={"terrain": jnp.ones(1)})
    with pytest.raises(ValueError, match="continuous control node"):
        system._validate(params)


# --------------------------------------------------------------- merge and the views


def _two(readings_on_b=True):
    a, pa = _model()
    b, pb = _model((("s", "u_s"),) if readings_on_b else (), [3.0, 5.0] if readings_on_b else None)
    return a, pa, b, pb


def test_merge_namespaces_the_declaration_and_a_view_carries_its_own_tables_entry():
    a, pa, b, pb = _two()
    composite, params = merge({"a": a, "b": b}, {"a": pa, "b": pb})
    assert composite.readings == (("b.s", "b.u_s"),)
    np.testing.assert_array_equal(np.asarray(params.readings["b.s"]), [3.0, 5.0])
    view, view_params = subsystem(composite, params, "b")
    assert view.readings == (("b.s", "b.u_s"),)
    assert list(view_params.readings) == ["b.s"]
    other, other_params = subsystem(composite, params, "a")
    assert other.readings == () and other_params.readings == {}


def test_merge_follows_a_same_step_identification_to_the_drivers_node():
    a, pa, b, pb = _two()
    composite, _ = joint_planner(
        *merge({"a": a, "b": b}, {"a": pa, "b": pb}, [Link("a.a_s", "b.u_s")])
    )
    assert composite.readings == (("b.s", "a.u_s"),)


def test_a_view_whose_node_is_another_components_refuses_and_a_removed_node_refuses_at_merge():
    a, pa, b, pb = _two()
    composite, params = joint_planner(
        *merge({"a": a, "b": b}, {"a": pa, "b": pb}, [Link("a.a_s", "b.u_s")])
    )
    with pytest.raises(
        NotImplementedError, match="another component chooses|terminating a \\*learnable\\*"
    ):
        subsystem(composite, params, "b")
    with pytest.raises(ValueError, match="binding removed"):
        merge({"a": a, "b": b}, {"a": pa, "b": pb}, [Clamp("b.u_s", 0)])


# ----------------------------------------------------------------------- the document


def test_the_document_records_the_declaration_and_round_trips_it():
    system, params = _model((("s", "u_s"),), [3.0, 5.0])
    doc, arrays = describe(system, params)
    assert doc["factors"]["s"]["transition"]["readings"] == {
        "control": "u_s",
        "counts": [3.0, 5.0],
    }
    rebuilt_system, rebuilt = build(doc, arrays=arrays)
    assert rebuilt_system.readings == (("s", "u_s"),)
    np.testing.assert_array_equal(np.asarray(rebuilt.readings["s"]), [3.0, 5.0])
    again, _ = describe(rebuilt_system, rebuilt)
    assert again == doc
    _, tree = _plan(rebuilt_system, rebuilt)
    _, ref = _plan(system, params)
    np.testing.assert_allclose(_leaf(tree), _leaf(ref), rtol=1e-6)


def test_a_document_without_a_declaration_is_unchanged_and_a_bad_one_is_refused():
    from hybrid_aif.description import DescriptionError

    doc, arrays = describe(*_model())
    assert "readings" not in doc["factors"]["s"]["transition"]
    bad, bad_arrays = describe(*_model((("s", "u_s"),), [3.0, 5.0]))
    bad["factors"]["s"]["transition"]["readings"]["counts"] = [3.0, -1.0]
    with pytest.raises(DescriptionError, match="readings"):
        build(bad, arrays=bad_arrays)


def test_the_description_length_prices_nothing_for_a_declaration():
    from hybrid_aif.description.mdl import description_length

    pooled, pooled_params = _model((("s", "u_s"),), [3.0, 5.0])
    plain, plain_params = _model()
    assert description_length(pooled, 10.0, params=pooled_params).total == pytest.approx(
        description_length(plain, 10.0, params=plain_params).total
    )


def test_the_count_is_sliced_from_a_mixed_step_and_a_belief_over_the_levels_averages_it():
    from hybrid_aif.controls import ControlStep
    from hybrid_aif.interval import Interval
    from hybrid_aif.objectives.pooling import Pooled, PooledEntry, scale_pooled

    pooled = Pooled((PooledEntry(KEY, 0, jnp.asarray([3.0, 5.0])),))
    tree = {"infogain": {KEY: Interval.exact(jnp.asarray(2.0)), "other": Interval.exact(1.0)}}
    mixed = ControlStep(entries=(jnp.asarray(1), jnp.asarray([0.2, 0.3])))
    scaled = scale_pooled(tree, pooled, mixed, None)
    assert float(scaled["infogain"][KEY].lower) == pytest.approx(10.0)
    assert float(scaled["infogain"]["other"].lower) == pytest.approx(1.0)
    soft = ControlStep(entries=(jnp.asarray([0.5, 0.5]), jnp.asarray([0.2, 0.3])))
    assert float(scale_pooled(tree, pooled, soft, None)["infogain"][KEY].lower) == pytest.approx(
        8.0
    )
    assert scale_pooled(tree, None, mixed, None) is tree


def test_a_reading_on_a_table_whose_novelty_is_not_selected_is_refused_naming_the_term():
    for epistemic in (False, ()):
        system, params = _model((("s", "u_s"),), [3.0, 5.0], epistemic=epistemic)
        with pytest.raises(ValueError, match=r"does not select.*add transition:'s' to epistemic"):
            system._validate(params)
    system, params = _model((("s", "u_s"),), [3.0, 5.0])
    system._validate(params)  # selected: accepted
