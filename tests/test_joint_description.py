"""The document, the description length and the views of what ``merge``, ``couple`` and
``joint_planner`` build: a driven control node is a slot of the decision variable that drives it,
a leaf's scoring and a stated planner field round-trip, durations size a node through the leaf
records, and the joint planner's configuration reads the node it plans.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition import Link, couple, joint_planner, merge, subsystem
from hybrid_aif.description import build, describe
from hybrid_aif.description.mdl import description_length
from hybrid_aif.reference.raif.agent import raif_mountain_car
from hybrid_aif.system import UNSET
from tests.test_action_nodes import KERNEL, pusher
from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world
from tests.test_splice import ACT, SENSES, zoo_coupling, zoo_world
from tests.test_unread_control_node import model


def _static_diffs(a, b):
    return [f.name for f in dataclasses.fields(a) if getattr(a, f.name) != getattr(b, f.name)]


def _leaves_equal(a, b):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ------------------------------------------------------ the driven node in the document and the MDL


def test_the_document_of_a_driven_coupling_has_the_variables_of_its_joint_planner():
    cs, cp = zoo_coupling()
    document, _ = describe(cs, cp, arrays="inline")
    mind, mp = joint_planner(cs, cp)
    joint_document, _ = describe(mind, mp, arrays="inline")
    assert set(document["variables"]) == set(joint_document["variables"])
    assert not {"world.u_location", "world.a_location"} & set(document["variables"])
    assert document["relations"]["driven"] == {"world.u_location": "agent.a_location"}
    assert document["relations"]["driven_actions"] == {"world.u_location": "world.a_location"}


def test_the_zoo_coupling_round_trips_exactly_through_describe_and_build():
    cs, cp = zoo_coupling()
    document, arrays = describe(cs, cp, arrays="inline")
    rebuilt, rp = build(document, arrays=arrays)
    assert _static_diffs(rebuilt, cs) == []
    world = dict(rebuilt.subsystems)["world"]
    assert world.actions == (("world.u_location", "world.a_location"),)
    assert world.identified == (("world.u_location", "agent.u_location"),)
    _leaves_equal(rp, cp)


def test_a_driven_coupling_costs_what_its_joint_planner_costs_and_pays_each_size_once():
    cs, cp = zoo_coupling()
    mind, mp = joint_planner(cs, cp)
    priced = description_length(cs, params=cp)
    assert priced.total == pytest.approx(description_length(mind, params=mp).total)
    assert priced.total == pytest.approx(111.2671, abs=1e-4)
    assert [k for k in priced.per_address if k.startswith("control:")] == [
        "control:agent.u_location"
    ]


def test_no_identified_variable_pays_its_level_count_twice_in_a_merge_either():
    (agent, ap), (world, wp) = zoo_world()
    merged, mp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        [ACT, *SENSES],
    )
    paid = [k for k in description_length(merged, params=mp).per_address if "control:" in k]
    assert paid == ["control:agent.u_location"]


# --------------------------------------------------------------------- durations size a node


def _two_clocks():
    a, ap = model("durations", durations=[1.0, 2.0])
    b, bp = model("durations", durations=[3.0, 4.0])
    return {"a": a, "b": b}, {"a": ap, "b": bp}


@pytest.mark.parametrize("compose", [merge, lambda s, p: couple({k: (s[k], p[k]) for k in s})])
def test_a_node_sized_only_by_durations_round_trips_where_two_leaves_carry_durations(compose):
    systems, params = _two_clocks()
    cs, cp = compose(systems, params)
    document, arrays = describe(cs, cp, arrays="inline")
    rebuilt, rp = build(document, arrays=arrays)
    assert rebuilt._space(rp).sizes == cs._space(cp).sizes
    assert set(rp.parts["a"].durations) == {"a.u_opt"} and rp.durations == {}


def test_the_composites_own_durations_size_a_node_before_a_leafs():
    systems, params = _two_clocks()
    cs, cp = merge(systems, params)
    stated = dataclasses.replace(cp, durations={"a.u_opt": jnp.asarray([5.0, 6.0, 7.0])})
    assert cs._space(stated).sizes == (3, 2)  # the composite's, then the other leaf's own
    view, vp = subsystem(cs, stated, "a")
    assert view._space(vp).sizes == (3,)  # the view agrees with the composite


# ------------------------------------------------------------------------ a leaf's scoring


def _scoring_pair():
    r = raif_mountain_car(jax.random.PRNGKey(0))
    scored, sp = r.system, dataclasses.replace(r.params, policy_prior=None)
    r2 = raif_mountain_car(jax.random.PRNGKey(1))
    plain = dataclasses.replace(r2.system, terms=(), scoring=())
    pp = dataclasses.replace(r2.params, scoring={}, policy_prior=None)
    return {"mind": scored, "other": plain}, {"mind": sp, "other": pp}


@pytest.mark.parametrize("compose", [merge, lambda s, p: couple({k: (s[k], p[k]) for k in s})])
def test_a_composite_with_a_scoring_leaf_round_trips_through_the_document(compose):
    systems, params = _scoring_pair()
    cs, cp = compose(systems, params)
    document, arrays = describe(cs, cp, arrays="inline")
    assert {t["leaf"] for t in document["objective"]["terms"] if "leaf" in t} == {"mind"}
    rebuilt, rp = build(document, arrays=arrays)
    assert _static_diffs(rebuilt, cs) == []
    assert dict(rebuilt.subsystems) == dict(cs.subsystems)
    assert sorted(rp.parts["mind"].scoring) == sorted(cp.parts["mind"].scoring)
    assert rp.scoring == {} and rebuilt.scoring == ()


def test_a_view_of_a_scoring_leaf_from_a_joint_planner_carries_its_parameters():
    systems, params = _scoring_pair()
    cs, cp = couple({k: (systems[k], params[k]) for k in systems})
    joint, jp = joint_planner(cs, cp)
    view, vp = subsystem(joint, jp, "mind")
    assert sorted(vp.scoring) == sorted(cp.parts["mind"].scoring)
    assert [n for n, _ in view.scoring] == [n for n, _ in systems["mind"].scoring]
    document, _ = describe(view, vp, arrays="inline")  # the view's own record does not block it
    assert any(t["term"] == "g_tail" and "parameters" in t for t in document["objective"]["terms"])


def test_a_scoring_term_without_its_parameters_is_refused_on_any_system():
    systems, params = _scoring_pair()
    scored, sp = systems["mind"], params["mind"]
    with pytest.raises(ValueError, match="owns parameters, and SystemParams.scoring carries no"):
        scored._validate(scored._p(dataclasses.replace(sp, scoring={})))


# ------------------------------------------------------------- a stated planner field


@pytest.mark.parametrize("compose", ["couple", "merge"])
def test_a_planner_field_the_composite_states_survives_the_document(compose):
    (agent, ap), (world, wp) = zoo_world()
    world = dataclasses.replace(world, policy_len=2, gamma=4.0)
    if compose == "couple":
        cs, cp = couple({"agent": (agent, ap), "world": (world, wp)}, SENSES)
    else:
        cs, cp = merge({"agent": agent, "world": world}, {"agent": ap, "world": wp}, SENSES)
    assert cs.policy_len is UNSET
    stated = dataclasses.replace(cs, policy_len=3, gamma=1.5)
    document, arrays = describe(stated, cp, arrays="inline")
    rebuilt, _ = build(document, arrays=arrays)
    assert (rebuilt.policy_len, rebuilt.gamma) == (3, 1.5)
    document, arrays = describe(cs, cp, arrays="inline")  # unstated stays unset
    assert build(document, arrays=arrays)[0].policy_len is UNSET


# ----------------------------------------------- the joint planner's configuration and kernel


def test_a_field_stated_on_an_enclosing_composite_does_not_reach_a_nested_node():
    (agent, ap), (world, wp) = agent_and_world(1)
    inner, ip = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        list(OBSERVATION_BINDINGS),
    )
    (solo, sp), _ = agent_and_world(1)
    outer, op = merge({"pair": inner, "solo": solo}, {"pair": ip, "solo": sp})
    stated = dataclasses.replace(outer, policy_len=2)
    assert joint_planner(stated, op)[0].policy_len == 2  # the whole System's own
    assert joint_planner(stated, op, "pair")[0].policy_len == 1  # the node's leaves agree on 1
    assert joint_planner(stated, op, "pair", policy_len=3)[0].policy_len == 3


def test_unset_is_not_a_configuration_value():
    (agent, ap), (world, wp) = agent_and_world(1)
    cs, cp = merge({"agent": agent, "world": world}, {"agent": ap, "world": wp}, [])
    with pytest.raises(ValueError, match=r"\['gamma', 'policy_len'\] is UNSET"):
        joint_planner(cs, cp, policy_len=UNSET, gamma=UNSET)


def test_an_identity_kernel_on_the_driver_is_accepted_and_any_other_is_refused():
    (a, ap), (b, bp) = pusher(), pusher()
    link = [Link("a.push", "b.u_push")]
    same = dataclasses.replace(ap, actuator={"push": jnp.eye(3)})
    mind, mp = joint_planner(*merge({"a": a, "b": b}, {"a": same, "b": bp}, link))
    assert mind.names.controls == ("a.u_push",)
    skewed = dataclasses.replace(ap, actuator={"push": KERNEL})
    merged, merged_params = merge({"a": a, "b": b}, {"a": skewed, "b": bp}, link)
    with pytest.raises(ValueError, match="not the identity"):
        joint_planner(merged, merged_params)
