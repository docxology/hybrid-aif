"""The splice and its bindings: ``couple`` returns a System, a leaf keeps its own configuration,
habit, precision, durations, scoring and recognition, an out-port is identified with an in-port,
and a driven control node carries the realised value of the action node that drives it.

The pieces, in order: a coupling's records (per-leaf ``config`` and ``parts``, ``coupled``
boundary, ``UNSET`` where leaves differ); the view of a coupled leaf against its own System; the
two leaf kinds a merge refuses (a leaf with recognition, a leaf with a scoring estimator);
port identification into ``observes`` (a modality, a driven control node), the variable count and
the document; ``subsystem(..., inputs=)`` of a driven slot against the sliced collapse of a
same-step ``Link``; the planner never choosing a driven node; what ``couple`` refuses; a merge of a
coupling and a coupling of a merge.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical
from hybrid_aif.composition import Clamp, Identify, Link, couple, joint_planner, merge, subsystem
from hybrid_aif.description import build, describe
from hybrid_aif.distributions import MLPConditional
from hybrid_aif.factors import LinearGaussianObservation, Recognition
from hybrid_aif.reads import Emission
from hybrid_aif.system import UNSET, PartParams, System
from tests.test_hybrid_lg_system import R_CHOL, C, H, make_params, make_system
from tests.test_merge import agent_and_world, merged
from tests.test_merge_gaussian_control import mission
from tests.test_merge_recognition import component
from tests.test_planning_with_observations import _probe_row
from tests.test_subsystem_conditional_target import _conditional_target, _rover, _two_tier

KEY = jax.random.PRNGKey(0)
CUES = ("location_obs", "reward", "cue")


def zoo_world(policy_len=1):
    """The zoo's world with its three sensors declared as emissions of its own latents, each with
    the agent's likelihood table as its kernel: the process an agent can be coupled to."""
    (agent, ap), (world, wp) = agent_and_world(policy_len)
    emissions = {
        m: Emission(tuple(world.factor_names[g] for g in agent.A_deps[i]), "sample")
        for i, m in enumerate(agent.modality_names)
    }
    world = dataclasses.replace(world, emissions=emissions)
    wp = dataclasses.replace(wp, emission=dict(ap.likelihood))
    return (agent, ap), (world, wp)


SENSES = [Identify(f"world.{m}", f"agent.{m}") for m in CUES]
ACT = Identify("agent.a_location", "world.u_location")


def zoo_coupling(**kwargs):
    (agent, ap), (world, wp) = zoo_world(**kwargs)
    return couple({"agent": (agent, ap), "world": (world, wp)}, [ACT, *SENSES])


def _static(system):
    """Every static field of ``system`` but its component records."""
    return {
        f.name: getattr(system, f.name)
        for f in dataclasses.fields(System)
        if f.name != "subsystems"
    }


def _same_leaves(a, b):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ------------------------------------------------------------------------ a coupling's records


def test_couple_returns_a_system_whose_components_are_coupled_leaves():
    cs, cp = zoo_coupling()
    assert isinstance(cs, System)
    records = dict(cs.subsystems)
    assert list(records) == ["agent", "world"]
    for name in records:
        assert records[name].boundary == "coupled" and not records[name].nested
        assert dict(records[name].config)["policy_len"] == 1
    assert cs.policy_len == 1 and cs.planner == "enumeration"
    assert cp.policy_prior is None and cp.precision is None and cp.durations == {}
    assert cs.observes == (
        ("agent.location_obs", "world.location_obs"),
        ("agent.reward", "world.reward"),
        ("agent.cue", "world.cue"),
        ("world.u_location", "agent.a_location"),
    )
    assert cs.action_names == ("agent.a_location", None)


def test_a_planner_field_the_leaves_differ_on_is_unset_and_a_face_that_needs_it_refuses():
    (agent, ap), (world, wp) = zoo_world()
    world = dataclasses.replace(world, policy_len=3)
    cs, cp = couple({"agent": (agent, ap), "world": (world, wp)}, [ACT, *SENSES])
    assert cs.policy_len is UNSET and cs.gamma == agent.gamma
    leaves = {name: dict(sub.config)["policy_len"] for name, sub in cs.subsystems}
    assert leaves == {"agent": 1, "world": 3}
    with pytest.raises(ValueError, match="coupling.*leaf by leaf"):
        cs.policies(cp)
    view, vp = subsystem(cs, cp, "world", inputs={"agent.a_location": 0})
    assert view.policy_len == 3
    assert subsystem(cs, cp, "agent")[0].policy_len == 1
    document, arrays = describe(cs, cp, arrays="inline")  # the leaf records carry the difference
    rebuilt, rp = build(document, arrays=arrays)
    assert rebuilt.policy_len is UNSET and rebuilt.gamma == agent.gamma
    assert {n: dict(s.config)["policy_len"] for n, s in rebuilt.subsystems} == leaves
    again, ap = merge({"pair": cs}, {"pair": cp})  # a merge carries the difference as well
    assert again.policy_len is UNSET
    assert {n: dict(s.config)["policy_len"] for n, s in again.subsystems if s.config} == {
        "pair.agent": 1,
        "pair.world": 3,
    }
    with pytest.raises(ValueError, match="disagree on policy_len"):
        joint_planner(again, ap)  # one mind plans with one policy_len


def test_each_leaf_keeps_its_own_habit_precision_and_durations_in_parts():
    (agent, ap), (world, wp) = zoo_world()
    ap = dataclasses.replace(ap, policy_prior=jnp.arange(5.0), precision=2.0)
    wp = dataclasses.replace(wp, durations={"u_location": jnp.full(5, 2.0)})
    cs, cp = couple({"agent": (agent, ap), "world": (world, wp)}, [ACT, *SENSES])
    assert cp.policy_prior is None and cp.precision is None and cp.durations == {}
    assert set(cp.parts) == {"agent", "world"}
    assert isinstance(cp.parts["agent"], PartParams)
    np.testing.assert_array_equal(cp.parts["agent"].policy_prior, np.arange(5.0))
    assert float(cp.parts["agent"].precision) == 2.0
    assert set(cp.parts["world"].durations) == {"world.u_location"}
    view, vp = subsystem(cs, cp, "agent")
    np.testing.assert_array_equal(vp.policy_prior, np.arange(5.0))
    assert float(vp.precision) == 2.0
    with pytest.raises(ValueError, match="coupling.*leaf by leaf"):
        cs.elapsed({}, cp)
    document, arrays = describe(cs, cp, arrays="inline")
    rebuilt, rp = build(document, arrays=arrays)
    _same_leaves(rp.parts, cp.parts)
    assert dict(rebuilt.subsystems)["agent"].config == dict(cs.subsystems)["agent"].config
    assert [s.boundary for _, s in rebuilt.subsystems] == ["coupled", "coupled"]
    assert float(subsystem(rebuilt, rp, "agent")[1].precision) == 2.0


def test_a_merge_of_leaves_without_per_leaf_parameters_carries_no_parts():
    composite, cp = merged()
    assert cp.parts == {}
    assert all(sub.config for _, sub in composite.subsystems)


# -------------------------------------------------------------- a coupled leaf's view is its own


def test_a_coupled_leaf_s_view_is_its_own_system_namespaced():
    (agent, ap), (world, wp) = zoo_world()
    cs, cp = couple({"agent": (agent, ap), "world": (world, wp)}, [ACT, *SENSES])
    own, own_p = merge({"agent": agent}, {"agent": ap})
    view, vp = subsystem(cs, cp, "agent")
    want = _static(own)
    got = _static(view)
    assert got == want
    _same_leaves(vp, own_p)
    assert dict(view.subsystems)["agent"].boundary == "coupled"
    assert view.modality_names == tuple(f"agent.{m}" for m in CUES)
    assert view.observes == ()  # its modalities observe the world's variables: observations


def test_a_leaf_with_a_recognition_entry_is_coupled_to_a_world_with_its_recognition_kept():
    module = dataclasses.replace(
        MLPConditional.from_shapes((2,), 2, key=KEY, width=2, depth=1), frozen=True
    )
    agent, ap = component((Recognition("x", ("obs:y",)),), module)
    world, wp = component()
    world = dataclasses.replace(
        world, A_deps=(), modality_names=(), emissions={"y": Emission(("x",), "sample")}
    )
    wp = dataclasses.replace(
        wp,
        likelihood={},
        emission={
            "y": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.3 * jnp.eye(2))
        },
    )
    cs, cp = couple({"mind": (agent, ap), "world": (world, wp)}, [Identify("world.y", "mind.y")])
    assert cs.recognition == (Recognition("mind.x", ("obs:mind.y",)),)
    assert dict(cs.subsystems)["mind"].recognition == ("mind.x",)
    assert cp.recognition["mind.x"] is module
    view, vp = subsystem(cs, cp, "mind")
    own, own_p = merge({"mind": agent}, {"mind": ap})
    assert _static(view) == _static(own)
    assert vp.recognition["mind.x"] is module
    q, _ = view.infer_states(vp, {"mind.y": jnp.asarray([0.5, -0.5])})
    assert np.isfinite(np.asarray(q["mind.x"].loc)).all()
    assert subsystem(cs, cp, "world")[0].recognition is None


def test_a_leaf_with_a_scoring_estimator_is_coupled_to_a_world_with_its_scoring_kept():
    _probe_row()
    agent = make_system(terms=("epistemic_probe",), policy_len=2)
    ap = dataclasses.replace(make_params(), scoring={"epistemic_probe": jnp.asarray(2.0)})
    world = make_system(
        A_deps=(), modality_names=(), policy_len=2, emissions={"position": ("state",)}
    )
    wp = dataclasses.replace(
        make_params(),
        likelihood={},
        state_targets={},
        emission={"position": LinearGaussianObservation(H=H, c=C, noise=R_CHOL)},
    )
    bindings = [
        Identify("world.position", "mind.position"),
        Identify("mind.a_state", "world.u_state"),
    ]
    cs, cp = couple({"mind": (agent, ap), "world": (world, wp)}, bindings)
    assert dict(cs.subsystems)["mind"].config and "epistemic_probe" not in cs.terms
    np.testing.assert_array_equal(cp.parts["mind"].scoring["epistemic_probe"], 2.0)
    assert cp.scoring == {}
    view, vp = subsystem(cs, cp, "mind")
    bare = dataclasses.replace(agent, terms=())
    own, own_p = merge({"mind": bare}, {"mind": dataclasses.replace(ap, scoring={})})
    assert {**_static(view), "terms": ()} == _static(own)  # the leaf's term is the leaf's alone
    assert view.terms == ("epistemic_probe",) and cs.terms == ()
    np.testing.assert_array_equal(vp.scoring["epistemic_probe"], 2.0)
    merged_, mp = merge({"mind": agent}, {"mind": ap})  # a merge keeps it on the leaf's record
    assert merged_.scoring == () and mp.scoring == {} and mp.parts["mind"].scoring
    one_mind, one_p = joint_planner(merged_, mp)
    assert dict(one_mind.scoring) == dict(agent.scoring) and one_p.scoring == ap.scoring


# ------------------------------------------------- identifying ports: variables and the document


def _two_agents_one_world():
    (agent, ap), (world, wp) = zoo_world()
    return couple(
        {"a": (agent, ap), "b": (agent, ap), "world": (world, wp)},
        [
            Identify("world.reward", "a.reward"),
            Identify("world.reward", "b.reward"),
            Identify("world.cue", "a.cue"),
            Identify("world.location_obs", "b.location_obs"),
        ],
    )


def test_a_coupling_counts_the_decision_variables_of_its_parts_less_the_identifications():
    """A control node and its action node are one decision variable, so a part's count is its
    document's variables less one per action node (agent 14, world 13): the coupling with the
    world's node driven holds 14 + 13 - 4 = 23."""
    (agent, ap), (world, wp) = zoo_world()
    cs, cp = _two_agents_one_world()
    document, _ = describe(cs, cp, arrays="inline")

    def decisions(system, params):
        doc, _ = describe(system, params, arrays="inline")
        return len(doc["variables"]) - sum(a is not None for a in system.action_names)

    parts = [decisions(s, p) for s, p in ((agent, ap), (agent, ap), (world, wp))]
    assert parts[0] == 14 and parts[2] == 13
    # four in-ports observe a variable, so each is a modality that is no variable of its own;
    # world.reward feeds two of them and is counted once
    assert decisions(cs, cp) == sum(parts) - 4
    driven, dp = zoo_coupling()
    assert decisions(driven, dp) == 14 + 13 - 4
    assert len(describe(driven, dp, arrays="inline")[0]["variables"]) == 24  # = its joint planner's
    joint, jp = joint_planner(driven, dp)
    assert len(describe(joint, jp, arrays="inline")[0]["variables"]) == 24


def test_the_document_writes_an_identified_variable_once_and_its_observers_name_it():
    cs, cp = _two_agents_one_world()
    document, _ = describe(cs, cp, arrays="inline")
    assert "world.reward" in document["variables"]
    assert not {"a.reward", "b.reward", "a.cue", "b.location_obs"} & set(document["variables"])
    for slot in ("a.reward", "b.reward"):
        assert document["factors"][slot]["likelihood"]["observes"] == "world.reward"
    assert document["factors"]["a.cue"]["likelihood"]["observes"] == "world.cue"
    assert "observes" not in document["factors"]["a.location_obs"]["likelihood"]


def test_a_coupling_with_a_driven_node_is_described_and_rebuilt():
    cs, cp = zoo_coupling()
    document, arrays = describe(cs, cp, arrays="inline")
    assert document["relations"]["driven"] == {"world.u_location": "agent.a_location"}
    assert document["relations"]["controls"] == {"agent.u_location": "agent.a_location"}
    rebuilt, rp = build(document, arrays=arrays)
    assert _static(rebuilt) == _static(cs)
    assert rebuilt.action_names == ("agent.a_location", None)
    _same_leaves(rp, cp)
    assert [(n, s.factors, s.modalities) for n, s in rebuilt.subsystems] == [
        (n, s.factors, s.modalities) for n, s in cs.subsystems
    ]


def test_an_identified_port_pair_is_refused_unless_it_joins_an_out_port_to_an_in_port():
    (agent, ap), (world, wp) = zoo_world()
    parts = {"agent": (agent, ap), "world": (world, wp)}
    with pytest.raises(ValueError, match="joins one out-port"):
        couple(parts, [Identify("world.reward", "world.cue")])
    with pytest.raises(ValueError, match="joins one out-port"):
        couple(parts, [Identify("agent.reward", "agent.cue")])
    with pytest.raises(ValueError, match="at most once"):
        couple(
            parts, [Identify("world.reward", "agent.reward"), Identify("world.cue", "agent.reward")]
        )
    with pytest.raises(ValueError, match="different components"):
        couple(parts, [Identify("world.reward", "world.u_location")])
    with pytest.raises(ValueError, match="already observes a variable"):
        couple(parts, [ACT, Identify("agent.a_location", "world.u_location")])
    with pytest.raises(
        ValueError, match=r"port is 5 levels \(level\), but the slot takes 3 levels \(level\)"
    ):
        couple(parts, [Identify("world.location_obs", "agent.cue")])


# --------------------------------------------------------- inputs= on the view of a driven slot


@pytest.mark.parametrize("level", [0, 1])
def test_a_driven_slot_sliced_by_inputs_is_the_collapsed_links_sliced_view(level):
    rs, rp = _rover(_conditional_target(), (("x", ("u_opt",)),))
    ms, mp = mission()
    linked, linked_p = _two_tier(rs, rp)
    coupled, coupled_p = couple(
        {"mission": (ms, mp), "rover": (rs, rp)}, [Identify("mission.option", "rover.u_opt")]
    )
    assert dict(coupled.observes) == {"rover.u_opt": "mission.option"}
    got, gp = subsystem(coupled, coupled_p, "rover", inputs={"mission.option": level})
    want, wp = subsystem(linked, linked_p, "rover", inputs={"mission.option": level})
    assert got.C_deps == () == want.C_deps
    for field in ("A_deps", "B_deps", "factor_names", "modality_names", "action_names", "controls"):
        assert getattr(got, field) == getattr(want, field), field
    _same_leaves(gp.transition, wp.transition)
    _same_leaves(gp.state_targets, wp.state_targets)
    _same_leaves(gp.likelihood, wp.likelihood)


def test_an_unsliced_view_keeps_the_driven_node_as_an_input_and_a_planner_never_chooses_it():
    rs, rp = _rover(_conditional_target(), (("x", ("u_opt",)),))
    ms, mp = mission()
    cs, cp = couple(
        {"mission": (ms, mp), "rover": (rs, rp)}, [Identify("mission.option", "rover.u_opt")]
    )
    assert cs.action_names == ("mission.option", "rover.a_v", None)
    assert "rover.u_opt" not in cs.ports(cp).ins and "rover.a_opt" not in cs.ports(cp).outs
    view, vp = subsystem(cs, cp, "rover")
    assert view.action_names == ("rover.a_v", None)
    assert "rover.u_opt" in view.ports(vp).ins
    with pytest.raises(ValueError, match=r"control node\(s\) \['rover.u_opt'\] are driven"):
        view.policies(vp)
    with pytest.raises(ValueError, match="coupling.*leaf by leaf"):
        cs.policies(cp)
    with pytest.raises(ValueError, match="driven"):
        view.infer_policies(vp, view.initial_priors(vp))
    sliced, sp = subsystem(cs, cp, "rover", inputs={"mission.option": 1})
    assert sliced.action_names == ("rover.a_v",) and sliced.names.controls == ("rover.v",)


def test_a_driven_continuous_slot_is_never_sliced_and_a_wrong_key_is_refused():
    (agent, ap), (world, wp) = zoo_world()
    cs, cp = couple({"agent": (agent, ap), "world": (world, wp)}, [ACT, *SENSES])
    with pytest.raises(ValueError, match="no variable the view of 'world' takes"):
        subsystem(cs, cp, "world", inputs={"world.u_location": 0})
    with pytest.raises(ValueError, match="no variable the view of 'agent' takes"):
        subsystem(cs, cp, "agent", inputs={"agent.a_location": 0})
    uniform, up = subsystem(cs, cp, "world", inputs={"agent.a_location": None})
    table = np.exp(np.asarray(wp.transition["location"].log_probs))
    got = np.exp(np.asarray(up.transition["world.location"].log_probs))[..., 0]
    np.testing.assert_allclose(got, table.mean(axis=-1), rtol=1e-6)
    with pytest.raises(ValueError, match="not one of its 5 control level"):
        subsystem(cs, cp, "world", inputs={"agent.a_location": 9})


# ------------------------------------------------------------------------- what couple refuses


def test_couple_refuses_a_structural_link_a_read_sugar_and_a_parameter_identification():
    (agent, ap), (world, wp) = zoo_world()
    parts = {"agent": (agent, ap), "world": (world, wp)}
    with pytest.raises(ValueError, match="Identify or a Clamp.*merge"):
        couple(parts, [Link(("world.location",), "agent.location_obs")])
    with pytest.raises(ValueError, match="Identify or a Clamp.*merge"):
        couple(parts, [Link("agent.location", "world.u_location")])
    with pytest.raises(ValueError, match="Identify or a Clamp.*merge"):
        couple(parts, [Link("world.location", "agent.cue", realise="mode")])


def test_couple_holds_a_control_node_at_a_level_with_fixed():
    (agent, ap), (world, wp) = zoo_world()
    cs, cp = couple(
        {"agent": (agent, ap), "world": (world, wp)}, SENSES, fixed={"world.u_location": 2}
    )
    assert "world.u_location" not in cs.names.controls
    cs2, _ = couple(
        {"agent": (agent, ap), "world": (world, wp)}, [*SENSES, Clamp("world.u_location", 2)]
    )
    assert _static(cs) == _static(cs2)


# ------------------------------------------------------------------- merge and couple compose


def test_a_merge_of_a_coupling_namespaces_its_leaves_and_its_driven_slot():
    cs, cp = zoo_coupling()
    ms, mp = merge({"pair": (cs)}, {"pair": cp})
    assert ms.action_names == ("pair.agent.a_location", None)
    assert dict(ms.observes)["pair.world.u_location"] == "pair.agent.a_location"
    assert list(dict(ms.subsystems)) == ["pair", "pair.agent", "pair.world"]
    assert dict(ms.subsystems)["pair"].config == ()
    assert dict(dict(ms.subsystems)["pair.agent"].config)["policy_len"] == 1
    document, arrays = describe(ms, mp, arrays="inline")
    rebuilt, rp = build(document, arrays=arrays)
    assert _static(rebuilt) == _static(ms)
    view, vp = subsystem(ms, mp, "pair.agent")
    assert view.modality_names == tuple(f"pair.agent.{m}" for m in CUES)


def test_a_coupling_of_a_merge_carries_the_merge_s_components_as_leaves():
    (agent, ap), (world, wp) = zoo_world()
    inner, ip = merge({"agent": agent}, {"agent": ap})
    cs, cp = couple(
        {"mind": (inner, ip), "world": (world, wp)},
        [
            Identify("world.location_obs", "mind.agent.location_obs"),
            Identify("world.reward", "mind.agent.reward"),
            Identify("world.cue", "mind.agent.cue"),
            Identify("mind.agent.a_location", "world.u_location"),
        ],
    )
    names = [n for n, _ in cs.subsystems]
    assert names == ["mind", "mind.agent", "world"]
    assert dict(cs.subsystems)["mind"].boundary == "coupled"
    assert dict(cs.subsystems)["mind"].config == ()
    document, arrays = describe(cs, cp, arrays="inline")
    rebuilt, rp = build(document, arrays=arrays)
    assert _static(rebuilt) == _static(cs)
    view, vp = subsystem(cs, cp, "mind.agent")
    assert view.policy_len == 1 and view.modality_names == tuple(f"mind.agent.{m}" for m in CUES)
    assert Categorical  # the public names resolve


def test_a_merge_with_a_same_step_link_is_the_driven_form_and_the_joint_planner_collapses_it():
    composite, cp = merged(bindings=[Link("agent.a_location", "world.u_location")])
    assert composite.policy_len == 2 and cp.parts == {}
    assert composite.names.controls == ("agent.u_location", "world.u_location")
    with pytest.raises(ValueError, match="are driven"):
        composite.policies(cp)
    one_mind, mp = joint_planner(composite, cp)
    assert one_mind.names.controls == ("agent.u_location",)
    one_mind.policies(mp)


def test_a_merge_takes_the_driven_form_too_and_leaves_the_node_in_its_leaf():
    (agent, ap), (world, wp) = zoo_world()
    cs, cp = merge({"agent": agent, "world": world}, {"agent": ap, "world": wp}, [ACT, *SENSES])
    assert cs.action_names == ("agent.a_location", None)
    assert cs.names.controls == ("agent.u_location", "world.u_location")
    assert dict(cs.subsystems)["world"].controls == ()  # no leaf chooses it
    assert dict(cs.subsystems)["world"].boundary == "blanket"
    with pytest.raises(ValueError, match="are driven"):
        cs.policies(cp)


# ---------------------------------------------------- the review's probes, as regression tests


def _habit():
    return jnp.log(jnp.asarray([0.6, 0.1, 0.1, 0.1, 0.1]))


def test_a_composite_edited_after_merge_is_authoritative_for_its_views_and_re_merges():
    (agent, ap), (world, wp) = agent_and_world(1)
    ap = dataclasses.replace(ap, precision=2.0)
    wp = dataclasses.replace(wp, precision=2.0)
    cs, cp = merge({"agent": agent, "world": world}, {"agent": ap, "world": wp})
    edited = dataclasses.replace(cs, policy_len=2, gamma=4.0, planner="sophisticated")
    view, _ = subsystem(edited, cp, "agent")
    assert (view.policy_len, view.gamma, view.planner) == (2, 4.0, "sophisticated")
    _, vp = subsystem(cs, dataclasses.replace(cp, precision=5.0), "agent")
    assert float(vp.precision) == 5.0
    assert float(joint_planner(cs, cp)[1].precision) == 2.0  # composed from the leaves' records
    (third, tp), _ = agent_and_world(1)
    tp = dataclasses.replace(tp, precision=2.0)
    third = dataclasses.replace(third, policy_len=2, gamma=4.0, planner="sophisticated")
    both, _ = merge({"pair": edited, "third": third}, {"pair": cp, "third": tp})
    assert (both.policy_len, both.gamma, both.planner) == (2, 4.0, "sophisticated")
    alone, _ = merge({"pair": edited}, {"pair": cp})
    assert (alone.policy_len, alone.gamma) == (2, 4.0)
    assert set(cp.parts) == {"agent", "world"}  # each leaf's own precision stays on its record


def test_coupling_a_merge_keeps_the_merged_component_s_own_habit_precision_and_durations():
    (agent, ap), (world, wp) = zoo_world()
    inner, ip = merge({"agent": agent}, {"agent": ap})
    ip = dataclasses.replace(
        ip, precision=0.5, policy_prior=_habit(), durations={"agent.u_location": jnp.full(5, 3.0)}
    )
    want = np.asarray(inner.infer_policies(ip, inner.initial_priors(ip))[0].probs)
    cs, cp = couple(
        {"mind": (inner, ip), "world": (world, wp)},
        [Identify(f"world.{m}", f"mind.agent.{m}") for m in CUES],
    )
    view, vp = subsystem(cs, cp, "mind.agent")
    got = np.asarray(view.infer_policies(vp, view.initial_priors(vp))[0].probs)
    np.testing.assert_allclose(got, want)
    assert float(vp.precision) == 0.5 and set(vp.durations) == {"mind.agent.u_location"}
    many, mp = merge({"a": agent, "b": agent}, {"a": ap, "b": ap})
    mp = dataclasses.replace(mp, policy_prior=jnp.zeros(25))
    with pytest.raises(NotImplementedError, match="cannot be split among them"):
        couple({"pair": (many, mp)})


def test_no_planning_face_runs_a_coupling_as_one_system():
    from hybrid_aif.replay import replay

    (agent, ap), (world, wp) = zoo_world()
    ap = dataclasses.replace(ap, policy_prior=_habit(), precision=0.5)
    lone, lp = couple({"agent": (agent, ap)})
    q = lone.initial_priors(lp)
    for call in (
        lambda: lone.infer_policies(lp, q),
        lambda: lone.act(KEY, lp, q),
        lambda: lone.policies(lp),
        lambda: lone.elapsed({"agent.a_location": jnp.asarray(0)}, lp),
    ):
        with pytest.raises(ValueError, match="coupling.*leaf by leaf"):
            call()
    cs, cp = zoo_coupling()
    obs = {dict(cs.observes).get(m, m): jnp.zeros((3,), jnp.int32) for m in cs.modality_names}
    with pytest.raises(ValueError, match="coupling.*leaf by leaf"):
        replay(cs, cp, obs, {"agent.a_location": jnp.zeros((3,), jnp.int32)}, key=KEY)
    view, vp = subsystem(lone, lp, "agent")
    own = agent.infer_policies(ap, agent.initial_priors(ap))[0]
    np.testing.assert_allclose(
        np.asarray(view.infer_policies(vp, view.initial_priors(vp))[0].probs),
        np.asarray(own.probs),
        atol=1e-6,
    )
    (a2, a2p), (w2, w2p) = zoo_world(1)
    driven, dp = merge({"agent": a2, "world": w2}, {"agent": a2p, "world": w2p}, [ACT, *SENSES])
    with pytest.raises(ValueError, match="are driven"):
        replay(driven, dp, {}, {}, key=KEY)


def test_an_action_keyed_by_the_driver_sets_the_driven_node_and_the_node_is_refused():
    cs, cp = zoo_coupling()
    q = cs.initial_priors(cp)
    obs = {f"world.{m}": jnp.asarray(0) for m in CUES}
    post, _ = cs.infer_states(cp, obs, q)
    nxt = cs.update_empirical_prior(cp, post, {"agent.a_location": jnp.asarray(2)})
    assert set(nxt) >= {"world.location"}
    with pytest.raises(ValueError, match="driven control node.*'agent.a_location'|realised value"):
        cs.update_empirical_prior(
            cp, post, {"agent.a_location": jnp.asarray(2), "world.u_location": jnp.asarray(4)}
        )
    with pytest.raises(ValueError, match="unknown key"):
        cs.update_empirical_prior(
            cp, post, {"agent.a_location": jnp.asarray(2), None: jnp.asarray(4)}
        )
    keyed = cs.to_positional("actions", {"agent.a_location": jnp.asarray(2)})
    assert keyed.tolist() == [2, 2]  # the driven node takes the driver's level
    assert "None" not in str(cs.from_positional("actions", keyed))


def test_a_driven_coupling_is_priced_as_its_joint_planner_and_no_variable_pays_its_size_twice():
    """A control node and its action node are one decision variable: a driven node is a slot of
    its driver's, so the level-count code is paid once, on the driver's control node."""
    from hybrid_aif.description.mdl import description_length, log_star

    (agent, ap), (world, wp) = zoo_world()
    parts = {"agent": (agent, ap), "world": (world, wp)}
    driven, dp = couple(parts, [ACT, *SENSES])
    chosen, chp = couple(parts, SENSES)
    a = description_length(driven, params=dp)
    mind, mind_p = joint_planner(driven, dp)
    joint = description_length(mind, params=mind_p)
    assert a.total == pytest.approx(joint.total) and a.total == pytest.approx(111.2671, abs=1e-4)
    assert a.total == pytest.approx(description_length(chosen, params=chp).total - 7.7275, abs=1e-4)
    paid = [k for k in a.per_address if k.startswith("control:")]
    assert paid == ["control:agent.u_location"]  # the driven slot pays nothing
    assert a.per_address[paid[0]].graph == pytest.approx(log_star(5))
    assert not [k for k in a.per_address if "policy" in k or "habit" in k]


def test_couple_refusals_name_couple():
    (a, apar), (w, wpar) = zoo_world()
    parts = {"agent": (a, apar), "world": (w, wpar)}
    with pytest.raises(ValueError, match="^couple\\(\\): Identify names 'world.reward2'"):
        couple(parts, [Identify("agent.reward", "world.reward2")])
    with pytest.raises(ValueError, match="^couple\\(\\)"):
        couple(parts, [ACT, Clamp("world.u_location", 1)])


def test_a_coupling_lists_an_undriven_control_node_among_its_in_ports():
    cs, cp = zoo_coupling()
    assert "agent.u_location" in cs.ports(cp).ins and "world.u_location" not in cs.ports(cp).ins
