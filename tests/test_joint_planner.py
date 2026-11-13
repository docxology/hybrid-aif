"""``joint_planner``: a node of a merged or coupled System planned as one mind.

A merge states no planning problem of its own: it records each leaf's configuration and planner
parameters and holds ``UNSET`` where the leaves differ. ``joint_planner`` composes them. The
tests pin, in order: the plan of a merge whose leaves agree is the hand-flattened declaration, the
control identification collapse against a hand-built System with one shared control node, the
composition of the habit, precision, durations and scoring, the configuration (argument, then
the composite, then the leaves), a nested node of a merged coupling, and every refusal.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical
from hybrid_aif.composition import (
    Identify,
    couple,
    driven_nodes,
    joint_planner,
    merge,
    subsystem,
)
from hybrid_aif.system import UNSET, System
from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world, hand_flattened, merged
from tests.test_merge_nodes import SAME_STEP, same_step_merge
from tests.test_splice import ACT, CUES, SENSES, zoo_coupling, zoo_world

KEY = jax.random.PRNGKey(0)


def _leaves(a, b):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _two_nodes(**changes):
    """The zoo pair with both control nodes live (the agent's and the world's own)."""
    (agent, ap), (world, wp) = agent_and_world(1)
    return (agent, ap), (world, dataclasses.replace(wp, **changes))


def _merge_two(**world_changes):
    (agent, ap), (world, wp) = _two_nodes(**world_changes)
    return merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        list(OBSERVATION_BINDINGS),
    )


# ------------------------------------------------------------------------- a merge that agrees


def test_the_joint_planner_of_a_merge_is_the_hand_flattened_declaration():
    composite, cp = merged()
    mind, mp = joint_planner(composite, cp)
    hand, hp = hand_flattened()
    for field in (f.name for f in dataclasses.fields(System) if f.name != "subsystems"):
        assert getattr(mind, field) == getattr(hand, field), field
    _leaves(mp, hp)
    assert mind.subsystems == composite.subsystems and mp.parts == {}


def test_the_faces_of_a_merge_whose_leaves_agree_are_the_joint_planners():
    composite, cp = merged()
    mind, mp = joint_planner(composite, cp)
    beliefs, _ = mind.infer_states(mp, {m: jnp.zeros((), jnp.int32) for m in mind.modality_names})
    q_joint, _ = mind.infer_policies(mp, beliefs)
    q_merge, _ = composite.infer_policies(cp, beliefs)
    assert jnp.array_equal(q_joint.logits, q_merge.logits)


# ----------------------------------------------------------------------- the control collapse


def test_the_same_step_link_of_a_merge_collapses_into_the_drivers_node():
    composite, cp = same_step_merge()
    assert composite.names.controls == ("agent.u_location", "world.u_location")
    assert composite.action_names == ("agent.a_location", None)
    assert driven_nodes(composite) == {"world.u_location": "agent.a_location"}
    mind, mp = joint_planner(composite, cp)
    assert mind.names.controls == ("agent.u_location",)
    assert mind.action_names == ("agent.a_location",)
    assert dict(mind.subsystems)["world"].identified == (("world.u_location", "agent.u_location"),)
    assert "world.u_location" not in dict(mind.observes)
    #  every factor that read the driven node reads the driver's
    assert mind.B_deps[2][-1] == mind.B_deps[0][-1] == 4
    assert driven_nodes(mind) == {}


def test_a_node_whose_driver_is_outside_the_node_stays_driven_and_takes_an_input():
    (agent, ap), (world, wp) = agent_and_world(1)
    left, lp = merge({"agent": agent}, {"agent": ap})
    right, rp = merge({"world": world}, {"world": wp})
    outer, op = merge(
        {"left": left, "right": right},
        {"left": lp, "right": rp},
        [Identify("left.agent.a_location", "right.world.u_location")],
    )
    whole, wholep = joint_planner(outer, op)
    assert whole.names.controls == ("left.agent.u_location",)
    held, hp = joint_planner(outer, op, "right")
    assert held.names.controls == ("right.world.u_location",) and held.action_names == (None,)
    assert driven_nodes(held) == {}  # the view keeps the node as an input without its driver
    sliced, sp = joint_planner(outer, op, "right", inputs={"left.agent.a_location": 2})
    assert sliced.names.controls == () and sliced.policy_len == 1
    assert sp.transition["right.world.location"].log_table.shape[-1] == 1


def test_a_collapse_that_makes_a_factor_read_one_node_twice_is_refused():
    from hybrid_aif import ConditionalCategorical
    from hybrid_aif.system import SystemParams
    from tests.test_subsystem import _binary

    a, ap = _binary("a", True)
    two = jnp.full((2, 2, 2, 2), 0.5)  # child, prev, u0, u1
    one = jnp.zeros((2, 2, 2)).at[0, 0, :].set(1.0).at[1, 1, :].set(1.0)
    w = System(
        B_deps=((0,), (1,)),
        controlled=(True, True),
        B_control_deps=((0, 1), (1,)),
        policy_len=1,
        epistemic=False,
        state_names=("w0", "w1"),
    )
    wp = SystemParams(
        transition={
            "w0": ConditionalCategorical(log_table=jnp.log(two), parent_ndims=3),
            "w1": ConditionalCategorical(log_table=jnp.log(one), parent_ndims=2),
        },
        initial_state_prior={f: Categorical(logits=jnp.zeros(2)) for f in ("w0", "w1")},
    )
    composite, cp = merge(
        {"a": a, "w": w},
        {"a": ap, "w": wp},
        [Identify("a.a_a", "w.u_w0"), Identify("a.a_a", "w.u_w1")],  # one out-port, two in-ports
    )
    assert driven_nodes(composite, "w") == {"w.u_w0": "a.a_a", "w.u_w1": "a.a_a"}
    with pytest.raises(ValueError, match="reads both .*which the collapse makes one node"):
        joint_planner(composite, cp)


def test_a_kernel_on_a_collapsed_drivers_action_node_is_refused():
    composite, cp = same_step_merge()
    kernel = jnp.full((5, 5), 0.05).at[jnp.arange(5), jnp.arange(5)].set(0.8)
    cp = dataclasses.replace(cp, actuator={"agent.a_location": kernel})
    with pytest.raises(ValueError, match="params.actuator kernel"):
        joint_planner(composite, cp)


def test_a_target_on_a_collapsed_node_is_refused_naming_the_drivers_node():
    composite, cp = same_step_merge()
    target = Categorical(logits=jnp.zeros(5))
    cp = dataclasses.replace(cp, state_targets={"world.u_location": target})
    with pytest.raises(ValueError, match="declare the target on 'agent.u_location'"):
        joint_planner(composite, cp)


# --------------------------------------------------------------------- configuration, in order


def test_leaves_that_differ_hold_unset_in_a_merge_and_joint_planner_asks_for_one_value():
    (agent, ap), (world, wp) = _two_nodes()
    world = dataclasses.replace(world, policy_len=3)
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        list(OBSERVATION_BINDINGS),
    )
    assert composite.policy_len is UNSET and composite.gamma == agent.gamma
    assert {n: dict(s.config)["policy_len"] for n, s in composite.subsystems} == {
        "agent": 1,
        "world": 3,
    }
    with pytest.raises(ValueError, match="coupling.*leaf by leaf"):
        composite.policies(cp)
    with pytest.raises(ValueError, match="leaves of 'the System' disagree on policy_len"):
        joint_planner(composite, cp)
    mind, mp = joint_planner(composite, cp, policy_len=2)
    assert mind.policy_len == 2 and mind.gamma == agent.gamma
    assert subsystem(composite, cp, "agent")[0].policy_len == 1  # a leaf's view keeps its own
    assert subsystem(composite, cp, "world")[0].policy_len == 3


def test_the_composites_own_field_beats_the_leaves_and_an_argument_beats_both():
    (agent, ap), (world, wp) = _two_nodes()
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        list(OBSERVATION_BINDINGS),
    )
    edited = dataclasses.replace(composite, gamma=4.0)
    assert joint_planner(edited, cp)[0].gamma == 4.0
    assert joint_planner(edited, cp, gamma=2.0)[0].gamma == 2.0
    assert joint_planner(composite, cp)[0].gamma == agent.gamma


def test_a_configuration_name_that_is_no_override_is_refused():
    composite, cp = merged()
    with pytest.raises(TypeError, match="unexpected configuration"):
        joint_planner(composite, cp, habit=1)


# ------------------------------------------------------------------- habit, precision, clock


def test_the_habits_of_the_leaves_sum_in_log_space_over_the_joint_policy_set():
    habit_a, habit_w = jnp.linspace(-1.0, 1.0, 5), jnp.linspace(1.0, -2.0, 5)
    (agent, ap), (world, wp) = _two_nodes(policy_prior=habit_w)
    ap = dataclasses.replace(ap, policy_prior=habit_a)
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        list(OBSERVATION_BINDINGS),
    )
    assert cp.policy_prior is None and set(cp.parts) == {"agent", "world"}
    mind, mp = joint_planner(composite, cp)
    want = (habit_a[:, None] + habit_w[None, :]).reshape(-1)
    np.testing.assert_allclose(np.asarray(mp.policy_prior), np.asarray(want), rtol=1e-6)
    assert mp.parts == {}


def test_a_habit_the_composite_states_is_authoritative():
    (agent, ap), (world, wp) = _two_nodes(policy_prior=jnp.zeros(5))
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        list(OBSERVATION_BINDINGS),
    )
    stated = dataclasses.replace(cp, policy_prior=jnp.arange(25.0))
    np.testing.assert_array_equal(
        np.asarray(joint_planner(composite, stated)[1].policy_prior), np.arange(25.0)
    )


def test_a_habit_over_a_node_a_collapse_removed_is_refused():
    (agent, ap), (world, wp) = agent_and_world(1)
    wp = dataclasses.replace(wp, policy_prior=jnp.zeros(5))
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        [SAME_STEP, *OBSERVATION_BINDINGS],
    )
    with pytest.raises(ValueError, match="not chosen by the joint planner"):
        joint_planner(composite, cp)


def test_precision_is_the_one_value_the_leaves_declare():
    (agent, ap), (world, wp) = _two_nodes(precision=jnp.asarray(7.0))
    ap = dataclasses.replace(ap, precision=jnp.asarray(7.0))
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        list(OBSERVATION_BINDINGS),
    )
    assert cp.precision is None
    assert float(joint_planner(composite, cp)[1].precision) == 7.0
    ap_other = dataclasses.replace(ap, precision=jnp.asarray(3.0))
    different, dp = merge(
        {"agent": agent, "world": world},
        {"agent": ap_other, "world": wp},
        list(OBSERVATION_BINDINGS),
    )
    with pytest.raises(ValueError, match="different params.precision values"):
        joint_planner(different, dp)
    one, op = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": dataclasses.replace(wp, precision=None)},
        list(OBSERVATION_BINDINGS),
    )
    with pytest.raises(ValueError, match="declare params.precision and the others do not"):
        joint_planner(one, op)


def test_durations_are_one_clock_on_a_node_the_planner_chooses():
    clock = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    composite, cp = _merge_two(durations={"u_location": clock})
    assert cp.durations == {} and set(cp.parts["world"].durations) == {"world.u_location"}
    mind, mp = joint_planner(composite, cp)
    assert set(mp.durations) == {"world.u_location"}
    (agent, ap), (world, wp) = _two_nodes(durations={"u_location": clock})
    ap = dataclasses.replace(ap, durations={"u_location": clock})
    two, tp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        list(OBSERVATION_BINDINGS),
    )
    with pytest.raises(ValueError, match="one clock"):
        joint_planner(two, tp)


def test_a_node_the_collapse_removes_cannot_carry_the_clock():
    (agent, ap), (world, wp) = agent_and_world(1)
    wp = dataclasses.replace(wp, durations={"u_location": jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])})
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        [SAME_STEP, *OBSERVATION_BINDINGS],
    )
    with pytest.raises(ValueError, match="removes control node 'world.u_location'"):
        joint_planner(composite, cp)


def test_two_leaves_that_declare_scoring_are_refused_and_one_is_carried():
    from tests.test_hybrid_lg_system import make_params, make_system
    from tests.test_planning_with_observations import _probe_row

    _probe_row()
    scored = make_system(terms=("epistemic_probe",), policy_len=2)
    sp = dataclasses.replace(make_params(), scoring={"epistemic_probe": jnp.asarray(2.0)})
    plain, pp = make_system(policy_len=2), make_params()
    composite, cp = merge({"a": scored, "b": plain}, {"a": sp, "b": pp})
    assert "epistemic_probe" not in " ".join(composite.terms) and composite.scoring == ()
    mind, mp = joint_planner(composite, cp)
    assert dict(mind.scoring) == dict(scored.scoring) and "epistemic_probe" in mp.scoring
    assert any(t.startswith("epistemic_probe") for t in mind.terms)
    both, bp = merge({"a": scored, "b": scored}, {"a": sp, "b": sp})
    with pytest.raises(NotImplementedError, match="leaves \\['a', 'b'\\] declare scoring"):
        joint_planner(both, bp)


# ------------------------------------------------------------------------------- the refusals


def test_a_leaf_a_plain_system_and_an_unknown_node_are_refused():
    composite, cp = merged()
    with pytest.raises(ValueError, match="'agent' is a leaf, which is its own planner"):
        joint_planner(composite, cp, "agent")
    with pytest.raises(ValueError, match="unknown node 'nobody'"):
        joint_planner(composite, cp, "nobody")
    (agent, ap), _ = agent_and_world()
    with pytest.raises(ValueError, match="no components"):
        joint_planner(agent, ap)


def test_a_policy_module_and_a_continuous_node_do_not_compose_a_habit():
    (agent, ap), (world, wp) = _two_nodes()
    module = lambda beliefs: jnp.zeros(5)  # noqa: E731
    ap = dataclasses.replace(ap, policy_prior=module)
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        list(OBSERVATION_BINDINGS),
    )
    with pytest.raises(NotImplementedError, match="policy module"):
        joint_planner(composite, cp)


def test_a_hand_call_over_a_coupling_plans_a_mirror_as_a_self_model():
    cs, cp = zoo_coupling()
    mind, mp = joint_planner(cs, cp)
    assert mind.names.controls == ("agent.u_location",)
    assert set(dict(mind.observes)) == {"agent.location_obs", "agent.reward", "agent.cue"}
    assert [s.boundary for _, s in mind.subsystems] == ["blanket", "blanket"]
    obs = {f"world.{m}": jnp.zeros((), jnp.int32) for m in CUES}
    q, _ = mind.infer_policies(mp, mind.infer_states(mp, obs)[0])
    assert q.logits.shape == (5,)


def test_the_joint_planner_of_a_node_of_a_merged_coupling_plans_that_node_alone():
    cs, cp = zoo_coupling()
    (agent, ap), _ = zoo_world()
    outer, op = merge({"pair": cs, "solo": agent}, {"pair": cp, "solo": ap})
    mind, mp = joint_planner(outer, op, "pair")
    assert all(name.startswith("pair.") for name in mind.names.states)
    assert mind.names.controls == ("pair.agent.u_location",)
    obs = {
        "pair.world.location_obs": jnp.zeros((), jnp.int32),
        "pair.world.reward": jnp.zeros((), jnp.int32),
        "pair.world.cue": jnp.zeros((), jnp.int32),
    }
    beliefs, _ = mind.infer_states(mp, obs)
    action, _, _ = mind.act(KEY, mp, beliefs)
    assert set(action) == {"pair.agent.a_location"}


def test_couple_and_merge_of_the_same_ports_collapse_to_one_joint_problem():
    (agent, ap), (world, wp) = zoo_world()
    coupled, cp = couple({"agent": (agent, ap), "world": (world, wp)}, [ACT, *SENSES])
    merged_, mp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        [ACT, *SENSES],
    )
    a, ap_ = joint_planner(coupled, cp)
    b, bp = joint_planner(merged_, mp)
    for field in (
        "A_deps",
        "B_deps",
        "controls",
        "action_names",
        "state_names",
        "control_names",
        "observes",
    ):
        assert getattr(a, field) == getattr(b, field), field
    _leaves(ap_, bp)
