"""The binding kinds of a merge: ``Link``, ``Identify`` and ``Clamp``.

A ``Link`` is an arc of the graphical model of the whole, read from its two ends and its
``realise``: an action node into a control node (the same-step identification), a state factor
into a control node (the relay), state factors into a modality (the modality's complete parent
list), one state factor into a modality with a ``realise`` (a belief read) and state factors into
``"terminal:<factor>"``. ``Identify`` makes two parameter variables one, ``Clamp`` holds a
control node at a constant. The refusals of each form are named here.
"""

import dataclasses

import jax.numpy as jnp
import pytest

import hybrid_aif.composition as composition
from hybrid_aif import ConditionalCategorical, Grid
from hybrid_aif.composition import Clamp, Identify, Link, joint_planner, merge, nested, subsystem
from hybrid_aif.reads import Emission
from tests.test_emissions import world as emitting_world
from tests.test_merge import DRIVE, OBSERVATION_BINDINGS, agent_and_world, merged
from tests.test_sibling_read import EMITTED, READ, SOURCE, _composite, _rover, _scheduler


def test_the_old_binding_names_are_gone():
    for name in (
        "ObservationLink",
        "SiblingRead",
        "Wire",
        "ActionControl",
        "SharedParameter",
        "BoundaryPrior",
        "TargetEmission",
    ):
        assert not hasattr(composition, name), name
        assert name not in composition.__all__


# ------------------------------------------------------------- an action node into a control node


def _world_pair():
    """Two components that each choose a control node: the T-maze agent and a copy."""
    (agent, ap), (world, wp) = agent_and_world()
    return {"agent": agent, "world": world}, {"agent": ap, "world": wp}


def test_a_link_from_an_action_node_to_a_control_node_makes_the_two_nodes_one():
    (agent, ap), _ = agent_and_world()
    other = dataclasses.replace(agent, modality_names=agent.modality_names)
    composite, params = joint_planner(
        *merge(
            {"a": agent, "b": other},
            {"a": ap, "b": ap},
            [Link("a.a_location", "b.u_location")],
        )
    )
    assert composite.names.controls == ("a.u_location",)
    assert composite.names.actions == ("a.a_location",)
    subs = dict(composite.subsystems)
    assert subs["b"].identified == (("b.u_location", "a.u_location"),)
    assert subs["b"].boundary == "blanket" and nested(composite, "b")
    names = composite.node_names
    assert composite.B_deps[names.index("b.location")] == (
        names.index("b.location"),
        names.index("a.u_location"),
    )


def test_a_control_node_is_refused_as_a_source_naming_its_action_node():
    (agent, ap), _ = agent_and_world()
    with pytest.raises(ValueError, match="a control node, which is no port.*'a.a_location'"):
        merge(
            {"a": agent, "b": agent},
            {"a": ap, "b": ap},
            [Link("a.u_location", "b.u_location")],
        )


def test_a_link_from_an_action_node_takes_no_realise_and_no_kernel():
    (agent, ap), _ = agent_and_world()
    subs, ps = {"a": agent, "b": agent}, {"a": ap, "b": ap}
    with pytest.raises(ValueError, match="a realised port: it is realised once"):
        merge(subs, ps, [Link("a.a_location", "b.u_location", realise="mode")])
    with pytest.raises(ValueError, match="a kernel without realise"):
        merge(subs, ps, [Link("a.a_location", "b.u_location", kernel="identity")])


def test_an_action_node_or_an_emission_into_a_modality_is_a_couplings_link():
    (agent, ap), _ = agent_and_world()
    subs, ps = {"a": agent, "b": agent}, {"a": ap, "b": ap}
    with pytest.raises(ValueError, match="coupling's wire, which couple\\(\\) carries"):
        merge(subs, ps, [Link("a.a_location", "b.cue")])
    emitter, eparams = emitting_world()
    with pytest.raises(ValueError, match="an emission.*coupling's wire"):
        merge(
            {"w": emitter, "b": agent},
            {"w": eparams, "b": ap},
            [Link("w.fix", "b.cue")],
        )
    with pytest.raises(ValueError, match="the emission 'w.fix' into the control node"):
        merge(
            {"w": emitter, "b": agent},
            {"w": eparams, "b": ap},
            [Link("w.fix", "b.u_location")],
        )


def test_an_unknown_source_or_target_names_what_a_link_takes():
    (agent, ap), _ = agent_and_world()
    subs, ps = {"a": agent, "b": agent}, {"a": ap, "b": ap}
    with pytest.raises(ValueError, match="the source is unknown"):
        merge(subs, ps, [Link("a.locaiton", "b.u_location")])
    with pytest.raises(ValueError, match="the target 'b.nowhere' is unknown"):
        merge(subs, ps, [Link("a.location", "b.nowhere")])
    with pytest.raises(ValueError, match="one source, not a tuple of 2"):
        merge(subs, ps, [Link(("a.location", "a.context"), "b.u_location")])


# ----------------------------------------------------------------------------- the relay


def test_a_link_from_a_state_factor_into_a_control_node_is_the_relay():
    composite, _ = merged()
    relay = dict(composite.subsystems)["world"].relayed
    assert relay == (("world.u_location", "agent.location"),)
    assert DRIVE == Link("agent.location", "world.u_location")
    assert dict(composite.subsystems)["world"].boundary == "shared"
    assert "world.u_location" not in composite.names.controls


def test_a_relay_takes_no_realise():
    (agent, ap), (world, wp) = agent_and_world()
    with pytest.raises(ValueError, match="a relay from a state factor.*takes no\\s+realise"):
        merge(
            {"agent": agent, "world": world},
            {"agent": ap, "world": wp},
            [Link("agent.location", "world.u_location", realise="mode")],
        )


# ------------------------------------------------------------------ a modality's parent list


def test_a_structural_link_states_a_modalitys_complete_parent_list():
    composite, _ = merged()
    assert composite.A_deps == ((2,), (2, 3), (2, 3))
    bound = dict(composite.subsystems)["agent"].bound_modalities
    assert bound == ((0, (2,)), (1, (2, 3)), (2, (2, 3)))
    assert dict(composite.subsystems)["agent"].boundary == "shared"
    single = merged(
        bindings=[
            DRIVE,
            Link("world.location", "agent.location_obs"),
            *OBSERVATION_BINDINGS[1:],
        ]
    )[0]
    assert single.A_deps == composite.A_deps


def test_a_structural_link_refuses_a_wrong_arity_a_wrong_support_and_a_kernel():
    (agent, ap), (world, wp) = agent_and_world()
    subs, ps = {"agent": agent, "world": world}, {"agent": ap, "world": wp}
    with pytest.raises(ValueError, match="names 1 parent.*has 2 parent axis"):
        merge(subs, ps, [Link(("world.location",), "agent.reward")])
    with pytest.raises(ValueError, match="support mismatch"):
        merge(subs, ps, [Link(("world.context", "world.location"), "agent.reward")])
    with pytest.raises(ValueError, match="a kernel without realise"):
        merge(
            subs,
            ps,
            [
                Link(
                    ("world.location",),
                    "agent.location_obs",
                    kernel=Grid(shape=(2,), low=0.0, high=1.0),
                )
            ],
        )


# ------------------------------------------------------------------------------ a belief read


def test_a_link_with_a_realise_is_a_belief_read():
    composite, _ = _composite()
    assert composite.observes == ((READ, EMITTED),)
    assert dict(composite.emissions)[EMITTED] == Emission((SOURCE,), "mode", composition_grid())
    assert dict(composite.subsystems)["mission"].reads == (
        (READ, SOURCE, composition_grid(), "mode"),
    )
    assert nested(composite, "mission") and nested(composite, "rover")


def composition_grid():
    from tests.test_sibling_read import GRID

    return GRID


def test_a_belief_read_without_a_kernel_reads_through_the_identity():
    (sr, pr), (ss, ps) = _rover(), _scheduler()
    composite, _ = merge(
        {"rover": sr, "mission": dataclasses.replace(ss, A_deps=((0,),))},
        {"rover": pr, "mission": ps},
        [Link("rover.x", "mission.cell_obs", realise="mode", kernel=composition_grid())],
    )
    assert dict(composite.emissions)[EMITTED].kernel == composition_grid()
    sched, sp = _scheduler()
    sched = dataclasses.replace(sched, modality_names=("seen",))
    sp = dataclasses.replace(sp, likelihood={"seen": sp.likelihood["cell_obs"]})
    categorical, cp = _scheduler(shared=True)
    composite, _ = merge(
        {"a": sched, "b": categorical},
        {"a": sp, "b": cp},
        [Link("a.visit", "b.cell_obs", realise="stats")],
    )
    (emission,) = (e for n, e in composite.emissions if n == "a.cell_obs")
    assert emission == Emission(("a.visit",), "stats", "identity")


def test_a_belief_read_refusals_are_named():
    (sr, pr), (ss, ps) = _rover(), _scheduler()
    subs, params = {"rover": sr, "mission": ss}, {"rover": pr, "mission": ps}
    with pytest.raises(ValueError, match="realise must be one of"):
        merge(subs, params, [Link(SOURCE, READ, realise="median", kernel=composition_grid())])
    with pytest.raises(ValueError, match="one source, a state factor"):
        merge(subs, params, [Link((SOURCE, SOURCE), READ, realise="mode")])
    with pytest.raises(ValueError, match="the source is unknown"):
        merge(subs, params, [Link("mission.u_nothing", READ, realise="mode")])
    with pytest.raises(ValueError, match="already observes a variable"):
        merge(
            subs,
            params,
            [
                Link(SOURCE, READ, realise="mode", kernel=composition_grid()),
                Link(SOURCE, READ, realise="sample", kernel=composition_grid()),
            ],
        )
    with pytest.raises(ValueError, match="both in component 'rover'"):
        merge(subs, params, [Link("rover.x", "rover.obs", realise="mode")])


def test_a_subsystem_view_leaves_a_read_modality_an_observation():
    composite, params = _composite()
    view, _ = subsystem(composite, params, "mission")
    assert view.observes == ()


# ----------------------------------------------------------------------- the terminal link


def test_a_link_to_a_terminal_target_declares_the_factors_h_deps():
    (agent, ap), (world, wp) = agent_and_world()
    target = ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])),
        parent_ndims=1,
    )
    wp = dataclasses.replace(wp, terminal_targets={"context": target})
    subs, ps = {"agent": agent, "world": world}, {"agent": ap, "world": wp}
    composite, cp = merge(
        subs, ps, [*OBSERVATION_BINDINGS, Link(("agent.location",), "terminal:world.context")]
    )
    assert composite.H_deps == ((), (), (), (0,), (), ())
    assert dict(composite.subsystems)["world"].target_emissions == ((3, (0,)),)
    one, _ = merge(
        subs, ps, [*OBSERVATION_BINDINGS, Link("agent.location", "terminal:world.context")]
    )
    assert one.H_deps == composite.H_deps


def test_a_single_name_source_is_one_parent_and_the_terminal_key_is_reserved():
    from hybrid_aif import System
    from hybrid_aif.system import _KEY_KINDS

    assert "terminal" in _KEY_KINDS
    with pytest.raises(ValueError, match="reserved"):
        System(B_deps=((),), state_names=("terminal",), policy_len=1)


def test_a_terminal_link_refuses_a_realise_a_port_and_a_second_declaration():
    (agent, ap), (world, wp) = agent_and_world()
    subs, ps = {"agent": agent, "world": world}, {"agent": ap, "world": wp}
    with pytest.raises(ValueError, match="structural.*takes no realise"):
        merge(subs, ps, [Link(("agent.location",), "terminal:world.context", realise="mode")])
    with pytest.raises(ValueError, match="a terminal target is conditioned on state factors"):
        merge(subs, ps, [Link("agent.a_location", "terminal:world.context")])
    with pytest.raises(ValueError, match="unknown factor 'world.nowhere'"):
        merge(subs, ps, [Link(("agent.location",), "terminal:world.nowhere")])
    with pytest.raises(ValueError, match="already declares H_deps"):
        merge(
            subs,
            ps,
            [
                Link(("agent.location",), "terminal:world.context"),
                Link(("agent.context",), "terminal:world.context"),
            ],
        )


# ------------------------------------------------------------------------- Identify and Clamp


def test_identify_makes_two_parameter_variables_one_and_names_its_refusals():
    sr, pr = _rover(shared=True)
    ss, ps = _scheduler(shared=True)
    subs, params = {"rover": sr, "mission": ss}, {"rover": pr, "mission": ps}
    composite, cp = merge(subs, params, [Identify("rover.field", "mission.field")])
    assert set(cp.parameters) == {"rover.field"}
    assert all(h == "rover.field" for _, _, h in cp.references)
    assert dict(composite.subsystems)["mission"].boundary == "shared"
    with pytest.raises(ValueError, match="Identify names 'rover.nope'"):
        merge(subs, params, [Identify("rover.nope", "mission.field")])
    with pytest.raises(ValueError, match="Identify names 'rover.field' on both sides"):
        merge(subs, params, [Identify("rover.field", "rover.field")])
    with pytest.raises(ValueError, match="same family|table shape"):
        merge(
            subs,
            {
                "rover": pr,
                "mission": dataclasses.replace(
                    ps,
                    parameters={
                        "field": pr.parameters["field"].__class__(
                            concentration=jnp.ones((3, 1)), parent_ndims=1
                        )
                    },
                ),
            },
            [Identify("rover.field", "mission.field")],
        )


def test_clamp_holds_a_control_node_at_a_level_or_a_distribution():
    delta, dp = merged(bindings=[*OBSERVATION_BINDINGS, Clamp("world.u_location", 3)])
    assert dict(delta.subsystems)["world"].terminated == ("world.u_location",)
    flat, _ = merged(bindings=[*OBSERVATION_BINDINGS, Clamp("world.u_location", jnp.full(5, 0.2))])
    assert delta.names.controls == flat.names.controls == ("agent.u_location",)
    with pytest.raises(ValueError, match="unknown control node 'world.nowhere'"):
        merged(bindings=[Clamp("world.nowhere", 0)])


def test_an_unknown_binding_is_refused_naming_the_three_kinds():
    with pytest.raises(TypeError, match="the kinds are Link .* Identify .* Clamp"):
        merged(bindings=[("agent.location", "world.u_location")])


def test_terminal_is_a_key_kind_of_links_only_and_no_epistemic_selector():
    from tests.test_system import tmaze_system

    system, params = tmaze_system()
    for key, match in (
        ("terminal:location", "unknown slot key 'terminal:location'"),
        ("terminal:*", "unknown first segment 'terminal'"),
    ):
        chosen = dataclasses.replace(system, epistemic=(key,))
        with pytest.raises(ValueError, match=match):
            chosen.planner_context(params, chosen.initial_priors(params))
