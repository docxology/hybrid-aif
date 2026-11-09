"""A merged composite with one step per view (design: nesting-and-coupling.md N4): ``particle`` as
one step for the wired form, and ``run``'s composite form, which share a per-round body. The
parameter home is written back each round a step learns and the views are re-taken from it; a
driving component's decision is routed into the driven view; ``hold`` is the clock inside it."""

import dataclasses
import importlib.util
import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import ConditionalDirichlet
from hybrid_aif.composition import (
    HoldCarry,
    Identify,
    Link,
    MindCarry,
    ParticleCarry,
    agent_step,
    couple,
    hold,
    merge,
    particle,
    run,
    subsystem,
    write_back,
)
from hybrid_aif.composition.merge import _partner_inputs, retake
from tests.test_merge import OBSERVATION_BINDINGS, merged
from tests.test_sibling_read import _composite, _rover
from tests.test_subsystem_conditional_target import _conditional_target, _two_tier
from tests.test_subsystem_conditional_target import _rover as _target_rover

KEY = jax.random.PRNGKey(0)
NODE = "agent.u_location"
ACTION = "agent.a_location"
SAME_STEP = Link("agent.a_location", "world.u_location")

_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "compose_minds.py"
_SPEC = importlib.util.spec_from_file_location("compose_minds", _PATH)
cm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cm)


def _same(a, b):
    la, lb = jax.tree_util.tree_flatten(a), jax.tree_util.tree_flatten(b)
    return la[1] == lb[1] and all(
        np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la[0], lb[0], strict=True)
    )


# ------------------------------------------------------------- the zoo pair: routing


def _zoo():
    comp, cp = merged(bindings=[SAME_STEP, *OBSERVATION_BINDINGS])
    obs, _ = comp.as_env(cp)[0].reset(KEY, comp.as_env(cp)[1])
    return comp, cp, obs


def _recording(log):
    def step(key, obs, carry, params):
        log.append(params)
        return {}, carry, params

    return step


def test_the_view_that_reads_a_partner_s_node_names_it():
    comp, _, _ = _zoo()
    assert _partner_inputs(comp, "world") == (ACTION,)
    assert _partner_inputs(comp, "agent") == ()


def _level_view(comp, cp, level):
    return subsystem(comp, cp, "world", inputs={ACTION: level} if level is not None else None)[1]


def test_a_driver_listed_first_is_seen_by_the_driven_view_the_same_round():
    comp, cp, obs = _zoo()
    log: list = []
    agent = agent_step(subsystem(comp, cp, "agent")[0], selection="deterministic")
    p = particle(comp, cp, {"agent": agent, "world": _recording(log)})
    action, carry, _ = p(KEY, obs, ParticleCarry(), cp)
    level = int(action["agent.a_location"])
    assert carry.standing == {NODE: level}
    assert _same(log[0], _level_view(comp, cp, level))


def test_a_later_round_sees_the_last_level_when_the_driver_is_listed_after():
    comp, cp, obs = _zoo()
    log: list = []
    agent = agent_step(subsystem(comp, cp, "agent")[0], selection="deterministic")
    p = particle(comp, cp, {"world": _recording(log), "agent": agent})
    action, carry, _ = p(KEY, obs, ParticleCarry(), cp)
    assert _same(log[0], _level_view(comp, cp, None))  # nothing chosen yet: uniform
    p(KEY, obs, carry, cp)
    assert _same(log[1], _level_view(comp, cp, int(action["agent.a_location"])))


def test_one_compiled_program_serves_every_routed_level():
    comp, cp, _ = _zoo()
    traces: list = []

    @eqx.filter_jit
    def probe(params):
        traces.append(1)
        return jnp.zeros(())

    views = [_level_view(comp, cp, level) for level in (None, *range(5))]
    for view in views:
        probe(view)
    assert len(traces) == 1  # the uniform round-0 view shares the program
    systems = [subsystem(comp, cp, "world", inputs={ACTION: lv})[0] for lv in (0, 3)]
    assert eqx.tree_equal(*systems)


# ------------------------------------------------------ the shared field: write-back


def _bump(view, params):
    """``params`` with the field's concentrations one higher: what a learner returns."""
    held = view._params_in(params, "test")
    field = held.parameters["rover.field"]
    new = ConditionalDirichlet(concentration=field.concentration + 1.0, parent_ndims=1)
    out = dataclasses.replace(held, parameters={**held.parameters, "rover.field": new})
    for slot, i, home in held.references:
        container = list(getattr(out, slot))
        container[i] = out.parameters[home]
        out = dataclasses.replace(out, **{slot: tuple(container)})
    return view._params_out(out)


def _shared():
    comp, cp = _composite(shared=True, bindings=[Identify("rover.field", "mission.field")])
    rover, mission = subsystem(comp, cp, "rover")[0], subsystem(comp, cp, "mission")[0]
    return comp, cp, rover, mission


def _learner(view, log=None):
    inner = agent_step(view)

    def step(key, obs, carry, params):
        action, carry, _ = inner(key, obs, carry, params)
        if log is not None:
            log.append(params)
        return action, carry, _bump(view, params)

    return step


def _field(params):
    return np.asarray(params.parameters["rover.field"].concentration)


def _watcher(view, seen):
    inner = agent_step(view)

    def step(key, obs, carry, params):
        seen.append(params)
        return inner(key, obs, carry, params)

    return step


def test_a_learning_view_is_written_back_each_round_and_the_other_view_is_retaken():
    comp, cp, rover, mission = _shared()
    obs = {"rover.obs": jnp.asarray([1.0, 1.0])}
    for order, offset in ((("rover", "mission"), 1), (("mission", "rover"), 0)):
        seen: list = []
        minds = {"rover": _learner(rover), "mission": _watcher(mission, seen)}
        p = particle(comp, cp, {k: minds[k] for k in order})
        carry, home = ParticleCarry(), cp
        for t in range(3):
            _, carry, home = p(KEY, obs, carry, home)
            assert np.array_equal(_field(home), _field(cp) + t + 1)
            # the reader sees this round's learning if the learner is listed first, else last's
            assert np.array_equal(_field(seen[t]), _field(cp) + t + offset)


def test_the_home_after_each_round_is_the_hand_loop_of_write_backs_bitwise():
    comp, cp, rover, mission = _shared()
    obs = {"rover.obs": jnp.asarray([1.0, 1.0])}
    p = particle(comp, cp, {"rover": _learner(rover), "mission": agent_step(mission)})
    carry, home = ParticleCarry(), cp
    run_home = cp
    for _ in range(4):
        _, carry, run_home = p(KEY, obs, carry, run_home)
        taken = subsystem(comp, home, "rover")[1]
        home = write_back(comp, home, "rover", taken, _bump(rover, taken))
        assert _same(run_home, home)


def test_retake_is_what_subsystem_would_return_now():
    comp, cp, rover, mission = _shared()
    for name, view in (("mission", mission), ("rover", rover)):
        before = subsystem(comp, cp, name)[1]
        taken = subsystem(comp, cp, "rover")[1]
        home = write_back(comp, cp, "rover", taken, _bump(rover, taken))
        assert _same(retake(comp, view, before, home), subsystem(comp, home, name)[1])


def test_write_back_refuses_a_before_taken_since_the_home_moved():
    """The rule the particle keeps by construction: it hands each mind a view taken from the
    current home, so its ``before`` is never stale."""
    comp, cp, rover, mission = _shared()
    p = particle(comp, cp, {"rover": _learner(rover), "mission": agent_step(mission)})
    stale = subsystem(comp, cp, "rover")[1]
    _, _, home = p(KEY, {"rover.obs": jnp.asarray([1.0, 1.0])}, ParticleCarry(), cp)
    with pytest.raises(ValueError, match="taken before"):
        write_back(comp, home, "rover", stale, _bump(rover, stale))


def test_a_step_that_does_not_learn_writes_nothing_and_a_learner_beside_it_still_learns():
    comp, cp, rover, mission = _shared()
    seen = []
    inner = agent_step(mission)

    def plain(key, obs, carry, params):
        seen.append(obs)
        action, carry, same = inner(key, obs, carry, params)
        assert same is params
        return action, carry, same

    p = particle(comp, cp, {"rover": _learner(rover), "mission": plain})
    _, _, home = p(KEY, {"rover.obs": jnp.asarray([1.0, 1.0])}, ParticleCarry(), cp)
    assert "mission.cell_obs" in seen[0] and seen[0]["mission.cell_obs"] is None
    assert np.array_equal(_field(home), _field(cp) + 1)
    idle = particle(comp, cp, {"rover": agent_step(rover), "mission": plain})
    _, _, same = idle(KEY, {"rover.obs": jnp.asarray([1.0, 1.0])}, ParticleCarry(), cp)
    assert same is cp  # nothing learned: the very object the runner handed in


# ------------------------------------------- the composite form of run: the same body


def _routed(order, listed, ticks=3, key=KEY):
    comp, cp, _ = _zoo()
    log: list = []
    agent = agent_step(subsystem(comp, cp, "agent")[0], selection="deterministic")
    steps = {"agent": agent, "world": _recording(log)}
    traj = run(comp, cp, {k: steps[k] for k in listed}, ticks=ticks, key=key, tick_order=order)
    levels = [int(a) for a in traj.actions["agent.a_location"]]
    return comp, cp, log, levels, traj


def test_run_simultaneous_hands_the_view_the_levels_standing_when_the_round_began():
    comp, cp, log, levels, traj = _routed("simultaneous", ("agent", "world"))
    assert _same(log[0], _level_view(comp, cp, None))  # nothing chosen before round 0: uniform
    for t in (1, 2):  # the driver is listed first, yet the reader sees the round before
        assert _same(log[t], _level_view(comp, cp, levels[t - 1]))
    assert traj.params is cp  # nothing learned


def test_run_sequential_hands_a_driver_listed_first_its_choice_of_this_round():
    comp, cp, log, levels, _ = _routed("sequential", ("agent", "world"))
    for t in range(3):
        assert _same(log[t], _level_view(comp, cp, levels[t]))


def test_run_sequential_hands_a_driver_listed_after_the_last_level():
    comp, cp, log, levels, _ = _routed("sequential", ("world", "agent"))
    assert _same(log[0], _level_view(comp, cp, None))
    for t in (1, 2):
        assert _same(log[t], _level_view(comp, cp, levels[t - 1]))


def test_run_async_runs_a_routed_view_that_has_a_uniform_view():
    comp, cp, log, levels, _ = _routed("async", ("agent", "world"), ticks=6)
    assert len(log) == 6
    views = [_level_view(comp, cp, lv) for lv in (None, *range(5))]
    assert all(any(_same(seen, v) for v in views) for seen in log)


def test_a_composite_with_a_routed_view_pins_its_trajectory():
    """A composite that learns nothing, with a view that reads a partner's node: its actions."""
    _, _, _, levels, traj = _routed("sequential", ("agent", "world"), ticks=4, key=KEY)
    assert levels == [3, 4, 1, 1]
    assert [int(x) for x in traj.states["world.location"]] == [0, 3, 3, 3, 3]


@pytest.mark.parametrize("tick_order", ["simultaneous", "sequential", "async"])
def test_run_writes_a_learner_back_and_its_sibling_sees_it_on_the_next_round(tick_order):
    comp, cp, rover, mission = _shared()
    for order, offset in ((("rover", "mission"), 1), (("mission", "rover"), 0)):
        seen: list = []
        minds = {"rover": _learner(rover), "mission": _watcher(mission, seen)}
        traj = run(comp, cp, {k: minds[k] for k in order}, ticks=3, key=KEY, tick_order=tick_order)
        assert np.array_equal(_field(traj.params), _field(cp) + 3)
        for t in range(3):
            if tick_order == "async":  # the learner is first or last, whichever the sweep drew
                assert _field(seen[t])[0, 0] - _field(cp)[0, 0] in (t, t + 1)
            else:
                # the reader sees this round's learning if the learner is listed first, else last's
                assert np.array_equal(_field(seen[t]), _field(cp) + t + offset)


def test_run_leaves_the_world_on_the_starting_params_and_its_home_is_the_hand_loop():
    comp, cp, rover, mission = _shared()
    traj = run(
        comp, cp, {"rover": _learner(rover), "mission": agent_step(mission)}, ticks=4, key=KEY
    )
    home = cp
    for _ in range(4):
        taken = subsystem(comp, home, "rover")[1]
        home = write_back(comp, home, "rover", taken, _bump(rover, taken))
    assert _same(traj.params, home)


def test_run_and_particle_share_one_round_body_on_the_same_learners():
    """The home a composite run reaches equals the home of the particle's rounds."""
    comp, cp, rover, mission = _shared()
    minds = {"rover": _learner(rover), "mission": agent_step(mission)}
    p = particle(comp, cp, minds)
    carry, home = ParticleCarry(), cp
    for _ in range(3):
        _, carry, home = p(KEY, {"rover.obs": jnp.asarray([1.0, 1.0])}, carry, home)
    traj = run(comp, cp, minds, ticks=3, key=KEY)
    assert _same(traj.params, home)


# ------------------------------------------------------------------- the wired form


def _one_component():
    sa, pa = cm.robot(2, 2)
    comp, cp = merge({"a": sa}, {"a": pa}, [])
    links = [
        Link("m.a.move", "world.a_move"),
        Link("world.a_pos", "m.a.me_obs"),
        Link("world.b_pos", "m.a.other_obs"),
        Link("world.status", "m.a.status"),
    ]
    return comp, cp, links


def test_a_one_mind_particle_coupled_to_a_world_equals_the_plain_wired_run_bitwise():
    comp, cp, links = _one_component()
    view, vp = subsystem(comp, cp, "a")
    kwargs = dict(processes={"world": cm.world((2, 2))}, links=links, fixed={"world.b_move": 0})
    plain = couple(minds={"m": (view, vp, agent_step(view, selection="stochastic"))}, **kwargs)
    inner = agent_step(view, selection="stochastic")
    pooled = couple(minds={"m": (comp, cp, particle(comp, cp, {"a": inner}))}, **kwargs)
    one, two = run(plain, ticks=12, key=KEY), run(pooled, ticks=12, key=KEY)
    assert _same(one["m"].actions, two["m"].actions)
    assert _same(one["m"].observations, two["m"].observations)
    assert _same(one["world"].states, two["world"].states)
    assert _same(one["m"].carry.beliefs, two["m"].carry.minds["a"].beliefs)
    assert two["m"].params is cp and one["m"].params is vp


def test_a_modality_the_particle_supplies_needs_no_wire_and_any_other_mind_does():
    comp, cp = _composite()
    rs, rp = _rover()
    world = rs.as_env(rp)
    links = [Link("w.obs", "m.rover.obs")]
    rover, mission = subsystem(comp, cp, "rover"), subsystem(comp, cp, "mission")
    minds = {"rover": agent_step(rover[0]), "mission": agent_step(mission[0])}
    wiring = couple(
        minds={"m": (comp, cp, particle(comp, cp, minds))}, processes={"w": world}, links=links
    )
    rec = run(wiring, ticks=3, key=KEY)
    assert set(rec["m"].observations) == {"rover.obs"}
    # a modality the composite's own emission feeds is no in-port, so no link is asked for it
    plain = couple(minds={"m": (comp, cp, agent_step(comp))}, processes={"w": world}, links=links)
    assert "mission.cell_obs" not in plain.minds["m"].system.ports(cp).ins
    with pytest.raises(ValueError, match="fed by no link"):
        couple(minds={"m": (comp, cp, agent_step(comp))}, processes={"w": world}, links=[])


def test_hold_inside_a_particle_under_the_wired_run():
    (sa, pa), (sb, pb) = cm.robot(2, 2), cm.robot(2, 2)
    pa = dataclasses.replace(pa, durations={"u_move": jnp.asarray([4.0, 2.0, 4.0])})
    comp, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    va, vb = subsystem(comp, cp, "a"), subsystem(comp, cp, "b")
    calls: list = []
    inner = agent_step(va[0], selection="stochastic")

    def counted(key, obs, carry, params):
        calls.append(1)
        return inner(key, obs, carry, params)

    minds = {
        "a": hold(counted, system=va[0]),
        "b": agent_step(vb[0], selection="stochastic"),
    }
    links = [
        Link(f"m.{s}", t) for s, t in ((("a.move", "world.a_move")), ("b.move", "world.b_move"))
    ]
    links += [
        Link("world.a_pos", "m.a.me_obs"),
        Link("world.b_pos", "m.a.other_obs"),
        Link("world.status", "m.a.status"),
        Link("world.b_pos", "m.b.me_obs"),
        Link("world.a_pos", "m.b.other_obs"),
        Link("world.status", "m.b.status"),
    ]
    wiring = couple(
        minds={"m": (comp, cp, particle(comp, cp, minds))},
        processes={"world": cm.world((2, 2))},
        links=links,
    )
    rec = run(wiring, ticks=20, key=KEY)
    moves = np.asarray(rec["m"].actions["a.move"])
    rounds, t = {0: 4, 1: 2, 2: 4}, 0
    decisions = 0
    while t < len(moves):
        decisions += 1
        assert (moves[t : t + rounds[int(moves[t])]] == moves[t]).all()
        t += rounds[int(moves[t])]
    assert len(calls) == decisions < len(moves)
    assert isinstance(rec["m"].carry.minds["a"], HoldCarry)


def test_a_held_step_returns_the_params_it_was_given_and_calls_nothing():
    sa, pa = cm.robot(2, 2)
    pa = dataclasses.replace(pa, durations={"u_move": jnp.asarray([3.0, 3.0, 3.0])})
    calls: list = []

    def fn(key, obs, carry, params):
        calls.append(1)
        return {"move": jnp.asarray(1, jnp.int32)}, carry, dataclasses.replace(params)

    held = hold(fn, system=sa)
    carry, learned = MindCarry(), []
    for _ in range(6):
        action, carry, out = held(KEY, {}, carry, pa)
        learned.append(out is not pa)
        assert int(action["move"]) == 1
    assert learned == [True, False, False, True, False, False]
    assert len(calls) == 2


# ----------------------------------------------------------------------------- refusals


def test_params_that_differ_outside_the_parameter_variables_are_refused():
    comp, cp, rover, mission = _shared()

    def learner(key, obs, carry, params):
        lik = params.likelihood["rover.obs"]
        new = dataclasses.replace(lik, noise=lik.noise * 2.0)
        return {}, carry, dataclasses.replace(params, likelihood={"rover.obs": new})

    p = particle(comp, cp, {"rover": learner, "mission": agent_step(mission)})
    with pytest.raises(ValueError, match=r"'rover'.*likelihood.*declared as a Parameter"):
        p(KEY, {"rover.obs": jnp.asarray([1.0, 1.0])}, ParticleCarry(), cp)


def test_a_wire_into_a_modality_the_particle_supplies_is_refused():
    comp, cp = _composite()
    rs, rp = _rover()
    rover, mission = subsystem(comp, cp, "rover"), subsystem(comp, cp, "mission")
    minds = {"rover": agent_step(rover[0]), "mission": agent_step(mission[0])}
    links = [Link("w.obs", "m.rover.obs"), Link("src.mission.cell_obs", "m.mission.cell_obs")]
    with pytest.raises(ValueError, match="m.mission.cell_obs. is not an in-port"):
        couple(
            minds={"m": (comp, cp, particle(comp, cp, minds))},
            processes={"w": rs.as_env(rp), "src": mission[0].as_env(mission[1])},
            links=links,
        )


def _two_tier_target():
    return _two_tier(*_target_rover(_conditional_target(), (("x", ("u_opt",)),)))


def _idle_rover(key, obs, carry, params):
    return {"rover.a_v": jnp.zeros(2, jnp.float32)}, carry, params


def _plain(key, obs, carry, params):
    return {"mission.option": jnp.asarray(1, jnp.int32)}, carry, params


def test_a_driver_listed_after_a_view_with_a_conditional_target_is_refused_at_construction():
    cs, cp = _two_tier_target()
    rover = _idle_rover
    with pytest.raises(ValueError, match="list 'mission' before 'rover'"):
        particle(cs, cp, {"rover": rover, "mission": _plain})
    p = particle(cs, cp, {"mission": _plain, "rover": rover})
    action, carry, _ = p(KEY, {}, ParticleCarry(), cp)
    assert int(action["mission.option"]) == 1 and carry.standing == {"mission.u_option": 1}


def test_run_refuses_a_conditional_target_view_unless_the_chooser_is_sure_to_go_first():
    cs, cp = _two_tier_target()
    rover = _idle_rover
    minds = {"mission": _plain, "rover": rover}  # the chooser is listed first
    for order in ("simultaneous", "async"):
        with pytest.raises(ValueError, match=r"'rover'.*'mission'.*does not guarantee"):
            run(cs, cp, minds, ticks=2, key=KEY, tick_order=order)
    traj = run(cs, cp, minds, ticks=2, key=KEY, tick_order="sequential")
    assert [int(x) for x in traj.actions["mission.option"]] == [1, 1]


def test_a_kernel_on_a_node_another_view_reads_is_refused_in_both_forms():
    comp, cp, _ = _zoo()
    kernel = 0.4 * jnp.eye(5) + 0.12
    noisy = dataclasses.replace(cp, actuator={"agent.a_location": kernel})
    agent = agent_step(subsystem(comp, cp, "agent")[0])
    minds = {"agent": agent, "world": _recording([])}
    with pytest.raises(ValueError, match=r"'world'.*'agent.u_location'.*non-identity"):
        run(comp, noisy, minds, ticks=1, key=KEY)
    with pytest.raises(ValueError, match=r"'world'.*'agent.u_location'.*non-identity"):
        particle(comp, noisy, minds)
    ident = dataclasses.replace(cp, actuator={"agent.a_location": jnp.eye(5)})
    run(comp, ident, minds, ticks=1, key=KEY)


def test_agent_step_refuses_an_all_masked_legal_on_its_first_call():
    comp, cp, obs = _zoo()
    view, vp = subsystem(comp, cp, "agent")
    legal = {NODE: jnp.zeros(5, bool)}
    step = agent_step(view, legal=legal)  # the params are not in hand yet: nothing is refused
    mods = {m: obs[m] for m in view.modality_names if m in obs}
    with pytest.raises(ValueError, match="masks every level"):
        step(KEY, mods, MindCarry(), vp)


def test_a_composite_run_refuses_params_that_differ_outside_the_parameter_variables():
    comp, cp, rover, mission = _shared()

    def learner(key, obs, carry, params):
        lik = params.likelihood["rover.obs"]
        new = dataclasses.replace(lik, noise=lik.noise * 2.0)
        return {}, carry, dataclasses.replace(params, likelihood={"rover.obs": new})

    minds = {"rover": learner, "mission": agent_step(mission)}
    with pytest.raises(ValueError, match=r"'rover'.*likelihood.*declared as a Parameter"):
        run(comp, cp, minds, ticks=1, key=KEY)


def test_a_composite_run_refuses_a_driver_listed_after_a_conditional_target_view():
    cs, cp = _two_tier_target()
    rover = _idle_rover
    with pytest.raises(ValueError, match="list 'mission' before 'rover'"):
        run(cs, cp, {"rover": rover, "mission": _plain}, ticks=1, key=KEY, tick_order="sequential")


def test_a_component_with_no_mind_is_refused():
    comp, cp = _composite()
    rover = subsystem(comp, cp, "rover")
    with pytest.raises(ValueError, match=r"\['mission'\].*no step"):
        particle(comp, cp, {"rover": agent_step(rover[0])})


# ------------------------------------------------------------- held driver, survey shape


def test_a_held_drivers_level_reaches_the_driven_view_on_held_rounds():
    comp, cp, obs = _zoo()
    cp = dataclasses.replace(cp, durations={NODE: jnp.full(5, 3.0)})
    aview, avp = subsystem(comp, cp, "agent")
    log: list = []
    agent = hold(agent_step(aview, selection="deterministic"))
    p = particle(comp, cp, {"agent": agent, "world": _recording(log)})
    carry, levels = ParticleCarry(), []
    for _ in range(4):
        action, carry, _ = p(KEY, obs, carry, cp)
        levels.append(int(action["agent.a_location"]))
    assert levels[0] == levels[1] == levels[2]
    for given, level in zip(log, levels, strict=True):
        assert _same(given, _level_view(comp, cp, level))


SURVEY = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _survey_modules():
    import sys

    sys.path.insert(0, str(SURVEY))
    try:
        import survey_model as sm
        import survey_scheduler as ss
    finally:
        sys.path.remove(str(SURVEY))
    return sm, ss


def test_a_slow_mind_under_hold_and_a_fast_learner_sharing_a_parameter():
    """N4 without the camera: decisions at rounds 0, 3, 6; the fast mind's write-back visible in the
    slow view at each decision; the routed level standing on held rounds; a float control node
    owned by a particle mind is emitted and never recorded as a level."""
    sm, ss = _survey_modules()
    rs, rp, grid, centres = sm.rover(
        None, grid_shape=4, camera=False, image_preference=False, tiered=True
    )
    ms, mp = ss.scheduler(grid, centres, durations=True, policy_len=rs.policy_len)
    comp, cp = merge(
        {"mission": ms, "rover": rs},
        {"mission": mp, "rover": rp},
        [
            Link("mission.option", "rover.u_opt"),
            Identify("rover.field", "mission.field"),
            Link("rover.x", "mission.cell_obs", realise="mode", kernel=grid),
        ],
    )
    mview = subsystem(comp, cp, "mission")[0]
    mview = dataclasses.replace(mview, policy_len=1)
    rview = subsystem(comp, cp, "rover", inputs={"mission.option": 0})[0]
    slow_seen, fast_seen = [], []
    slow_inner = agent_step(mview, selection="deterministic")

    def slow(key, obs, carry, params):
        slow_seen.append(params)
        return slow_inner(key, obs, carry, params)

    def fast(key, obs, carry, params):
        fast_seen.append(params)
        action = {"rover.a_v": jnp.zeros(2, jnp.float32)}
        emitted = {"rover.cell_obs": jnp.asarray(0, jnp.int32)}
        return action, carry._replace(emitted=emitted), _bump(rview, params)

    minds = {"mission": hold(slow, 3 / ss.SURVEY_TICKS, system=mview), "rover": fast}
    p = particle(comp, cp, minds)
    obs = {"mission.gauge": jnp.asarray(3, jnp.int32)}
    carry, options, home = ParticleCarry(), [], cp
    for _ in range(7):
        action, carry, home = p(KEY, obs, carry, home)
        options.append(int(action["mission.option"]))
        assert action["rover.a_v"].dtype == jnp.float32
    assert len(slow_seen) == 3  # decisions at rounds 0, 3, 6
    base = _field(cp)
    for decision, t in enumerate((0, 3, 6)):
        assert np.array_equal(
            _field(slow_seen[decision]), base + t
        )  # the fast mind's counts so far
    for t, given in enumerate(fast_seen):
        want = subsystem(comp, cp, "rover", inputs={"mission.option": options[t]})[1]
        assert _same(given.state_targets, want.state_targets)  # the level in force, held or fresh
        assert np.array_equal(_field(given), base + t)
    assert options[0] == options[1] == options[2] and options[3] == options[4] == options[5]
    assert set(carry.standing) == {"mission.u_option"}
    assert np.array_equal(_field(home), base + 7)
