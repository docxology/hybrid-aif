"""``particle``: a merged composite with one mind per view, as one mind for the wired form
(design: nesting-and-coupling.md N4). The parameter home is written back each round a view learns
and the views are re-taken from it; a driving component's decision is routed into the driven view;
``hold`` is the clock inside it."""

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
    ActionControl,
    HoldCarry,
    MindCarry,
    ParticleCarry,
    SharedParameter,
    SiblingRead,
    ViewRespond,
    Wire,
    agent_respond,
    couple,
    hold,
    merge,
    particle,
    particle_params,
    run,
    subsystem,
    view_respond,
    write_back,
)
from hybrid_aif.composition.merge import partner_controls, retake
from tests.test_merge import OBSERVATION_BINDINGS, merged
from tests.test_sibling_read import _composite, _rover
from tests.test_subsystem_conditional_target import _conditional_target, _two_tier
from tests.test_subsystem_conditional_target import _rover as _target_rover

KEY = jax.random.PRNGKey(0)
NODE = "agent.u_location"
SAME_STEP = ActionControl(driver=NODE, control="world.u_location")

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
    def respond(key, obs, carry, params):
        log.append(params)
        return {}, carry, params

    return ViewRespond(respond)


def test_the_view_that_reads_a_partner_s_node_names_it():
    comp, _, _ = _zoo()
    assert partner_controls(comp, "world") == (NODE,)
    assert partner_controls(comp, "agent") == ()


def _level_view(comp, cp, level):
    return subsystem(
        comp, cp, "world", partner_priors={NODE: level} if level is not None else None
    )[1]


def test_a_driver_listed_first_is_seen_by_the_driven_view_the_same_round():
    comp, cp, obs = _zoo()
    log: list = []
    agent = view_respond(subsystem(comp, cp, "agent")[0], selection="deterministic")
    p = particle(comp, cp, {"agent": agent, "world": _recording(log)})
    action, carry = p(KEY, obs, ParticleCarry())
    level = int(action["agent.a_location"])
    assert carry.standing == {NODE: level}
    assert _same(log[0], _level_view(comp, cp, level))


def test_a_later_round_sees_the_last_level_when_the_driver_is_listed_after():
    comp, cp, obs = _zoo()
    log: list = []
    agent = view_respond(subsystem(comp, cp, "agent")[0], selection="deterministic")
    p = particle(comp, cp, {"world": _recording(log), "agent": agent})
    action, carry = p(KEY, obs, ParticleCarry())
    assert _same(log[0], _level_view(comp, cp, None))  # nothing chosen yet: uniform
    p(KEY, obs, carry)
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
    systems = [subsystem(comp, cp, "world", partner_priors={NODE: lv})[0] for lv in (0, 3)]
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
    comp, cp = _composite(shared=True, bindings=[SharedParameter("rover.field", "mission.field")])
    rover, mission = subsystem(comp, cp, "rover")[0], subsystem(comp, cp, "mission")[0]
    return comp, cp, rover, mission


def _learner(view, log=None):
    inner = view_respond(view)

    def respond(key, obs, carry, params):
        action, carry, _ = inner(key, obs, carry, params)
        if log is not None:
            log.append(params)
        return action, carry, _bump(view, params)

    return ViewRespond(respond)


def _field(params):
    return np.asarray(params.parameters["rover.field"].concentration)


def _watcher(view, seen):
    inner = view_respond(view)

    def respond(key, obs, carry, params):
        seen.append(params)
        return inner(key, obs, carry, params)

    return ViewRespond(respond)


def test_a_learning_view_is_written_back_each_round_and_the_other_view_is_retaken():
    comp, cp, rover, mission = _shared()
    obs = {"rover.obs": jnp.asarray([1.0, 1.0])}
    for order, offset in ((("rover", "mission"), 1), (("mission", "rover"), 0)):
        seen: list = []
        minds = {"rover": _learner(rover), "mission": _watcher(mission, seen)}
        p = particle(comp, cp, {k: minds[k] for k in order})
        carry = ParticleCarry()
        for t in range(3):
            _, carry = p(KEY, obs, carry)
            assert np.array_equal(_field(particle_params(carry)), _field(cp) + t + 1)
            # the reader sees this round's learning if the learner is listed first, else last's
            assert np.array_equal(_field(seen[t]), _field(cp) + t + offset)


def test_the_home_after_each_round_is_the_hand_loop_of_write_backs_bitwise():
    comp, cp, rover, mission = _shared()
    obs = {"rover.obs": jnp.asarray([1.0, 1.0])}
    p = particle(comp, cp, {"rover": _learner(rover), "mission": view_respond(mission)})
    carry, home = ParticleCarry(), cp
    for _ in range(4):
        _, carry = p(KEY, obs, carry)
        taken = subsystem(comp, home, "rover")[1]
        home = write_back(comp, home, "rover", taken, _bump(rover, taken))
        assert _same(particle_params(carry), home)


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
    p = particle(comp, cp, {"rover": _learner(rover), "mission": view_respond(mission)})
    stale = subsystem(comp, cp, "rover")[1]
    _, carry = p(KEY, {"rover.obs": jnp.asarray([1.0, 1.0])}, ParticleCarry())
    with pytest.raises(ValueError, match="taken before"):
        write_back(comp, particle_params(carry), "rover", stale, _bump(rover, stale))


def test_a_plain_respond_mind_is_handed_no_params_and_a_learner_beside_it_still_learns():
    comp, cp, rover, mission = _shared()
    seen = []

    def plain(key, obs, carry):
        seen.append(obs)
        return view_respond(mission)(key, obs, carry, subsystem(comp, cp, "mission")[1])[:2]

    p = particle(comp, cp, {"rover": _learner(rover), "mission": plain})
    _, carry = p(KEY, {"rover.obs": jnp.asarray([1.0, 1.0])}, ParticleCarry())
    assert "mission.cell_obs" in seen[0] and seen[0]["mission.cell_obs"] is None
    assert np.array_equal(_field(particle_params(carry)), _field(cp) + 1)


# ------------------------------------------------------------------- the wired form


def _one_component():
    sa, pa = cm.robot(2, 2)
    comp, cp = merge({"a": sa}, {"a": pa}, [])
    wires = [
        Wire("m.a.move", "world.a_move"),
        Wire("world.a_pos", "m.a.me_obs"),
        Wire("world.b_pos", "m.a.other_obs"),
        Wire("world.status", "m.a.status"),
    ]
    return comp, cp, wires


def test_a_one_mind_particle_coupled_to_a_world_equals_the_plain_wired_run_bitwise():
    comp, cp, wires = _one_component()
    view, vp = subsystem(comp, cp, "a")
    kwargs = dict(processes={"world": cm.world((2, 2))}, wires=wires, fixed={"world.b_move": 0})
    plain = couple(
        minds={"m": (view, vp, agent_respond(view, vp, selection="stochastic"))}, **kwargs
    )
    inner = view_respond(view, selection="stochastic")
    pooled = couple(minds={"m": (comp, cp, particle(comp, cp, {"a": inner}))}, **kwargs)
    one, two = run(plain, ticks=12, key=KEY), run(pooled, ticks=12, key=KEY)
    assert _same(one["m"].actions, two["m"].actions)
    assert _same(one["m"].observations, two["m"].observations)
    assert _same(one["world"].states, two["world"].states)
    assert _same(one["m"].carry.beliefs, two["m"].carry.minds["a"].beliefs)
    assert particle_params(two["m"].carry) is cp


def test_a_modality_the_particle_supplies_needs_no_wire_and_any_other_mind_does():
    comp, cp = _composite()
    rs, rp = _rover()
    world = rs.as_env(rp)
    wires = [Wire("w.obs", "m.rover.obs")]
    rover, mission = subsystem(comp, cp, "rover"), subsystem(comp, cp, "mission")
    minds = {"rover": view_respond(rover[0]), "mission": view_respond(mission[0])}
    wiring = couple(
        minds={"m": (comp, cp, particle(comp, cp, minds))}, processes={"w": world}, wires=wires
    )
    rec = run(wiring, ticks=3, key=KEY)
    assert set(rec["m"].observations) == {"rover.obs"}
    with pytest.raises(ValueError, match="fed by no wire"):
        couple(
            minds={"m": (comp, cp, agent_respond(comp, cp))}, processes={"w": world}, wires=wires
        )


def test_hold_inside_a_particle_under_the_wired_run():
    (sa, pa), (sb, pb) = cm.robot(2, 2), cm.robot(2, 2)
    pa = dataclasses.replace(pa, durations={"u_move": jnp.asarray([4.0, 2.0, 4.0])})
    comp, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    va, vb = subsystem(comp, cp, "a"), subsystem(comp, cp, "b")
    calls: list = []
    inner = view_respond(va[0], selection="stochastic")

    def counted(key, obs, carry, params):
        calls.append(1)
        return inner(key, obs, carry, params)

    minds = {
        "a": hold(*va, ViewRespond(counted)),
        "b": view_respond(vb[0], selection="stochastic"),
    }
    wires = [
        Wire(f"m.{s}", t) for s, t in ((("a.move", "world.a_move")), ("b.move", "world.b_move"))
    ]
    wires += [
        Wire("world.a_pos", "m.a.me_obs"),
        Wire("world.b_pos", "m.a.other_obs"),
        Wire("world.status", "m.a.status"),
        Wire("world.b_pos", "m.b.me_obs"),
        Wire("world.a_pos", "m.b.other_obs"),
        Wire("world.status", "m.b.status"),
    ]
    wiring = couple(
        minds={"m": (comp, cp, particle(comp, cp, minds))},
        processes={"world": cm.world((2, 2))},
        wires=wires,
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


def test_a_held_view_respond_returns_the_params_it_was_given_and_calls_nothing():
    sa, pa = cm.robot(2, 2)
    pa = dataclasses.replace(pa, durations={"u_move": jnp.asarray([3.0, 3.0, 3.0])})
    calls: list = []

    def fn(key, obs, carry, params):
        calls.append(1)
        return {"move": jnp.asarray(1, jnp.int32)}, carry, dataclasses.replace(params)

    held = hold(sa, pa, ViewRespond(fn))
    assert isinstance(held, ViewRespond)
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

    p = particle(comp, cp, {"rover": ViewRespond(learner), "mission": view_respond(mission)})
    with pytest.raises(ValueError, match=r"'rover'.*likelihood.*declared as a Parameter"):
        p(KEY, {"rover.obs": jnp.asarray([1.0, 1.0])}, ParticleCarry())


def test_a_wire_into_a_modality_the_particle_supplies_is_refused():
    comp, cp = _composite()
    rs, rp = _rover()
    rover, mission = subsystem(comp, cp, "rover"), subsystem(comp, cp, "mission")
    minds = {"rover": view_respond(rover[0]), "mission": view_respond(mission[0])}
    wires = [Wire("w.obs", "m.rover.obs"), Wire("src.mission.cell_obs", "m.mission.cell_obs")]
    with pytest.raises(ValueError, match="mission.cell_obs.*supplied inside the particle"):
        couple(
            minds={"m": (comp, cp, particle(comp, cp, minds))},
            processes={"w": rs.as_env(rp), "src": mission[0].as_env(mission[1])},
            wires=wires,
        )


def _two_tier_target():
    return _two_tier(*_target_rover(_conditional_target(), (("x", ("u_opt",)),)))


def _plain(key, obs, carry):
    return {"mission.a_option": jnp.asarray(1, jnp.int32)}, carry


def test_a_driver_listed_after_a_view_with_a_conditional_target_is_refused_at_construction():
    cs, cp = _two_tier_target()
    rover = ViewRespond(lambda key, obs, carry, params: ({}, carry, params))
    with pytest.raises(ValueError, match="list 'mission' before 'rover'"):
        particle(cs, cp, {"rover": rover, "mission": _plain})
    particle(cs, cp, {"mission": _plain, "rover": rover})


def test_view_respond_refuses_an_all_masked_legal_on_its_first_call():
    comp, cp, obs = _zoo()
    view, vp = subsystem(comp, cp, "agent")
    legal = {NODE: jnp.zeros(5, bool)}
    with pytest.raises(ValueError):
        agent_respond(view, vp, legal=legal)
    respond = view_respond(view, legal=legal)
    mods = {m: obs[m] for m in view.modality_names if m in obs}
    with pytest.raises(ValueError, match="masks every level"):
        respond(KEY, mods, MindCarry(), vp)


def test_a_view_respond_is_refused_by_run_and_couple_including_when_held():
    comp, cp, obs = _zoo()
    view, vp = subsystem(comp, cp, "agent")
    held = hold(
        view, dataclasses.replace(vp, durations={NODE: jnp.full(5, 2.0)}), view_respond(view)
    )
    for mind in (view_respond(view), held):
        assert isinstance(mind, ViewRespond)
        with pytest.raises(ValueError, match="mind for particle"):
            run(comp, cp, {"agent": mind}, ticks=1, key=KEY)
        with pytest.raises(ValueError, match="mind for particle"):
            couple(minds={"m": (view, vp, mind)})


def test_a_component_with_no_mind_is_refused():
    comp, cp = _composite()
    rover = subsystem(comp, cp, "rover")
    with pytest.raises(ValueError, match=r"\['mission'\].*no mind"):
        particle(comp, cp, {"rover": view_respond(rover[0])})


# ------------------------------------------------------------- held driver, survey shape


def test_a_held_drivers_level_reaches_the_driven_view_on_held_rounds():
    comp, cp, obs = _zoo()
    cp = dataclasses.replace(cp, durations={NODE: jnp.full(5, 3.0)})
    aview, avp = subsystem(comp, cp, "agent")
    log: list = []
    agent = hold(aview, avp, view_respond(aview, selection="deterministic"))
    p = particle(comp, cp, {"agent": agent, "world": _recording(log)})
    carry, levels = ParticleCarry(), []
    for _ in range(4):
        action, carry = p(KEY, obs, carry)
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
            ActionControl("mission.u_option", "rover.u_opt"),
            SharedParameter("rover.field", "mission.field"),
            SiblingRead("mission.cell_obs", "rover.x", grid, "mode"),
        ],
    )
    mview, mparams = subsystem(comp, cp, "mission")
    mview = dataclasses.replace(mview, policy_len=1)
    rview = subsystem(comp, cp, "rover", partner_priors={"mission.u_option": 0})[0]
    slow_seen, fast_seen = [], []
    slow_inner = view_respond(mview, selection="deterministic")

    def slow(key, obs, carry, params):
        slow_seen.append(params)
        return slow_inner(key, obs, carry, params)

    def fast(key, obs, carry, params):
        fast_seen.append(params)
        priors = rview.initial_priors(params)
        action = {"rover.a_v": jnp.zeros(2, jnp.float32)}
        return action, carry._replace(posterior=priors), _bump(rview, params)

    minds = {"mission": hold(mview, mparams, ViewRespond(slow), 3 / ss.SURVEY_TICKS)}
    minds["rover"] = ViewRespond(fast)
    p = particle(comp, cp, minds)
    obs = {"mission.gauge": jnp.asarray(3, jnp.int32)}
    carry, options = ParticleCarry(), []
    for _ in range(7):
        action, carry = p(KEY, obs, carry)
        options.append(int(action["mission.option"]))
        assert action["rover.a_v"].dtype == jnp.float32
    assert len(slow_seen) == 3  # decisions at rounds 0, 3, 6
    base = _field(cp)
    for decision, t in enumerate((0, 3, 6)):
        assert np.array_equal(
            _field(slow_seen[decision]), base + t
        )  # the fast mind's counts so far
    for t, given in enumerate(fast_seen):
        want = subsystem(comp, cp, "rover", partner_priors={"mission.u_option": options[t]})[1]
        assert _same(given.state_targets, want.state_targets)  # the level in force, held or fresh
        assert np.array_equal(_field(given), base + t)
    assert options[0] == options[1] == options[2] and options[3] == options[4] == options[5]
    assert set(carry.standing) == {"mission.u_option"}
    assert np.array_equal(_field(particle_params(carry)), base + 7)
