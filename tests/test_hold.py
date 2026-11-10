"""A mind that decides once per option: a node whose view declares durations under ``run``
(the robot coupled to the corridor, and the robot in a merge against its own mirror), and the
checks of ``composition.hold``, the wrapper of the composite and wired forms.

``tests/test_clock_parity.py`` compares both runs here bit for bit against what ``hold`` under
the wired and composite forms recorded (cases ``wired/*/cadence`` and ``composite/*``, the same
durations, key and length); ``tests/test_clocks.py`` holds ``run``'s refusals that match the
``hold`` checks below.
"""

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition import (
    HoldCarry,
    MindCarry,
    agent_step,
    couple,
    hold,
    merge,
    plan,
    run,
    sample,
    with_mirror,
)

_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "compose_minds.py"
_SPEC = importlib.util.spec_from_file_location("compose_minds", _PATH)
cm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cm)

TICKS = 20
KEY = jax.random.PRNGKey(1)


def _slow(durations):
    system, params = cm.robot(2, 2)
    return system, dataclasses.replace(params, durations={"u_move": jnp.asarray(durations)})


def _counted(step):
    calls = []

    def wrapped(key, obs, carry, params):
        calls.append(1)
        return step(key, obs, carry, params)

    wrapped.system = step.system
    return wrapped, calls


def _counted_plan(**kwargs):
    """``plan(**kwargs)`` counting its calls after the reset."""
    inner, calls = plan(**kwargs), []

    def step(system, key, obs, carry, params):
        if obs is not None:
            calls.append(1)
        return inner(system, key, obs, carry, params)

    step.sliced = True
    return step, calls


def _boundaries(actions, rounds_of):
    """The decision rounds implied by the held actions: each option lasts its level's rounds."""
    out, t = [], 0
    while t < len(actions):
        out.append(t)
        t += rounds_of[int(actions[t])]
    return out


def _check(actions, calls, rounds_of):
    actions = np.asarray(actions)
    bounds = _boundaries(actions, rounds_of)
    for t in bounds:
        assert (actions[t : t + rounds_of[int(actions[t])]] == actions[t]).all()
    assert len(calls) == len(bounds)
    assert len(bounds) < len(actions)


def _coupled(slow):
    """Robot ``a`` (``slow``) and robot ``b`` coupled to the corridor, ``a`` counted."""
    (sa, pa), (sb, pb) = slow, cm.robot(2, 2)
    system, params = couple(
        {"a": (sa, pa), "b": (sb, pb), "world": cm.world((2, 2))}, list(cm.BINDINGS)
    )
    step_a, calls = _counted_plan(selection="stochastic")
    steps = {"a": step_a, "b": plan(selection="stochastic"), "world": sample()}
    return system, params, steps, calls


@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_coupled_robot_with_durations_redecides_only_at_option_boundaries(order):
    system, params, steps, calls = _coupled(_slow([4.0, 2.0, 4.0]))
    rec = run(system, params, TICKS, KEY, steps, tick_order=order)
    _check(rec["a"].emitted["move"], calls, {0: 4, 1: 2, 2: 4})
    assert int(rec["a"].decided.sum()) == len(calls)


@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_merged_robot_with_durations_redecides_only_at_option_boundaries(order):
    """The cadence of the robot with durations in a merge of the two robots, run against the
    merge's own mirror: the robot's view declares the durations, the world leaf carries none and
    steps every round, so the positions move once per round under the option standing."""
    (sa, pa), (sb, pb) = _slow([4.0, 2.0, 4.0]), cm.robot(2, 2)
    comp, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    system, params = with_mirror(comp, cp)
    step_a, calls = _counted_plan(selection="stochastic")
    steps = {"model.a": step_a, "model.b": plan(selection="stochastic"), "world": sample()}
    tr = run(system, params, TICKS, KEY, steps, tick_order=order)
    _check(tr["model.a"].emitted["move"], calls, {0: 4, 1: 2, 2: 4})
    assert bool(tr["world"].decided.all()) and bool(tr["model.b"].decided.all())


def test_ticks_per_unit_converts_the_unit_to_rounds():
    system, params = _slow([1.5, 1.5, 1.5])
    inner, calls = _counted(agent_step(system, selection="stochastic"))
    held = hold(inner, ticks_per_unit=2)
    # drive the wrapper directly: it decides on rounds 0, 3, 6 and holds in between
    carry = MindCarry()
    obs = {m: jnp.zeros((), jnp.int32) for m in system.names.modalities}
    seen = []
    for t in range(9):
        action, carry, given = held(jax.random.PRNGKey(t), obs, carry, params)
        assert given is params
        seen.append((len(calls), int(action["move"])))
        assert isinstance(carry, HoldCarry)
    assert [n for n, _ in seen] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    for start in (0, 3, 6):
        assert len({a for _, a in seen[start : start + 3]}) == 1


def test_a_carries_seed_decides_on_the_first_round():
    system, params, steps, calls = _coupled(_slow([4.0, 2.0, 4.0]))
    rec = run(system, params, 1, KEY, steps, carries={"a": MindCarry()})
    assert len(calls) == 1
    assert rec["a"].emitted["move"].shape == (1,)


def test_a_duration_that_is_not_a_whole_number_of_rounds_is_refused():
    system, params = _slow([1.5, 1.5, 1.5])
    held = hold(agent_step(system))
    obs = {m: jnp.zeros((), jnp.int32) for m in system.names.modalities}
    with pytest.raises(ValueError, match="not a whole number of rounds"):
        held(KEY, obs, MindCarry(), params)


def test_a_stochastic_actuator_on_the_durations_node_is_refused():
    system, params = _slow([2.0, 2.0, 2.0])
    kernel = jnp.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    params = dataclasses.replace(params, actuator={"move": kernel})
    held = hold(agent_step(system))
    with pytest.raises(ValueError, match="non-identity actuator.*T9"):
        held(KEY, {}, MindCarry(), params)


def test_a_permutation_actuator_is_refused_though_it_is_deterministic():
    system, params = _slow([2.0, 2.0, 2.0])
    params = dataclasses.replace(params, actuator={"move": jnp.asarray(np.eye(3)[[1, 2, 0]])})
    held = hold(agent_step(system))
    with pytest.raises(ValueError, match="non-identity actuator"):
        held(KEY, {}, MindCarry(), params)


def test_a_system_with_no_durations_is_refused():
    system, params = cm.robot(2, 2)
    held = hold(agent_step(system))
    with pytest.raises(ValueError, match="declares no params.durations"):
        held(KEY, {}, MindCarry(), params)


def test_elapsed_is_legal_under_jit_and_cond():
    system, params = _slow([4.0, 2.0, 4.0])

    @jax.jit
    def f(p, level):
        return system.elapsed({"move": level}, p)

    assert float(f(params, jnp.asarray(1))) == 2.0
    assert float(f(params, jnp.asarray(2))) == 4.0

    @jax.jit
    def g(level):
        return jax.lax.cond(
            level > 0, lambda: system.elapsed({"move": level}, params), lambda: jnp.asarray(0.0)
        )

    assert float(g(jnp.asarray(1))) == 2.0


def test_a_non_positive_duration_is_still_refused_when_concrete():
    system, params = _slow([4.0, 0.0, 4.0])
    with pytest.raises(ValueError, match="non-positive"):
        system.elapsed({"move": jnp.asarray(0)}, params)
