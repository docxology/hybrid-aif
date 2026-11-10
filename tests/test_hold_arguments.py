"""``hold``'s argument checks, the duration arithmetic, and carry resumption; and the same
arithmetic under ``run``'s ``clocks=`` (``run``'s other clock checks: ``tests/test_clocks.py``).
"""

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition import HoldCarry, MindCarry, hold

_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "compose_minds.py"
_SPEC = importlib.util.spec_from_file_location("compose_minds_args", _PATH)
cm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cm)

KEY = jax.random.PRNGKey(0)


def _timed(durations):
    system, params = cm.robot(2, 2)
    return system, dataclasses.replace(params, durations={"u_move": jnp.asarray(durations)})


def _stub(calls):
    def step(key, obs, carry, params):
        calls.append(1)
        action = {"move": jnp.asarray(0, jnp.int32), "look": jnp.asarray(0, jnp.int32)}
        return action, carry, params

    return step


def test_a_hold_carry_seed_resumes_its_count():
    system, params = _timed([4.0, 4.0, 4.0])
    calls = []
    held = hold(_stub(calls), system=system)
    seed = HoldCarry(MindCarry(), 2, {"move": jnp.asarray(1)})
    action, carry, _ = held(KEY, {}, seed, params)
    assert calls == [] and int(action["move"]) == 1 and carry.remaining == 1


def test_a_float32_duration_times_ticks_per_unit_passes():
    system, params = _timed([33.3, 33.3, 33.3])
    held = hold(_stub([]), ticks_per_unit=10, system=system)
    _, carry, _ = held(KEY, {}, MindCarry(), params)
    assert carry.remaining == 332


@pytest.mark.parametrize(
    "good", [np.int64(2), np.float32(2.0), jnp.asarray(2), np.asarray(2.0), 2, 2.0]
)
def test_ticks_per_unit_is_any_real_scalar(good):
    system, params = _timed([1.0, 1.0, 1.0])
    held = hold(_stub([]), ticks_per_unit=good, system=system)
    _, carry, _ = held(KEY, {}, MindCarry(), params)
    assert carry.remaining == 1


@pytest.mark.parametrize("bad", [0, -1, "2", None, True, float("nan")])
def test_a_bad_ticks_per_unit_is_refused_at_construction(bad):
    system, params = _timed([2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="ticks_per_unit"):
        hold(_stub([]), ticks_per_unit=bad, system=system)


def test_a_duration_shorter_than_a_round_has_its_own_sentence():
    system, params = _timed([0.25, 0.25, 0.25])
    held = hold(_stub([]), system=system)
    with pytest.raises(ValueError, match="shorter than one round"):
        held(KEY, {}, MindCarry(), params)


def test_an_identity_kernel_is_accepted():
    system, params = _timed([2.0, 2.0, 2.0])
    params = dataclasses.replace(params, actuator={"move": jnp.eye(3)})
    _, carry, _ = hold(_stub([]), system=system)(KEY, {}, MindCarry(), params)
    assert carry.remaining == 1


def test_a_step_that_names_no_system_is_refused():
    with pytest.raises(TypeError, match="names no System"):
        hold(_stub([]))


@pytest.mark.parametrize("value", [jnp.asarray([2.0, 0.0, 1.0]), [2.0, -1.0, 1.0]])
def test_a_non_positive_duration_is_refused(value):
    system, params = cm.robot(2, 2)
    params = dataclasses.replace(params, durations={"u_move": value})
    with pytest.raises(ValueError, match="non-positive"):
        system._durations(system._params_in(params, "t"), "t")


def test_a_system_that_differs_from_the_steps_own_is_refused():
    system, _ = _timed([2.0, 2.0, 2.0])
    other, _ = _timed([2.0, 2.0, 2.0])
    step = _stub([])
    step.system = system
    hold(step, system=system)  # the same System: fine
    with pytest.raises(TypeError, match="different System"):
        hold(step, system=other)


# ------------------------------------------- the same arithmetic under run's clocks


def test_a_float32_duration_times_a_clock_holds_a_whole_number_of_rounds_under_run():
    """``33.3 * 10`` in float32 is a whole number of rounds within the tolerance ``hold`` uses."""
    from hybrid_aif.composition import plan, run
    from tests.test_clocks import robots, steps_of

    system, params = robots([33.3, 33.3, 33.3])
    traj = run(system, params, 3, KEY, steps_of(plan()), clocks={"a": 10})
    assert np.asarray(traj["a"].decided).tolist() == [True, False, False]


@pytest.mark.parametrize(
    "good", [np.int64(2), np.float32(2.0), jnp.asarray(2), np.asarray(2.0), 2, 2.0]
)
def test_a_clock_is_any_real_scalar_under_run(good):
    from hybrid_aif.composition import plan, run
    from tests.test_clocks import robots, steps_of

    system, params = robots([1.0, 1.0, 1.0])
    traj = run(system, params, 4, KEY, steps_of(plan()), clocks={"a": good})
    assert np.asarray(traj["a"].decided).tolist() == [True, False, True, False]
