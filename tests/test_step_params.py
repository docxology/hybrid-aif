"""Params flow through the runner: a step is handed its params each round and returns them, and
the record of a wired run holds the last ones (the starting object if the step never learned)."""

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition import agent_step, ancestral_step, couple, run

_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "compose_minds.py"
_SPEC = importlib.util.spec_from_file_location("compose_minds_step_params", _PATH)
cm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cm)

KEY = jax.random.PRNGKey(3)
TICKS = 5


def _counting(inner):
    """``inner`` that also counts rounds in ``params.precision``: a learner."""

    def step(key, obs, carry, params):
        action, carry, _ = inner(key, obs, carry, params)
        return action, carry, dataclasses.replace(params, precision=params.precision + 1.0)

    return step


def _wiring(first_step, first_params, second_step, second_params, system_a, system_b):
    return couple(
        minds={
            "a": (system_a, first_params, first_step),
            "b": (system_b, second_params, second_step),
        },
        processes={"world": cm.world((2, 2))},
        links=list(cm.PORT_LINKS),
    )


def _robots():
    (sa, pa), (sb, pb) = cm.robot(2, 2), cm.robot(2, 2)
    return sa, dataclasses.replace(pa, precision=jnp.asarray(1.0)), sb, pb


def test_a_learner_under_the_wired_form_returns_its_learned_params_in_the_record():
    sa, pa, sb, pb = _robots()
    wiring = _wiring(_counting(agent_step(sa)), pa, agent_step(sb), pb, sa, sb)
    rec = run(wiring, ticks=TICKS, key=KEY)
    np.testing.assert_array_equal(rec["a"].params.precision, 1.0 + TICKS)
    assert rec["a"].params is not pa
    assert pa.precision == 1.0  # the starting params are not mutated


def test_a_learner_is_handed_what_it_returned_the_round_before():
    sa, pa, sb, pb = _robots()
    seen = []
    inner = _counting(agent_step(sa))

    def watcher(key, obs, carry, params):
        seen.append(float(params.precision))
        return inner(key, obs, carry, params)

    run(_wiring(watcher, pa, agent_step(sb), pb, sa, sb), ticks=TICKS, key=KEY)
    assert seen == [1.0 + t for t in range(TICKS)]


def test_a_non_learner_s_record_params_is_the_starting_object():
    sa, pa, sb, pb = _robots()
    wiring = _wiring(agent_step(sa), pa, agent_step(sb), pb, sa, sb)
    rec = run(wiring, ticks=TICKS, key=KEY)
    assert rec["a"].params is pa and rec["b"].params is pb


def test_a_run_of_no_ticks_records_the_starting_params():
    sa, pa, sb, pb = _robots()
    rec = run(_wiring(_counting(agent_step(sa)), pa, agent_step(sb), pb, sa, sb), ticks=0, key=KEY)
    assert rec["a"].params is pa


def test_ancestral_step_under_the_wired_form_returns_the_params_it_was_given():
    sa, pa, sb, pb = _robots()
    rec = run(_wiring(agent_step(sa), pa, ancestral_step(sb), pb, sa, sb), ticks=TICKS, key=KEY)
    assert rec["b"].params is pb and rec["a"].params is pa


def test_a_step_that_returns_something_else_as_params_is_refused_naming_the_mind():
    sa, pa, sb, pb = _robots()
    inner = agent_step(sa)

    def broken(key, obs, carry, params):
        action, carry, _ = inner(key, obs, carry, params)
        return action, carry, None

    wiring = _wiring(broken, pa, agent_step(sb), pb, sa, sb)
    with pytest.raises(ValueError, match="mind 'a'.*NoneType"):
        run(wiring, ticks=2, key=KEY)
