"""The survey runner (``examples/survey_run.py``) over a closed System, pinned to the wired-form
runner.

``tests/data/survey_run.npz`` holds three short runs of the survey runner as it ran on the wired
form of ``run``: the agent one mind (a ``particle`` of the merged scheduler and rover, the
scheduler wrapped in ``hold``), the world a hand-written process. The cases are the two-tier run
(``two-level``), the run with the option pinned (``one-level``) and the rover alone (``epi``, the
first level-1 setting), each with a small untrained camera (``small_camera``), a 4 by 4 field grid
and a short plan, the two-tier runs with an option two rounds long. Every number of every tick
record, the data the launch page reads (``seconds`` aside), is compared bit for bit with the
runner's own under the wired form's key layout (``_survey_layout``). The fixture's generator ran
the runner of a ``git archive`` in its own interpreter and took the cases from this module
(``CASES``, ``case_setting``, ``small_camera``, ``flatten``).
"""

import contextlib
import dataclasses
import pathlib
import sys
from unittest import mock

import jax
import numpy as np
import pytest

from hybrid_aif.composition import runner
from hybrid_aif.distributions import MLPConditional

FIXTURE = pathlib.Path(__file__).parent / "data" / "survey_run.npz"
EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
CASES = ("two-level", "one-level", "epi", "two-level-sampled")
TICKS = {"two-level": 5, "one-level": 3, "epi": 3, "two-level-sampled": 5}
GRID = 4  # field cells per axis
SLOW_TICKS = 2  # rounds an option lasts in the two-tier runs: decisions at rounds 0, 2 and 4
STEPS = 3  # descent steps of the rover's plan
STARTS = 2  # its warm starts


def survey_run():
    """The example's runner module (``examples/survey_run.py``)."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        import survey_run as module
    finally:
        sys.path.remove(str(EXAMPLES))
    return module


def case_setting(module, case):
    """``(setting, ticks)`` of a case, from the runner ``module``'s settings grids."""
    grids = (*module.settings_grid_l2(), *module.settings_grid())
    sid = case.removesuffix("-sampled")
    setting = dict(next(s for s in grids if s["id"] == sid))
    setting["grid"] = GRID
    if "tiers" in setting:
        setting.update(slow_ticks=SLOW_TICKS, readings=module.READINGS_AVAILABLE)
    return setting, TICKS[case]


def case_patch(module, case):
    """The context a case runs in: for ``two-level-sampled``, the runner ``module``'s ``Link``
    made to realise by ``sample`` where it was asked for ``mode`` (the cell read), else nothing."""
    if not case.endswith("-sampled"):
        return contextlib.nullcontext()
    link = module.Link

    def sampled(*args, **kwargs):
        if kwargs.get("realise") == "mode":
            kwargs["realise"] = "sample"
        return link(*args, **kwargs)

    return mock.patch.object(module, "Link", sampled)


def small_camera(pixels=48 * 48, pixel_sd=0.04):
    """A camera likelihood of the survey's shape (position and velocity, and a one-hot terrain, to
    the image's pixels), small and untrained: an ``MLPConditional`` of width 8 and depth 1 from a
    fixed key, frozen."""
    net = MLPConditional.from_shapes(
        (4, 3), pixels, key=jax.random.PRNGKey(0), init_scale=pixel_sd, width=8, depth=1
    )
    return dataclasses.replace(net, frozen=True)


def flatten(obj, path):
    """Every number of ``obj`` (nested dicts and lists of tick records) as arrays keyed by path,
    ``seconds`` left out; a list of numbers (or of lists of numbers) is one array, and ``None``
    an empty one."""
    if isinstance(obj, dict):
        out = {}
        for k in sorted(obj):
            if k != "seconds":
                out.update(flatten(obj[k], f"{path}/{k}"))
        return out
    if obj is None:
        return {path: np.zeros(0)}
    if isinstance(obj, list) and not _numeric(obj):
        out = {}
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{path}/{i}"))
        return out
    return {path: np.asarray(obj)}


def _numeric(obj):
    if isinstance(obj, list):
        return all(_numeric(v) for v in obj)
    return isinstance(obj, bool | int | float)


def _survey_layout():
    """The wired form's keys for its one mind (the agent) and one process (the world), handed to
    the stepped nodes of the closed System, the agent's leaves first and the world last.

    The wired form split a round key in four (``2M + 2`` for ``M = 1``): the mind stepped with
    key 0, the realisations (no kernel draws here) and the world's step took key 3, and inside the
    particle the scheduler stepped with the mind's key and the rover with ``fold_in(key, 1)``; the
    world reset with the reset key itself. ``run`` steps node ``j`` with key ``j`` of its own
    ``2N + 1``, realises with key ``N + j`` and resets with ``fold_in(reset key, j)`` after the
    first: the patches hand each node its counterpart's key. The scheduler draws from no key
    (deterministic selection) and the rover's planner is deterministic; the rover's one emission
    draws from its key only where it is realised by ``sample`` (``two-level-sampled``)."""

    def round_keys(key, count):
        wired = list(jax.random.split(key, 4))
        agent = [wired[0] if j == 0 else jax.random.fold_in(wired[0], j) for j in range(count - 1)]
        return [*agent, wired[3], *([wired[3]] * count), wired[3]]

    def reset_keys(key, count):
        return [jax.random.fold_in(key, j) for j in range(count - 1)] + [key]

    original = runner._sweep_generator
    return (
        mock.patch.object(runner, "_round_keys", round_keys),
        mock.patch.object(runner, "_reset_keys", reset_keys),
        mock.patch.object(runner, "_sweep_generator", lambda key, count: original(key, 1)),
    )


@pytest.mark.parametrize("case", CASES)
def test_the_survey_runs_on_a_closed_system_are_the_wired_runs_bitwise(case):
    module = survey_run()
    setting, ticks = case_setting(module, case)
    a, b, c = _survey_layout()
    with a, b, c, case_patch(module, case):
        result = module.run_setting(setting, ticks, small_camera(), STEPS, STARTS)
    rows = result["ticks"]
    got = flatten(rows, case)
    fixture = np.load(FIXTURE)
    want = {k: fixture[k] for k in fixture.files if k.split("/", 1)[0] == case}
    assert sorted(got) == sorted(want)
    for k, v in want.items():
        assert got[k].dtype == v.dtype, k
        np.testing.assert_array_equal(got[k], v, err_msg=k)
    if case.startswith("two-level"):
        decided = [t for t, r in enumerate(rows) if r["slow"]["decision"]]
        assert decided == [0, 2, 4]
        assert rows[0]["slow"]["cell"] is None  # the rover has emitted no cell before round 0
        assert rows[2]["slow"]["cell"] is not None
