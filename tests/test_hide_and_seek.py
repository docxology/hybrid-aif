"""``examples/hide_and_seek.py``, the composition vignette's model, runs and its wiring is honest.

Loaded from ``examples/`` the way the other demo suites load theirs (``importlib`` off the
file path) and driven on the smallest grid, so the seams it reads through (``couple``, ``run``'s
tick orders, the mirror of a two-input System) cannot drift without this noticing.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import jax
import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "examples" / "hide_and_seek.py"
_spec = importlib.util.spec_from_file_location("hide_and_seek", _PATH)
hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hs)


def test_the_smallest_game_runs(capsys):
    assert hs.main(["3"]) == 0
    assert "3 ticks: caught" in capsys.readouterr().out


def test_the_world_emits_the_distance_of_the_positions_it_emits():
    #  the world's control nodes are driven by both minds' moves, so it steps after them in every
    #  round: each mind's entry t is what the world emitted entering round t
    record = hs.play(
        hs.mind("prey", n=3),
        hs.mind("predator", n=3),
        5,
        jax.random.PRNGKey(2),
        tick_order="simultaneous",
        n=3,
    )
    world = record["world"]
    prey, pred, distance = (
        np.concatenate([np.asarray(world.start[k])[None], world.emitted[k]])
        for k in ("prey_pos", "pred_pos", "distance")
    )
    want = [min(hs.manhattan(int(a), int(b), 3), 3) for a, b in zip(prey, pred, strict=True)]
    assert distance.tolist() == want
    #  both minds received the world's emissions, each under its own slot names
    assert np.array_equal(record["prey"].received["me_obs"], prey)
    assert np.array_equal(record["predator"].received["other_obs"], prey)
    assert np.array_equal(record["prey"].received["distance"], distance)
    assert bool(np.all(record["prey"].present["distance"]))
    #  one move per tick, as the minds' tables say
    assert all(
        hs.manhattan(int(a), int(b), 3) <= 1 for a, b in zip(prey[:-1], prey[1:], strict=True)
    )


def test_a_minds_objective_is_its_own_declaration():
    system, params = hs.mind("prey", n=3)
    mop = dataclasses.replace(system, epistemic=False, terms=("occupancy:transition:me",))
    params = dataclasses.replace(params, obs_targets={})
    predator = hs.mind("predator", n=3)
    record = hs.play((mop, params), predator, 3, jax.random.PRNGKey(0), n=3)
    assert record["prey"].intended["move"].shape == (3,)
    #  the swapped mind's tree is the one leaf it declared; the other mind's tree is untouched
    flat = {f: hs.Categorical.from_probs(np.full(9, 1 / 9)) for f in ("me", "other")}
    prey_obs = {m: v[0] for m, v in record["prey"].received.items()}
    beliefs, _ = mop.infer_states(params, prey_obs, flat)
    _, tree = mop.infer_policies(params, beliefs)
    assert set(tree) == {"occupancy"}
    pred_obs = {m: v[0] for m, v in record["predator"].received.items()}
    beliefs, _ = predator[0].infer_states(predator[1], pred_obs, flat)
    _, tree = predator[0].infer_policies(predator[1], beliefs)
    assert set(tree) == {"utility", "infogain"}
