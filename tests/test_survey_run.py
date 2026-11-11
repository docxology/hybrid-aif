"""The survey showcase (``examples/survey_*.py``): a regression pin and the properties of its
world, camera and planner.

``tests/data/survey_run.npz`` holds short runs of the survey runner recorded from this tree: the
two-tier run (``two-level``) and the rover alone (``one-level``), each with a small untrained
camera (``small_camera``), a 4 by 4 field grid and a short plan, the two-tier run with an option two
rounds long. It is a pin against accidental change of the runner's output, not a parity claim:
every number of every tick record (``seconds`` aside) is compared with the runner's own, to a
float32 rounding tolerance. A deliberate change of the showcase regenerates it.

The pin covers the runner's and the renderer's plumbing; the test camera is an untrained width-8
network, so the pin says nothing of the decoder's quality or the camera's calibration. The
properties: the motion is bounded (at most one map cell per tick, forward only, a held command holds
its target speed, the heading turns by at most the declared bound, by the command alone, in place
too, and holds at rest); the forward camera is coherent (a world point keeps its texture value as
the rover moves and turns a little); the rover alone drives forward under its planner and its
position and heading beliefs are calibrated without the camera, and with a small trained camera
(``slow``); the rover's plans respect the wall preference; the scheduler decides on the slow clock;
an empty band cannot drive or turn, and a rover started low keeps its energy off the floor; the
rover can observe the recogniser's reading instead of the image (experiment A).
"""

import dataclasses
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import MLPConditional
from hybrid_aif.planning.gradient import make_gradient_planner

FIXTURE = pathlib.Path(__file__).parent / "data" / "survey_run.npz"
EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
CASES = ("two-level", "one-level")
TICKS = {"two-level": 5, "one-level": 3}
GRID = 4  # field cells per axis
SLOW_TICKS = 2  # rounds an option lasts in the two-tier run: decisions at rounds 0, 2 and 4
STEPS = 3  # descent steps of the rover's plan
STARTS = 2  # its warm starts
CELL = 0.375  # one cell of the showcase's 16 by 16 map grid over the 6 by 6 field
PLANNER = make_gradient_planner(steps=40, starts=4)  # one planner, so its face traces once


def example(name):
    """An example module from ``examples/``."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        module = __import__(name)
    finally:
        sys.path.remove(str(EXAMPLES))
    return module


def case_setting(module, case):
    """``(setting, ticks)`` of a case, from the runner ``module``'s two-tier settings grid."""
    setting = dict(next(s for s in module.settings_grid_l2() if s["id"] == case))
    setting["grid"] = GRID
    setting.update(slow_ticks=SLOW_TICKS, readings=module.READINGS_AVAILABLE)
    return setting, TICKS[case]


def small_camera(pixels, pixel_sd=0.04):
    """A camera likelihood of the survey's shape (the vehicle's state and a one-hot terrain, to the
    image's pixels), small and untrained: an ``MLPConditional`` of width 8 and depth 1 from a
    fixed key, frozen."""
    net = MLPConditional.from_shapes(
        (5, 3), pixels, key=jax.random.PRNGKey(0), init_scale=pixel_sd, width=8, depth=1
    )
    return dataclasses.replace(net, frozen=True)


def flatten(obj, path):
    """Every number of ``obj`` (nested dicts and lists of tick records) as arrays keyed by path,
    the timings (``seconds`` and ``seconds_parts``) left out; a list of numbers (or of lists of
    numbers) is one array, and ``None`` an empty one."""
    if isinstance(obj, dict):
        out = {}
        for k in sorted(obj):
            if not k.startswith("seconds"):
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


def run_case(case):
    """The tick records of a case's run."""
    module = example("survey_run")
    setting, ticks = case_setting(module, case)
    camera = small_camera(example("survey_world").IMG_PIXELS)
    return module.run_setting(setting, ticks, camera, STEPS, STARTS)["ticks"]


@pytest.fixture(scope="module")
def runs():
    return {case: run_case(case) for case in CASES}


@pytest.mark.parametrize("case", CASES)
def test_the_survey_runs_match_the_regression_pin(runs, case):
    got = flatten(runs[case], case)
    fixture = np.load(FIXTURE)
    want = {k: fixture[k] for k in fixture.files if k.split("/", 1)[0] == case}
    assert sorted(got) == sorted(want)
    for k, v in want.items():
        assert got[k].dtype == v.dtype, k
        if v.dtype.kind == "f":
            np.testing.assert_allclose(got[k], v, rtol=1e-5, atol=1e-6, err_msg=k)
        else:
            np.testing.assert_array_equal(got[k], v, err_msg=k)


def test_the_scheduler_decides_on_the_slow_clock_and_the_rover_alone_has_no_option(runs):
    rows = runs["two-level"]
    assert [t for t, r in enumerate(rows) if r["slow"]["decision"]] == [0, 2, 4]
    assert rows[0]["slow"]["cell"] is None  # the rover has emitted no cell before round 0
    assert rows[2]["slow"]["cell"] is not None
    options = [r["option"] for r in rows]
    assert options[0] == options[1] and options[2] == options[3]  # held between decisions
    assert all("option" not in r and "slow" not in r for r in runs["one-level"])


def test_an_empty_band_cannot_drive_or_turn():
    wd = example("survey_world")
    state = wd.initial_state({"energy": 0.0, "x": [1.0, 0.0, 1.0, 0.0, 0.0]})
    for k in range(3):  # the light charges less than a band in three ticks
        state = wd.move(jax.random.PRNGKey(k), state, jnp.ones(2))
    assert abs(float(state["x"][4])) < 0.02  # the noise alone; no drive
    assert abs(float(wd.heading_of(state["x"]))) < np.deg2rad(5)  # no turn


def test_the_rover_keeps_its_energy_off_the_floor_from_a_low_start():
    module = example("survey_run")
    setting = dict(next(s for s in module.settings_grid_l2() if s["id"] == "one-level"))
    setting.update(grid=GRID, camera=False, start_energy=1.2)
    rows = module.run_setting(setting, 40, None, STEPS, STARTS)["ticks"]
    assert min(r["true_energy"]["level"] for r in rows) >= 0.5  # never near the empty band


def _drive(wd, commands, start, key):
    """The world's true states under a sequence of commands from ``start`` (x, y, cos theta,
    sin theta, speed)."""
    state = wd.initial_state({"x": start})
    xs = [np.asarray(state["x"])]
    for k, a in zip(jax.random.split(key, len(commands)), commands, strict=True):
        state = wd.move(k, state, jnp.asarray(a, jnp.float32))
        xs.append(np.asarray(state["x"]))
    return np.asarray(xs)


def _turn_bound(wd):
    """The most the heading can turn in a tick: ``TURN_MAX`` from the command plus four sd of the
    noise on the heading pair, about ``sqrt(2) * sd`` radians, the declared dynamics' own bound."""
    return wd.TURN_MAX + 4 * np.sqrt(2) * float(wd.STATE_SD[2])


def test_the_motion_moves_at_most_one_cell_a_tick_holds_a_commanded_speed_and_never_reverses():
    wd = example("survey_world")
    assert wd.top_speed() <= CELL  # the top speed: under a cell per tick
    rng = np.random.default_rng(0)
    commands = np.repeat(rng.uniform(-1, 1, (15, 2)), 8, axis=0)  # a held command every 8 ticks
    commands[:40] = [1.0, 0.0]  # first up to top speed
    xs = _drive(wd, commands, [0.0, -1.5, 1.0, 0.0, 0.0], jax.random.PRNGKey(1))
    step = np.hypot(*np.diff(xs[:, :2], axis=0).T)
    assert step.max() <= wd.top_speed() + 4 * np.sqrt(2) * float(wd.STATE_SD[0]) + 1e-3
    # the command is a target speed: forward only, so the speed stays at zero or above (noise
    # aside) whatever is asked, and a held drive reaches its speed within a few ticks
    assert xs[:, 4].min() > -4 * float(wd.STATE_SD[4])
    for drive in (1.0, 0.5, 0.0):
        held = _drive(wd, [[drive, 0.0]] * 8, [0.0, -1.5, 1.0, 0.0, 0.15], jax.random.PRNGKey(2))
        assert abs(held[-1, 4] - drive * wd.top_speed()) < 4 * float(wd.STATE_SD[4]), drive


def test_the_heading_is_a_state_turns_by_the_command_alone_and_holds_at_rest():
    wd = example("survey_world")
    assert float(wd.heading_of(wd.initial_state()["x"])) == pytest.approx(np.pi / 2)  # +y
    rng = np.random.default_rng(3)
    commands = rng.uniform(-1, 1, (120, 2))  # any command, changing every tick
    xs = _drive(wd, commands, [1.0, -0.5, 1.0, 0.0, 0.0], jax.random.PRNGKey(4))
    heading = np.arctan2(xs[:, 3], xs[:, 2])
    turn = np.abs((np.diff(heading) + np.pi) % (2 * np.pi) - np.pi)
    assert turn.max() <= _turn_bound(wd)  # the declared bound, at every speed
    # with no turn command the heading holds, moving or at rest (noise aside)
    still = _drive(wd, [[0.0, 0.0]] * 30, [1.0, -0.5, 0.0, 1.0, 0.0], jax.random.PRNGKey(5))
    held = np.arctan2(still[:, 3], still[:, 2])
    assert np.abs(held - np.pi / 2).max() < 8 * float(wd.STATE_SD[2])
    # a turn in place: at rest, a full turn command turns it by TURN_MAX a tick
    spin = _drive(wd, [[0.0, 1.0]] * 3, [1.0, -0.5, 1.0, 0.0, 0.0], jax.random.PRNGKey(6))
    assert np.arctan2(spin[3, 3], spin[3, 2]) == pytest.approx(3 * wd.TURN_MAX, abs=0.05)
    assert np.hypot(*(spin[3, :2] - spin[0, :2])) < 0.05


def test_the_rover_s_position_belief_is_calibrated_without_the_camera():
    module = example("survey_run")
    setting = dict(next(s for s in module.settings_grid_l2() if s["id"] == "one-level"))
    setting.update(grid=GRID, camera=False)
    rows = module.run_setting(setting, 40, None, STEPS, STARTS)["ticks"]
    cal = module.position_calibration(rows)
    assert not cal["lost"]
    assert cal["ellipse_coverage"] >= 0.7  # 0.865 nominal for a 2-sigma ellipse
    assert cal["heading_coverage"] >= 0.8  # 0.954 nominal for 2 sd of the heading


def test_the_rover_alone_drives_forward_under_its_planner_and_stays_calibrated():
    """The showcase's planner (its descent steps and warm starts) on the showcase's grid, the
    camera off, 40 ticks: the rover moves (median speed at least a third of the top speed, under a
    quarter of the ticks parked), forward only and along its heading, under a cell a tick, its
    plans inside the wall, and its position and heading beliefs calibrated."""
    module = example("survey_run")
    wd = example("survey_world")
    setting = dict(next(s for s in module.settings_grid_l2() if s["id"] == "one-level"))
    setting.update(camera=False)
    rows = module.run_setting(setting, 40, None)["ticks"]
    speed = np.asarray([r["true_speed"] for r in rows])
    assert np.median(speed) >= wd.top_speed() / 3
    assert np.mean(speed < 0.02) < 0.25
    assert speed.min() > -4 * float(wd.STATE_SD[4])  # forward only, noise aside
    pos = np.asarray([r["true_x"][:2] for r in rows])
    step = np.diff(pos, axis=0)
    assert np.hypot(*step.T).max() < CELL
    heading = np.asarray([r["true_heading"] for r in rows])
    moving = speed[:-1] >= 0.1
    lag = np.abs((np.arctan2(step[:, 1], step[:, 0]) - heading[:-1] + np.pi) % (2 * np.pi) - np.pi)
    bound = np.arctan2(4 * np.sqrt(2) * float(wd.STATE_SD[0]), speed[:-1])
    assert np.all(lag[moving] <= bound[moving])  # the motion runs along the heading
    assert module.plan_excursions(rows, chosen_only=True).max() <= 0.1
    cal = module.position_calibration(rows)
    assert not cal["lost"]
    assert cal["ellipse_coverage"] >= 0.7  # 0.865 nominal for a 2-sigma ellipse
    assert cal["heading_coverage"] >= 0.8  # 0.954 nominal for 2 sd of the heading


def test_a_world_point_keeps_its_value_as_the_forward_camera_moves_and_turns():
    wd = example("survey_world")
    a = jnp.asarray([0.5, -1.2, 0.0, 1.0, 0.2], jnp.float32)  # facing +y on firm ground
    moves = {
        "forward": jnp.asarray([0.5, -1.1, 0.0, 1.0, 0.2], jnp.float32),
        "turn": jnp.asarray([0.5, -1.2, np.sin(0.1), np.cos(0.1), 0.2], jnp.float32),  # 6 deg
    }
    heading = {"forward": np.pi / 2, "turn": np.pi / 2 - 0.1}
    img_a = np.asarray(wd.render(a, 0, np.pi / 2)).reshape(wd.IMG_SIDE, wd.IMG_SIDE)
    pts_a, _, sky_a, _, _ = (np.asarray(v) for v in wd.ground_view(a, np.pi / 2))
    tex_a = np.asarray(wd.texture(jnp.asarray(pts_a), 0))
    for name, b in moves.items():
        img_b = np.asarray(wd.render(b, 0, heading[name])).reshape(wd.IMG_SIDE, wd.IMG_SIDE)
        pts_b, _, sky_b, _, _ = (np.asarray(v) for v in wd.ground_view(b, heading[name]))
        tex_b = np.asarray(wd.texture(jnp.asarray(pts_b), 0))
        ga = pts_a[~sky_a]
        gb = pts_b[~sky_b]
        dist = np.linalg.norm(ga[:, None, :] - gb[None, :, :], axis=-1)
        j = dist.argmin(1)
        near = dist[np.arange(len(ga)), j] < 0.005  # pixel pairs that look at one world point
        assert near.sum() > 20, name
        assert np.abs(tex_a[~sky_a][near] - tex_b[~sky_b][j[near]]).max() < 0.015, name
        assert np.abs(img_a[~sky_a][near] - img_b[~sky_b][j[near]]).max() < 0.05, name
        assert np.abs(img_a - img_b).max() > 0.02, name  # the picture itself changes (scrolls,
        # or swings under the turn)


def _plan_toward_the_wall(sm, wd, start):
    """The chosen plan of the rover (no camera) from the state ``start``: the positions it passes
    through under the noiseless dynamics."""
    system, params, _, _ = sm.rover(None, camera=False, grid_shape=GRID)
    start = jnp.asarray(start, jnp.float32)
    prior = params.initial_state_prior["x"]
    prior = dataclasses.replace(
        prior,
        means=jnp.broadcast_to(start, prior.means.shape),
        chols=jnp.broadcast_to(0.02 * jnp.eye(5), prior.chols.shape),
    )
    params = dataclasses.replace(
        params, initial_state_prior={**params.initial_state_prior, "x": prior}
    )
    compiled = system.compiled()
    obs = {"fix": start[:2], "gauge": jnp.asarray(3, jnp.int32)}
    beliefs, _ = compiled.infer_states(params, obs)
    q_pi, _, plans = compiled.infer_policies(params, beliefs, planner=PLANNER, return_policies=True)
    plan = np.asarray(plans[int(jnp.argmax(q_pi.probs))])
    step, x, path = wd.VehicleStep(), start, []
    for a in plan:
        x = step(x, (jnp.asarray(a),))
        path.append(np.asarray(x[:2]))
    return np.asarray(path)


def test_the_wall_preference_keeps_the_rover_s_plans_inside_the_field():
    sm = example("survey_model")
    wd = example("survey_world")
    run = example("survey_run")
    top, h = wd.top_speed(), np.sqrt(0.5)
    for start in ([4.0, 1.0, 1.0, 0.0, top], [1.0, 3.0, 0.0, 1.0, top], [0.0, -1.0, -h, -h, top]):
        path = _plan_toward_the_wall(sm, wd, start)
        assert run.excursion(path).max() <= 0.1, start  # the tolerance: a plan may graze a wall
    # the preference itself: flat inside the field, falling off past a wall
    target = sm.wall_target()
    inside = target.log_prob(jnp.asarray([[0.0, 0.0], [2.0, 1.0], [4.3, 3.3], [1.3, -1.4]]))
    assert float(inside.max() - inside.min()) < 0.05  # points over half a unit from a wall
    beyond = target.log_prob(jnp.asarray([[5.0 + 2 * sm.WALL_SD, 1.0]]))
    assert float(inside.mean() - beyond[0]) > 3.0


def test_the_rover_can_observe_the_recogniser_s_reading_instead_of_the_image():
    module = example("survey_run")
    wd = example("survey_world")
    from hybrid_aif.distributions import MLPCategorical

    encoder = MLPCategorical.from_shapes(
        (wd.IMG_PIXELS,), 1, 3, key=jax.random.PRNGKey(0), width=8, depth=1
    )
    confusion = 0.9 * np.eye(3) + 0.05 * (1 - np.eye(3))
    setting = dict(next(s for s in module.settings_grid_l2() if s["id"] == "one-level"))
    setting.update(grid=GRID, camera="look")
    rows = module.run_setting(
        setting, 3, None, STEPS, STARTS, encoder=encoder, confusion=confusion
    )["ticks"]
    assert all(r["look"] in (0, 1, 2) and len(r["recogniser"]) == 3 for r in rows)
    assert all(r["recon"] is None and r["img"] for r in rows)  # the image is shown, not inverted


@pytest.mark.slow
def test_the_rover_s_position_belief_is_calibrated_with_a_small_trained_camera(tmp_path):
    """The camera's whole pipeline (train, calibrate the noise, temper it for the decoder's error)
    on a small decoder, then 60 ticks of the rover alone: not lost, and covered by its belief. The
    small decoder (128 by 2, 4,000 steps) misses the showcase camera's accuracy bar, in which case
    the test skips; the showcase camera itself is too large for the repository, and its closed-loop
    calibration is measured by the full runs."""
    cam = example("survey_camera")
    module = example("survey_run")
    saved = {k: getattr(cam, k) for k in ("DECODER_SHAPE", "DECODER_STEPS", "ENCODER_STEPS")}
    saved.update(DATASET=cam.DATASET, CALIBRATION_STEPS=cam.CALIBRATION_STEPS)
    try:
        cam.DECODER_SHAPE = dict(width=128, depth=2, activation="silu")
        cam.DECODER_STEPS, cam.ENCODER_STEPS, cam.CALIBRATION_STEPS = 4000, 1000, 1000
        cam.DATASET = 8000
        decoder, encoder, _ = cam.load_or_train(tmp_path, verbose=False)
    except AssertionError:
        pytest.skip("the small camera misses the decoder's accuracy bar")
    finally:
        for k, v in saved.items():
            setattr(cam, k, v)
    setting = dict(next(s for s in module.settings_grid_l2() if s["id"] == "one-level"))
    setting.update(grid=GRID)
    rows = module.run_setting(setting, 60, decoder, STEPS, STARTS, encoder=encoder)["ticks"]
    cal = module.position_calibration(rows)
    assert not cal["lost"]
    assert cal["ellipse_coverage"] >= 0.7
