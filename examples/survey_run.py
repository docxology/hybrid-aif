"""Closed-loop runs of the survey rover: a rover that maps a terrain field by driving over it
while keeping its energy up.

The rover (``survey_model``) infers its state, plans a velocity command by gradient descent from
several warm starts, acts, and learns the terrain field by counting each terrain it sees into the
cells it was in. The true world (``survey_world``) is a separate process wired in with ``couple``.
With ``--level 2`` the rover is the fast tier of a composite with a survey scheduler
(``survey_scheduler``): every ``SLOW_TICKS`` fast ticks the scheduler infers, plans and chooses an
option (survey a sector, or charge) that the rover holds; the ablation pins the option to one
sector.
The mission view decides every twelfth tick, the rover view runs between decisions with the
option pinned, the rover's learned field is written back to the composite before each decision,
and the scheduler never learns. The scheduler's cell reading is computed in the runner from the
rover view's position belief, because ``couple`` wires process outputs only.
A level-1 setting is the epistemic terms (the field's novelty, the position fix's information
gain, both or neither) and whether the image preference is declared. The runs of a grid are
written to one bundle for the launch page, and a REPORT.md summarises them.

    python examples/survey_run.py --out DIR [--settings id,id] [--ticks 200] [--level 2]
    python examples/survey_run.py --out DIR --assemble     # merge the part files into bundle
    python examples/survey_trim.py DIR/bundle.json OUT_DIR   # the page's bundle.js
"""

import argparse
import dataclasses
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import survey_camera as cam
import survey_model as sm
import survey_scheduler as ss
import survey_world as wd

from hybrid_aif import Parameter, __version__, cell_mass
from hybrid_aif.composition import (
    ActionControl,
    SharedParameter,
    Wire,
    couple,
    merge,
    run,
    subsystem,
    write_back,
)
from hybrid_aif.description import describe
from hybrid_aif.description.mdl import description_length
from hybrid_aif.objectives import total
from hybrid_aif.objectives.base import term_sign
from hybrid_aif.planning.gradient import make_gradient_planner

PLANNER_STEPS = 40
PLANNER_STARTS = 4  # warm starts of the plan's descent; the best final plan is the policy
LIVE_WEIGHT = 1e-3
FIELD_LR = 1.0  # weight of one tick's terrain counts in the field's Dirichlet (one observation)
FIELD_STRENGTH = 0.1  # pseudo-counts per cell of the field's uniform prior
COVERED_MAX = 0.6  # a cell counts as mapped when its field's top probability exceeds this
SEEN_COUNTS = 1.0  # a cell counts as seen once it has absorbed this many counts
MDL_RESOLUTION = 100.0
MARKS = (50, 100, 200)  # the ticks REPORT.md gives coverage at
MAP_SIDE = 61  # the world maps are published at this many samples per axis
IMAGE_PREFERENCE = True  # whether the default settings declare the image target
EPI = ("transition:terrain", "obs:fix")
NAMES = {"terrain": sm.TERRAIN_NAMES, "energy": sm.ENERGY_NAMES}


def settings_grid():
    """The four settings, each one seed: epistemic terms on (the default), off, the field's
    novelty alone (these three with the image preference set by ``IMAGE_PREFERENCE``), and
    epistemic terms on with no image preference."""

    def setting(sid, label, epistemic, image_preference):
        return dict(
            id=sid,
            label=label,
            epistemic=list(epistemic),
            image_preference=image_preference,
            field_lr=FIELD_LR,
            field_strength=FIELD_STRENGTH,
            energy_logits=list(sm.ENERGY_TARGET),
            planning_belief=sm.PLANNING_BELIEF,
            grid=sm.GRID_SHAPE,
            seed=0,
        )

    return [
        setting("epi", "Novelty and fix information gain", EPI, IMAGE_PREFERENCE),
        setting("pragmatic", "No epistemic terms", (), IMAGE_PREFERENCE),
        setting("novelty-only", "Field novelty only", EPI[:1], IMAGE_PREFERENCE),
        setting("no-image-preference", "Epistemic terms on, no image preference", EPI, False),
    ]


def interval_pair(interval, i):
    return [float(interval.lower[i]), float(interval.upper[i])]


def term_record(tree, i):
    return {
        term: {leaf: interval_pair(term_sign(term) * iv, i) for leaf, iv in leaves.items()}
        for term, leaves in tree.items()
    }


def approximate_record(tree):
    return {
        term: {leaf: bool(iv.approximate) for leaf, iv in leaves.items()}
        for term, leaves in tree.items()
    }


def position_belief(beliefs, pre=""):
    x = beliefs[pre + "x"]
    weights = np.exp(np.asarray(x.log_weights))
    weights = weights / weights.sum()
    comps = []
    for k in range(x.num_components):
        if weights[k] < LIVE_WEIGHT:
            continue
        chol = np.asarray(x.chols[k])
        cov = (chol @ chol.T)[:2, :2]
        comps.append(
            dict(
                w=float(weights[k]),
                mu=[float(m) for m in np.asarray(x.means[k])[:2]],
                cov=[[float(c) for c in row] for row in cov],
            )
        )
    mean = np.asarray(x.mean())
    return dict(
        components=comps,
        terrain=[float(p) for p in np.asarray(beliefs[pre + "terrain"].probs)],
        energy=[float(p) for p in np.asarray(beliefs[pre + "energy"].probs)],
        velocity=[float(mean[2]), float(mean[3])],
        mean=[float(mean[0]), float(mean[1])],
    )


def image_rows(flat):
    """A flat pixel vector (row-major, as the renderer lays it out) as uint8 rows of the picture:
    row 0 is the top of the picture, columns run left to right."""
    grid = np.asarray(flat).reshape(wd.IMG_SIDE, wd.IMG_SIDE)
    return [[int(v) for v in row] for row in wd.to_uint8(grid)]


def field_record(params, strength, home="field"):
    """The field's posterior mean per cell as uint8 probabilities ``(cells, 3)``, the counts it
    has absorbed per cell beyond its prior, the fraction of cells whose top probability exceeds
    ``COVERED_MAX`` and the fraction that has absorbed ``SEEN_COUNTS`` counts."""
    conc = np.asarray(params.parameters[home].concentration, np.float64)
    total_conc = conc.sum(axis=0)
    mean = conc / total_conc
    rows = np.asarray(np.round(mean.T * 255), np.uint8)
    return (
        [[int(v) for v in row] for row in rows],
        [float(c) for c in total_conc - strength],
        float(np.mean(mean.max(axis=0) > COVERED_MAX)),
        float(np.mean(total_conc - strength >= SEEN_COUNTS)),
    )


def make_mind(
    system,
    params0,
    log,
    setting,
    steps,
    starts,
    pre="",
    level_params=None,
    shared=None,
    schedule=None,
):
    """The rover's response: infer, plan, act, push the prior and learn the field.

    ``pre`` is the prefix of the view's names (``"rover."`` on a composite's view). With
    ``level_params`` (the view's params for each option level) and ``shared`` (a dict holding the
    option in force under ``"option"``), the params in force are the pinned level's, carrying the
    field the rover has learned; the learned params are left in ``shared["rover_params"]``.
    ``schedule(beliefs, obs)``, when given, is the slow tier's step: it runs after inference on
    each round, may change ``shared["option"]``, and returns the slow record of the round."""
    compiled = system.compiled()
    planner = make_gradient_planner(steps=steps, starts=starts)
    home = pre + "field"
    state = dict(params=params0, prev=None)
    decoder = params0.likelihood[pre + "img"]

    def in_force():
        if level_params is None:
            return state["params"]
        pinned = level_params[shared["option"]]
        return dataclasses.replace(pinned, parameters=state["params"].parameters)

    def respond(key, obs, carry):
        del key
        t_start = time.time()
        beliefs, detail = compiled.infer_states(in_force(), obs, priors=carry.beliefs, detail=True)
        slow = None if schedule is None else schedule(beliefs, obs)
        option = None if shared is None else shared["option"]
        params = in_force()
        if state["prev"] is not None:
            prev_beliefs, prev_action = state["prev"]
            params = compiled.learn(
                params,
                obs,
                beliefs,
                prev_beliefs=prev_beliefs,
                action=prev_action,
                lr=setting["field_lr"],
            )
            state["params"] = params
            if shared is not None:
                shared["rover_params"] = params
        q_pi, tree, policies = compiled.infer_policies(
            params, beliefs, planner=planner, metrics=True, return_policies=True
        )
        v_plans = policies  # (starts, policy_len, 2): every warm start, q_pi on the picked one
        best = int(jnp.argmax(q_pi.probs))
        action = {pre + "a_v": v_plans[best, 0]}
        pushed = compiled.update_empirical_prior(params, beliefs, action)
        state["prev"] = (beliefs, action)
        mean = np.asarray(beliefs[pre + "x"].mean())
        terrain = int(jnp.argmax(beliefs[pre + "terrain"].probs))
        recon, _ = decoder.diag_moments((jnp.asarray(mean), jax.nn.one_hot(terrain, 3)))
        g = total(tree)
        records = [
            dict(
                id=i,
                start=[float(mean[0]), float(mean[1])],
                v_plan=[[float(c) for c in step] for step in v_plans[i]],
                terms=term_record(tree, i),
                G=interval_pair(g, i),
                q_pi=float(q_pi.probs[i]),
            )
            for i in range(v_plans.shape[0])
        ]
        table, counts, coverage, seen = field_record(params, setting["field_strength"], home)
        entry = dict(
            seconds=time.time() - t_start,
            belief=position_belief(beliefs, pre),
            policies=records,
            chosen=best,
            approximate=approximate_record(tree),
            camera=dict(log_evidence=float(detail.log_evidence[pre + "img"])),
            free_energy=float(detail.free_energy),
            recon=image_rows(recon),
            map=table,
            counts=counts,
            coverage=coverage,
            seen=seen,
        )
        if option is not None:
            entry["option"] = option
        if slow is not None:
            entry["slow"] = slow
        log.append(entry)
        return action, carry._replace(beliefs=pushed)

    return respond


def tick_rows(rec, log, mind, pre, ticks, extra=None):
    """The per-tick records of a run: the world's true state, the rover's readings and actions
    (from the wiring's record of participant ``mind``), and the rover's own log entry."""
    world, rover = rec["world"], rec[mind]
    st = world.states
    rows = []
    for t in range(ticks):
        energy = float(st["energy"][t])
        row = dict(
            t=t,
            v=[float(c) for c in rover.actions[pre + "a_v"][t]],
            true_x=[float(c) for c in st["x"][t]],
            true_terrain=int(st["terrain"][t]),
            true_energy=dict(band=int(wd.band_of(jnp.asarray(energy))), level=energy),
            stalled=bool(wd.band_of(jnp.asarray(energy)) == 0),
            fix=[float(c) for c in rover.observations[pre + "fix"][t]],
            gauge=int(rover.observations[pre + "gauge"][t]),
            img=image_rows(rover.observations[pre + "img"][t]),
            **log[t],
        )
        if extra is not None:
            row.update(extra(t))
        rows.append(row)
    return rows


def run_setting(setting, ticks, decoder, steps=PLANNER_STEPS, starts=PLANNER_STARTS):
    if "tiers" in setting:
        return run_tiered(setting, ticks, decoder, steps, starts)
    system, params, _, _ = sm.rover(
        decoder,
        epistemic=tuple(setting["epistemic"]),
        field_strength=setting["field_strength"],
        grid_shape=setting["grid"],
        planning_belief=setting["planning_belief"],
        energy_target=tuple(setting["energy_logits"]),
        image_preference=setting["image_preference"],
    )
    log = []
    mind = make_mind(system, params, log, setting, steps, starts)
    env, env_params = wd.world()
    wiring = couple(
        minds={"rover": (system, params, mind)},
        processes={"world": (env, env_params)},
        wires=[
            Wire("rover.a_v", "world.a_v"),
            Wire("world.fix", "rover.fix"),
            Wire("world.img", "rover.img"),
            Wire("world.gauge", "rover.gauge"),
        ],
    )
    start = setting.get("start_energy")
    states = None if start is None else {"world": {"energy": start}}
    rec = run(wiring, ticks=ticks, key=jax.random.PRNGKey(setting["seed"]), states=states)
    rows = tick_rows(rec, log, "rover", "", ticks)
    return dict(ticks=rows, planner=dict(steps=steps, starts=starts))


SLOW_TICKS = 12  # fast ticks per slow step
READINGS_AVAILABLE = "readings" in {f.name for f in dataclasses.fields(ss.System)}
PINNED_OPTION = 0  # the one-level run holds this survey sector for the whole run


def settings_grid_l2():
    """The two-tier settings: the full composite, and the same composite with the option pinned
    to one survey sector for the whole run (level 1)."""

    def setting(sid, label, tiers):
        return dict(
            id=sid,
            label=label,
            tiers=tiers,
            epistemic=list(EPI),
            image_preference=True,
            field_lr=FIELD_LR,
            field_strength=FIELD_STRENGTH,
            energy_logits=list(sm.ENERGY_TARGET),
            planning_belief=sm.PLANNING_BELIEF,
            grid=sm.GRID_SHAPE,
            slow_ticks=SLOW_TICKS,
            seed=0,
        )

    return [
        setting("two-level", "Scheduler over the rover", 2),
        setting("one-level", "The option pinned to one survey sector", 1),
    ]


def build_composite(decoder, setting):
    """The merged composite, with the rover and scheduler's ``subsystem`` views: ``(composite,
    composite params, mission view, mission params, rover views by option level, grid)``. The rover
    views are one System with params that differ in the pinned option's stub only."""
    rs, rp, grid, centres = sm.rover(
        decoder,
        epistemic=tuple(setting["epistemic"]),
        field_strength=setting["field_strength"],
        grid_shape=setting["grid"],
        planning_belief=setting["planning_belief"],
        energy_target=tuple(setting["energy_logits"]),
        image_preference=setting["image_preference"],
        tiered=True,
    )
    ms, mp = ss.scheduler(
        grid,
        centres,
        energy_target=tuple(setting.get("scheduler_logits") or setting["energy_logits"]),
        field_strength=setting["field_strength"],
        policy_len=rs.policy_len,
        durations=setting.get("durations", False),
        readings=setting.get("readings", False),
    )
    cs, cp = merge(
        {"mission": ms, "rover": rs},
        {"mission": mp, "rover": rp},
        [
            ActionControl("mission.u_option", "rover.u_opt"),
            SharedParameter("rover.field", "mission.field"),
        ],
    )
    mview, mparams = subsystem(cs, cp, "mission")
    mview = dataclasses.replace(mview, policy_len=2)
    levels = {}
    for level in range(sm.N_OPTIONS):
        rview, rparams = subsystem(cs, cp, "rover", partner_priors={"mission.u_option": level})
        levels[level] = (rview, rparams)
    return cs, cp, mview, mparams, levels, grid


def write_back_gains(cs, composite, before, after):
    """Return the rover view's learning since ``before`` to the composite's shared field."""
    if len(inspect.signature(write_back).parameters) >= 5:
        return write_back(cs, composite, "rover", before, after)
    return write_back(cs, composite, "rover", after)


def make_schedule(cs, cp, mview, mparams0, shared, setting, baseline, grid):
    """The scheduler's step, called by the rover's mind on every round: every ``slow_ticks``
    rounds it reads the rover's cell (the argmax of the position belief's cell mass) and the
    gauge, infers, plans and chooses the option, which is left in ``shared["option"]`` for the
    rover to hold. Before a decision the rover's learned field (its gain since the last
    write-back) is written into the composite and the scheduler's params are rebuilt from it, so
    its novelty reads the counts gathered so far. The scheduler never learns."""
    compiled = mview.compiled()
    state = dict(
        t=0, params=mparams0, composite=cp, baseline=baseline, carry=mview.initial_priors(mparams0)
    )
    every = setting["slow_ticks"]

    def step(beliefs_fast, obs):
        decide = state["t"] % every == 0
        state["t"] += 1
        if not decide:
            return dict(decision=False)
        t_start = time.time()
        if "rover_params" in shared:
            state["composite"] = write_back_gains(
                cs, state["composite"], state["baseline"], shared["rover_params"]
            )
            state["baseline"] = shared["rover_params"]
            state["params"] = subsystem(cs, state["composite"], "mission")[1]
        params = state["params"]
        cell = int(jnp.argmax(cell_mass(beliefs_fast["rover.x"], grid)))
        slow_obs = {"mission.cell_obs": jnp.asarray(cell), "mission.gauge": obs["rover.gauge"]}
        beliefs, _ = compiled.infer_states(params, slow_obs, priors=state["carry"])
        key = jax.random.PRNGKey(0)  # unused by the deterministic selection
        action, q_pi, tree = compiled.act(
            key, params, beliefs, metrics=True, selection="deterministic"
        )
        option = int(action["mission.option"])
        shared["option"] = option
        state["carry"] = compiled.update_empirical_prior(params, beliefs, action)
        sequences = np.asarray(mview.policies(params))
        g = total(tree)
        records = [
            dict(
                options=[int(a) for a in sequences[i].ravel()],
                terms=term_record(tree, i),
                G=interval_pair(g, i),
                q_pi=float(q_pi.probs[i]),
            )
            for i in range(sequences.shape[0])
        ]
        midpoint = [float(np.mean(r["G"])) for r in records]
        first = [r["options"][0] for r in records]
        efe = [
            min(m for m, f in zip(midpoint, first, strict=True) if f == o)
            for o in range(sm.N_OPTIONS)
        ]
        q_option = [
            float(sum(r["q_pi"] for r in records if r["options"][0] == o))
            for o in range(sm.N_OPTIONS)
        ]
        chosen = max(
            (i for i, f in enumerate(first) if f == option), key=lambda i: records[i]["q_pi"]
        )
        return dict(
            decision=True,
            option=option,
            cell=cell,
            efe_by_option=efe,
            q_by_option=q_option,
            policies=records,
            chosen=chosen,
            belief=dict(
                energy=[float(p) for p in np.asarray(beliefs["mission.energy"].probs)],
                visit=[float(p) for p in np.asarray(beliefs["mission.visit"].probs)],
            ),
            seconds=time.time() - t_start,
        )

    return step


def run_tiered(setting, ticks, decoder, steps=PLANNER_STEPS, starts=PLANNER_STARTS):
    """A run on the composite's views: with ``tiers == 2`` the scheduler picks the option every
    ``slow_ticks`` rounds; with ``tiers == 1`` the option is pinned to ``PINNED_OPTION``."""
    cs, cp, mview, mparams, levels, grid = build_composite(decoder, setting)
    two = setting["tiers"] == 2
    shared = dict(option=0 if two else PINNED_OPTION)
    log = []
    rview, rparams = levels[shared["option"]]
    level_params = {k: v[1] for k, v in levels.items()}
    schedule = (
        make_schedule(cs, cp, mview, mparams, shared, setting, rparams, grid) if two else None
    )
    fast = make_mind(
        rview, rparams, log, setting, steps, starts, "rover.", level_params, shared, schedule
    )
    env, env_params = wd.world()
    wiring = couple(
        minds={"fast": (rview, rparams, fast)},
        processes={"world": (env, env_params)},
        wires=[
            Wire("fast.rover.a_v", "world.a_v"),
            Wire("world.fix", "fast.rover.fix"),
            Wire("world.img", "fast.rover.img"),
            Wire("world.gauge", "fast.rover.gauge"),
        ],
    )
    rec = run(wiring, ticks=ticks, key=jax.random.PRNGKey(setting["seed"]))
    rows = tick_rows(rec, log, "fast", "rover.", ticks)
    return dict(ticks=rows, planner=dict(steps=steps, starts=starts))


def downsampled_maps():
    xs, ys, height, openness, light, slope, sd_fix, terrain = wd.maps()
    step = (len(xs) - 1) // (MAP_SIDE - 1)

    def thin(a, digits=3):
        return np.round(np.asarray(a)[::step, ::step], digits).tolist()

    return dict(
        xs=[round(float(v), 4) for v in xs[::step]],
        ys=[round(float(v), 4) for v in ys[::step]],
        layout="each map is indexed [ix][iy], x first",
        height=thin(height),
        light=thin(light),
        openness=thin(openness),
        sd_fix=thin(sd_fix),
        slope=thin(slope),
        terrain=np.asarray(terrain)[::step, ::step].astype(int).tolist(),
    )


def dynamics_record():
    """The true dynamics a velocity plan integrates under, per terrain (firm, rough, ice)."""
    return dict(
        dt=float(wd.DT),
        gain=[float(g) for g in wd.GAIN],
        keep=[float(k) for k in wd.KEEP],
        terrain_sd=[float(v) for v in wd.TERRAIN_SD],
        terrain_names=list(wd.TERRAIN_NAMES),
        rule=(
            "per tick, with the terrain at the current position: position += dt * velocity (the "
            "velocity before the tick), velocity = keep * velocity + gain * command (command in "
            "[-1, 1]^2) + noise of sd terrain_sd; then the position is clipped to the field, "
            "the velocity into a wall is zeroed, and an empty energy band zeroes the gain"
        ),
    )


def meta_record(ticks, camera_report, grid):
    return dict(
        library_version=__version__,
        ticks=ticks,
        grid=dict(shape=list(grid.shape), low=list(grid.low), high=list(grid.high)),
        field=[list(wd.FIELD[0]), list(wd.FIELD[1])],
        start=list(wd.START),
        world=downsampled_maps(),
        dynamics=dynamics_record(),
        names=NAMES,
        image=dict(
            side=wd.IMG_SIDE,
            layout=(
                "rows of uint8 0..255, the forward camera: row 0 is the top of the picture, the "
                "top quarter is sky, columns run left to right"
            ),
            half_view_note=(
                "the camera looks along +y from the rover; the ground rows span a perspective "
                "depth of 1 / (0.12 + v) ahead, so the picture shows only the half of the field "
                "in front of the rover"
            ),
            camera_report=camera_report,
        ),
        map_layout=(
            "map: one row per grid cell (row-major over grid.shape, x first), three uint8 "
            "probabilities (firm, rough, ice) of the field's posterior mean, scaled to 0..255; "
            "counts: the observations absorbed per cell"
        ),
        sign_convention=(
            "every leaf is added to G with its registered sign already applied, so the leaves of "
            "a policy sum to its G and an epistemic leaf appears negative"
        ),
    )


def first_tick(rows, inside):
    """The first tick whose true position satisfies ``inside(x, y)``, or None."""
    return next((r["t"] for r in rows if inside(*r["true_x"][:2])), None)


def run_summary(run_, marks=MARKS):
    """The figures REPORT.md gives per setting; coverage and ``seen`` at the ticks ``marks``."""
    rows = run_["ticks"]
    n = len(rows)
    err = [
        float(np.hypot(*(np.asarray(r["belief"]["mean"]) - np.asarray(r["true_x"][:2]))))
        for r in rows
    ]
    bands = np.bincount([r["true_energy"]["band"] for r in rows], minlength=4)
    terrain = np.bincount([r["true_terrain"] for r in rows], minlength=3)
    secs = [r["seconds"] for r in rows[1:]] or [rows[0]["seconds"]]
    coverage = {f"t{k}": rows[min(k, n) - 1]["coverage"] for k in marks}
    (fx, fy), fr = wd.ROUGH_PATCHES[0]
    (ix, iy), ir = wd.ICE_PATCH
    return dict(
        coverage=coverage,
        seen={f"t{k}": rows[min(k, n) - 1]["seen"] for k in marks},
        first_ice=first_tick(rows, lambda x, y: np.hypot(x - ix, y - iy) <= ir),
        first_far_rough=first_tick(rows, lambda x, y: np.hypot(x - fx, y - fy) <= fr),
        first_in_sector=[
            first_tick(rows, lambda x, y, k=k: int(wd.zone_of(np.asarray([x, y]))) == k)
            for k in range(sm.N_SURVEY)
        ],
        position_error=float(np.mean(err)),
        ticks_per_band=[int(b) for b in bands],
        ticks_on_ice=int(terrain[2]),
        ticks_on_rough=int(terrain[1]),
        final_band=int(rows[-1]["true_energy"]["band"]),
        mean_seconds=float(np.mean(secs)),
        first_velocities=[[round(c, 3) for c in r["v"]] for r in rows[:10]],
    )


def report_text(settings, runs, marks=MARKS, mdl=None):
    on = ", ".join(s["id"] for s in settings if s["image_preference"]) or "none"
    off = ", ".join(s["id"] for s in settings if not s["image_preference"]) or "none"
    lines = ["# Survey rover runs", ""]
    lines += [f"Image preference on: {on}; off: {off}.", ""]
    summaries = {s["id"]: run_summary(runs[s["id"]], marks) for s in settings}
    if len(settings) > 1 and all("tiers" in s for s in settings):
        head = "".join(f" tick {k} |" for k in marks)
        lines += [
            "Coverage over time (cells whose field's top probability exceeds "
            f"{COVERED_MAX}), side by side:",
            "",
            f"| setting |{head}",
            "|---|" + "---|" * len(marks),
        ]
        for s in settings:
            cells = "".join(f" {summaries[s['id']]['coverage'][f't{k}']:.3f} |" for k in marks)
            lines.append(f"| {s['id']} |{cells}")
        lines.append("")
    for s in settings:
        r = summaries[s["id"]]
        cov = ", ".join(f"{k[1:]}: {v:.3f}" for k, v in r["coverage"].items())
        seen = ", ".join(f"{k[1:]}: {v:.3f}" for k, v in r["seen"].items())
        bands = ", ".join(
            f"{n} {c}" for n, c in zip(sm.ENERGY_NAMES, r["ticks_per_band"], strict=True)
        )
        lines += [
            f"## {s['id']}: {s['label']}",
            "",
            f"- seen (cells with at least {SEEN_COUNTS} counts) at tick {seen}",
            f"- coverage (cells whose field's top probability exceeds {COVERED_MAX}) at tick {cov}",
            f"- mean position error: {r['position_error']:.3f}",
            f"- ticks per true energy band: {bands}",
            f"- ticks on ice: {r['ticks_on_ice']}, on rough ground: {r['ticks_on_rough']}",
            f"- first tick on ice: {r['first_ice']}, "
            f"in the far rough patch: {r['first_far_rough']}",
            f"- final energy band: {sm.ENERGY_NAMES[r['final_band']]}",
            f"- mean seconds per tick after the first: {r['mean_seconds']:.2f}",
            f"- first ten chosen velocities: {r['first_velocities']}",
        ]
        slow = [t["slow"] for t in runs[s["id"]]["ticks"] if t.get("slow", {}).get("decision")]
        if slow:
            names = ss.OPTION_NAMES
            decisions = [
                (t["t"], names[t["slow"]["option"]])
                for t in runs[s["id"]]["ticks"]
                if t.get("slow", {}).get("decision")
            ]
            charges = [t for t, o in decisions if o == "charge"]
            first_charge = charges[0] if charges else None
            options = [d["option"] for d in slow]
            surveyed = options.index(0) if 0 in options else None
            after = (
                slow[surveyed + 1] if surveyed is not None and surveyed + 1 < len(slow) else None
            )
            lines += [
                f"- first charge tick: {first_charge}",
                f"- first tick in each sector (1 to 4): {r['first_in_sector']}",
                f"- EFE by option at the first decision: "
                f"{np.round(slow[0]['efe_by_option'], 2).tolist()}",
                f"- EFE by option at the decision after sector 1 is surveyed: "
                f"{np.round(after['efe_by_option'], 2).tolist() if after else 'no such decision'}",
                f"- options chosen (tick: option): {', '.join(f'{t}: {o}' for t, o in decisions)}",
                f"- charge chosen at ticks: {charges if charges else 'never'}",
                f"- mean seconds per slow decision: "
                f"{np.mean([d['seconds'] for d in slow]):.2f} (the first includes compile)",
            ]
        lines.append("")
    if mdl is not None:
        parts = {c["name"]: c["nats"] for c in mdl["components"]}
        lines += [
            "## Description length",
            "",
            f"- the composite: {mdl['total']:.0f} nats ({mdl['total_exact']:.0f} exact)",
            f"- the scheduler's known visit table alone (transition:mission.visit): "
            f"{parts.get('transition:mission.visit')} nats",
            f"- the scheduler's known energy table (transition:mission.energy): "
            f"{parts.get('transition:mission.energy')} nats",
            f"- the cell reading (likelihood:mission.cell_obs): "
            f"{parts.get('likelihood:mission.cell_obs')} nats",
            f"- the shared field (parameter:rover.field): "
            f"{parts.get('parameter:rover.field')} nats",
            "",
        ]
    return "\n".join(lines)


def graph_record(system, params):
    """The model's graph for the page: tiers (a composite's members, by name prefix; one tier
    for a single System) of factors, modalities and controls, the transition, likelihood and
    target edges, and the parameter variables with the factors that read them."""
    names = system.names
    all_nodes = (*system.factor_names, *system.action_names)
    tiers = {}

    def tier(n):
        name = n.split(".")[0] if "." in n else "rover"
        return tiers.setdefault(name, dict(name=name, factors=[], modalities=[], controls=[]))

    cells = dict(system.cells)
    for f in names.states:
        d = params.initial_state_prior[f]
        if hasattr(d, "logits"):
            rec = dict(name=f, family="categorical", levels=int(d.logits.shape[-1]))
        else:
            rec = dict(name=f, family="gaussian_mixture", dim=int(d.means.shape[-1]))
        if f in cells:
            rec["cells"] = dict(shape=list(cells[f].shape), axes=list(cells[f].axes))
        tier(f)["factors"].append(rec)
    families = {
        "LinearGaussianObservation": "linear_gaussian",
        "MLPConditional": "neural",
        "ConditionalCategorical": "categorical",
    }
    for m, parents in zip(names.modalities, system.A_deps, strict=True):
        lik = type(params.likelihood[m]).__name__
        if lik not in families:
            raise ValueError(f"no graph family for the likelihood of {m!r}: {lik}")
        tier(m)["modalities"].append(
            dict(name=m, family=families[lik], parents=[all_nodes[p] for p in parents])
        )
    levels = system._control_levels(params)
    for node, n_levels in zip(names.controls, levels, strict=True):
        if node in params.control:
            box = params.control[node]
            rec = dict(
                name=node,
                kind="float",
                box=[[float(b) for b in box.low], [float(b) for b in box.high]],
                durations=None,
            )
        else:
            rec = dict(name=node, kind="level", levels=int(n_levels), durations=None)
        tier(node)["controls"].append(rec)
    edges = []
    for child, parents in zip(system.factor_names, system.B_deps, strict=True):
        if child in names.controls:
            continue
        for p in parents:
            if all_nodes[p] != child:
                edges.append({"from": all_nodes[p], "to": child, "kind": "transition"})
    for m, parents in zip(names.modalities, system.A_deps, strict=True):
        for p in parents:
            edges.append({"from": all_nodes[p], "to": m, "kind": "likelihood"})
    for key, parents in system.C_deps:
        for p in parents:
            edges.append({"from": p, "to": key, "kind": "target"})
    variables = []
    for home in params.parameters:
        readers = [
            f
            for group in (params.transition, params.likelihood)
            for f, decl in group.items()
            if isinstance(decl, Parameter) and decl.name == home
        ]
        variables.append(dict(name=home, readers=readers))
        edges += [{"from": home, "to": f, "kind": "parameter"} for f in readers]
    return dict(tiers=list(tiers.values()), edges=edges, parameters=variables)


def _entry_size(params, kind, name):
    table = {
        "likelihood": params.likelihood,
        "transition": params.transition,
        "initial_state_prior": params.initial_state_prior,
        "parameter": params.parameters,
    }.get(kind)
    if table is None or name not in table:
        return None
    leaves = jax.tree_util.tree_leaves(eqx.filter(table[name], eqx.is_array))
    return int(sum(int(np.prod(leaf.shape)) for leaf in leaves))


def mdl_record(system, params):
    """The description-length table: nats per address of the model's description."""
    document, arrays = describe(system, params)
    dl = description_length(document, MDL_RESOLUTION, arrays=arrays)
    unscored = {a for a, link in dl.unscored if link == "family"}
    rows = []
    for address, links in dl.per_address.items():
        kind, _, name = address.partition(":")
        nats = None if address in unscored else float(links.structure + links.parameters)
        rows.append(
            dict(
                name=address,
                kind=kind,
                nats=nats,
                exact=bool(links.exact),
                entries=_entry_size(params, kind, name),
            )
        )
    scored = [r for r in rows if r["nats"] is not None]
    return dict(
        total=float(sum(r["nats"] for r in scored)),
        total_exact=float(sum(r["nats"] for r in scored if r["exact"])),
        library_total=float(dl.total),
        components=rows,
    )


def assemble(out_dir, camera_report, decoder):
    """Merge the part files into the bundle and write the report."""
    runs, settings = {}, {}
    for path in sorted(out_dir.glob("part_*.json")):
        part = json.loads(path.read_text())
        runs.update(part["runs"])
        settings.update(part["settings"])
    if not runs:
        raise SystemExit(f"no part files under {out_dir}")
    ticks = {len(r["ticks"]) for r in runs.values()}
    if len(ticks) != 1:
        raise SystemExit(f"the runs have different lengths: {sorted(ticks)}")
    tiered = all("tiers" in s for s in settings.values())
    order = [
        s["id"] for s in (settings_grid_l2() if tiered else settings_grid()) if s["id"] in runs
    ]
    settings = [settings[sid] for sid in order]
    runs = {sid: runs[sid] for sid in order}
    n_grid = settings[0].get("grid", sm.GRID_SHAPE)
    grid, _ = sm.field_grid(n_grid)
    if tiered:
        cs, cp, _, _, _, _ = build_composite(decoder, settings[0])
        model = (cs, cp)
    else:
        model = sm.rover(decoder, image_preference=True, grid_shape=n_grid)[:2]
    n = ticks.pop()
    meta = meta_record(n, camera_report, grid)
    if tiered:
        meta["tiers"] = dict(
            slow_ticks=settings[0]["slow_ticks"],
            options=ss.OPTION_NAMES,
            zones=ss.SECTOR_NAMES,
            zone_split=list(wd.ZONE_SPLIT),
            sector_centres=[[float(c) for c in p] for p in sm.sector_centres()],
            pinned_option=PINNED_OPTION,
        )
    bundle = dict(meta=meta, graph=graph_record(*model), settings=settings, runs=runs)
    try:
        bundle["mdl"] = mdl_record(*model)
        if tiered:
            bundle["description"] = describe(*model)[0]
    except Exception as exc:  # the description refuses some models; say so and carry on
        print(f"mdl skipped: {type(exc).__name__}: {exc}")
    text = json.dumps(bundle)
    (out_dir / "bundle.json").write_text(text)
    marks = (tuple(k for k in (100, 200, 300, 400) if k <= n) or (n,)) if tiered else MARKS
    (out_dir / "REPORT.md").write_text(report_text(settings, runs, marks, bundle.get("mdl")))
    print(f"bundle: {len(runs)} runs, {len(text) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--out", required=True, help="output directory (also holds the camera cache)")
    ap.add_argument("--settings", default=None, help="comma-separated setting ids")
    ap.add_argument("--assemble", action="store_true", help="merge the part files and stop")
    ap.add_argument("--steps", type=int, default=PLANNER_STEPS, help="descent steps per plan")
    ap.add_argument("--starts", type=int, default=PLANNER_STARTS, help="warm starts per plan")
    ap.add_argument("--field-lr", type=float, default=FIELD_LR, help="weight of the counts")
    ap.add_argument("--field-strength", type=float, default=FIELD_STRENGTH, help="prior counts")
    ap.add_argument("--energy-logits", default=None, help="four log-preferences, empty..full")
    ap.add_argument("--planning-belief", default=sm.PLANNING_BELIEF, choices=("prior", "posterior"))
    ap.add_argument("--start-energy", type=float, default=None, help="the world's initial energy")
    ap.add_argument("--grid", type=int, default=sm.GRID_SHAPE, help="field cells per axis")
    ap.add_argument("--level", type=int, default=1, choices=(1, 2), help="the settings grid")
    ap.add_argument("--slow-ticks", type=int, default=SLOW_TICKS, help="fast ticks per slow step")
    ap.add_argument("--scheduler-logits", default=None, help="the scheduler's own ladder")
    ap.add_argument("--durations", action="store_true", help="declare 12-tick durations")
    ap.add_argument(
        "--readings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pool the scheduler's map reading over an option (default: on if available)",
    )
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    decoder, _, camera_report = cam.load_or_train(out_dir)
    if args.assemble:
        assemble(out_dir, camera_report, decoder)
        return
    settings = settings_grid() if args.level == 1 else settings_grid_l2()
    if args.settings:
        keep = set(args.settings.split(","))
        settings = [s for s in settings if s["id"] in keep]
    for s in settings:
        s["field_lr"] = args.field_lr
        s["field_strength"] = args.field_strength
        s["planning_belief"] = args.planning_belief
        s["grid"] = args.grid
        if args.scheduler_logits:
            s["scheduler_logits"] = [float(v) for v in args.scheduler_logits.split(",")]
        if args.durations:
            s["durations"] = True
        if "tiers" in s:
            s["readings"] = READINGS_AVAILABLE if args.readings is None else args.readings
        if "slow_ticks" in s:
            s["slow_ticks"] = args.slow_ticks
        if args.start_energy is not None:
            s["start_energy"] = args.start_energy
        if args.energy_logits:
            s["energy_logits"] = [float(v) for v in args.energy_logits.split(",")]
        t0 = time.time()
        result = run_setting(s, args.ticks, decoder, args.steps, args.starts)
        elapsed = time.time() - t0
        result["seconds"] = elapsed
        part = out_dir / f"part_{s['id']}{args.tag}.json"
        part.write_text(json.dumps(dict(runs={s["id"]: result}, settings={s["id"]: s})))
        secs = sorted(r["seconds"] for r in result["ticks"])
        print(
            f"{s['id']}: {elapsed:.1f}s ({elapsed / args.ticks:.2f}s/tick; median tick "
            f"{secs[len(secs) // 2]:.2f}s, max {secs[-1]:.2f}s) final coverage "
            f"{result['ticks'][-1]['coverage']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
