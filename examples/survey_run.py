"""Closed-loop runs of the survey rover: a rover that maps a terrain field by driving over it
while keeping its energy up.

The rover (``survey_model``) infers its state, plans a velocity command by gradient descent from
several warm starts, acts, and learns the terrain field by counting each terrain it sees into the
cells it was in. The true world (``survey_world``) is a System of its own, coupled to the rover with
``couple``: the rover's velocity action node drives the world's control node ``v``, and the world's
emissions (the fix, the image and the gauge) feed the rover's modalities. With ``--level 2`` the
rover and a survey scheduler (``survey_scheduler``) are first merged into one agent: the scheduler's
option action node drives the rover's option control node (an ``Identify``), the field is one
parameter variable (an ``Identify``), and the scheduler's cell modality observes an emission of the
rover, its position belief through the cell indicator realised by ``mode`` (the ``Link`` sugar).
The agent is coupled to the world as one component, and one ``run`` call over the closed System
steps three leaves: the scheduler by ``plan()`` (the enumerated planner, its metrics kept in the
carry's trace), the rover by a custom step (it learns the field, decodes the image for the page and
emits the cell), and the world by its hand-written step. The scheduler's params declare a duration
for every option, so it decides once per option, every ``SLOW_TICKS`` rounds, and the rover holds
the option between decisions. What the rover learns is written back to the shared field each
round, so the scheduler's novelty reads the counts gathered so far; the scheduler never learns. The
ablation pins the option to one sector. The order is sequential: the scheduler steps first, the
rover as soon as the option it reads is realised, the world last on the command just realised; the
scheduler's cell reading is the rover's emission of the round before, absent in round 0. A level-1
setting is the epistemic terms (the field's novelty, the position fix's information gain, both or
neither) and whether the image preference is declared. The runs of a grid are written to one
bundle for the launch page, and a REPORT.md summarises them.

    python examples/survey_run.py --out DIR [--settings id,id] [--ticks 200] [--level 2]
    python examples/survey_run.py --out DIR --assemble     # merge the part files into bundle
    python examples/survey_trim.py DIR/bundle.json OUT_DIR   # the page's bundle.js
"""

import argparse
import dataclasses
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import survey_camera as cam
import survey_model as sm
import survey_scheduler as ss
import survey_world as wd

from hybrid_aif import Parameter, __version__
from hybrid_aif.composition import Identify, Link, couple, joint_planner, merge, plan, run
from hybrid_aif.description import describe
from hybrid_aif.description.mdl import description_length
from hybrid_aif.objectives import total
from hybrid_aif.objectives.base import term_sign
from hybrid_aif.planning.gradient import make_gradient_planner
from hybrid_aif.reads import EMIT_TAG

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


def page_name(leaf, outer):
    """A term tree's leaf name as the page knows it, the agent's own: ``outer``, the prefix the
    coupling gives the agent's names (``"agent."`` or ``"rover."``), dropped after any
    ``<slot>:`` head (``transition:agent.rover.terrain`` is ``transition:rover.terrain``)."""
    head, colon, name = leaf.rpartition(":")
    return f"{head}{colon}{name.removeprefix(outer)}"


def term_record(tree, i, outer):
    return {
        term: {
            page_name(leaf, outer): interval_pair(term_sign(term) * iv, i)
            for leaf, iv in leaves.items()
        }
        for term, leaves in tree.items()
    }


def approximate_record(tree, outer):
    return {
        term: {page_name(leaf, outer): bool(iv.approximate) for leaf, iv in leaves.items()}
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


class RoverCarry(NamedTuple):
    """The rover step's carry: the prior for the next round, the round's posterior and action
    (what the next round's learning reads), and the round's record for the page."""

    beliefs: Any = None
    previous: Any = None
    action: Any = None
    record: Any = None


def rover_step(setting, steps, starts, pre, decoder):
    """The rover's step for ``run``: infer, learn the field, plan, act, push the prior and emit.
    ``params`` arrive as the argument, and the params that learned the field are returned (the
    field is counted from the second round on, with the previous round's posterior and action
    from the carry); ``pre`` is the prefix of the view's names (``"agent.rover."`` for the
    two-tier agent coupled to the world, ``"rover."`` for the rover coupled alone). The view's
    emissions (the cell the scheduler reads, two-tier only) are realised from the posterior.
    The step is handed its view sliced at the option it reads (``sliced``), as a planner is,
    since the rover plans over a horizon at one option. Each round's record for the page is the
    carry's ``record``; ``decoder`` renders the reconstruction it holds."""
    planner = make_gradient_planner(steps=steps, starts=starts)
    home = pre + "field"
    outer = pre.split(".", 1)[0] + "."  # the coupling's prefix of the agent's names
    faces = {}

    def step(system, key, obs, carry, params):
        if obs is None:
            return {}, RoverCarry(), params
        t_start = time.time()
        if faces.get("system") is not system:
            faces.update(system=system, compiled=system.compiled())
        compiled = faces["compiled"]
        beliefs, detail = compiled.infer_states(params, obs, priors=carry.beliefs, detail=True)
        if carry.previous is not None:
            params = compiled.learn(
                params,
                obs,
                beliefs,
                prev_beliefs=carry.previous,
                action=carry.action,
                lr=setting["field_lr"],
            )
        q_pi, tree, policies = compiled.infer_policies(
            params, beliefs, planner=planner, metrics=True, return_policies=True
        )
        v_plans = policies  # (starts, policy_len, 2): every warm start, q_pi on the picked one
        best = int(jnp.argmax(q_pi.probs))
        action = {pre + "a_v": v_plans[best, 0]}
        pushed = compiled.update_empirical_prior(params, beliefs, action)
        mean = np.asarray(beliefs[pre + "x"].mean())
        terrain = int(jnp.argmax(beliefs[pre + "terrain"].probs))
        recon, _ = decoder.diag_moments((jnp.asarray(mean), jax.nn.one_hot(terrain, 3)))
        g = total(tree)
        records = [
            dict(
                id=i,
                start=[float(mean[0]), float(mean[1])],
                v_plan=[[float(c) for c in step] for step in v_plans[i]],
                terms=term_record(tree, i, outer),
                G=interval_pair(g, i),
                q_pi=float(q_pi.probs[i]),
            )
            for i in range(v_plans.shape[0])
        ]
        table, counts, coverage, seen = field_record(params, setting["field_strength"], home)
        record = dict(
            seconds=time.time() - t_start,
            belief=position_belief(beliefs, pre),
            policies=records,
            chosen=best,
            approximate=approximate_record(tree, outer),
            camera=dict(log_evidence=float(detail.log_evidence[pre + "img"])),
            free_energy=float(detail.free_energy),
            recon=image_rows(recon),
            map=table,
            counts=counts,
            coverage=coverage,
            seen=seen,
        )
        #  the view's emissions (the cell the scheduler reads) are realised from the posterior
        emitted = system.emit(jax.random.fold_in(key, EMIT_TAG), params, beliefs)
        carry = RoverCarry(beliefs=pushed, previous=beliefs, action=action, record=record)
        return {**action, **emitted}, carry, params

    step.sliced = True
    return step


def tick_rows(rec, rover, ticks, extra=None):
    """The per-tick records of a run, read from its ``Trajectory``: the world's true state at the
    start of the tick (its carry), the readings the rover was handed and the command it chose
    (the record of the stepped node ``rover``), and the rover's own record of the tick (its carry
    after the tick)."""
    world, mind = rec["world"], rec[rover]
    rows = []
    for t in range(ticks):
        state = world.carries[t]
        energy = float(state["energy"])
        row = dict(
            t=t,
            v=[float(c) for c in mind.intended["a_v"][t]],
            true_x=[float(c) for c in state["x"]],
            true_terrain=int(state["terrain"]),
            true_energy=dict(band=int(wd.band_of(jnp.asarray(energy))), level=energy),
            stalled=bool(wd.band_of(jnp.asarray(energy)) == 0),
            fix=[float(c) for c in mind.received["fix"][t]],
            gauge=int(mind.received["gauge"][t]),
            img=image_rows(mind.received["img"][t]),
            **mind.carries[t + 1].record,
        )
        if extra is not None:
            row.update(extra(t))
        rows.append(row)
    return rows


def world_bindings(agent):
    """The identifications that couple the world to the rover inside the component ``agent``
    (``"agent.rover"`` or ``"rover"``): the rover's command drives the world's ``v``, and the
    world's three emissions feed the rover's modalities."""
    return [
        Identify(f"{agent}.a_v", "world.v"),
        Identify("world.fix", f"{agent}.fix"),
        Identify("world.img", f"{agent}.img"),
        Identify("world.gauge", f"{agent}.gauge"),
    ]


def run_setting(setting, ticks, decoder, steps=PLANNER_STEPS, starts=PLANNER_STARTS):
    """One run of a setting: the rover coupled to the world (level 1), or the two-tier agent
    (``run_tiered``)."""
    if "tiers" in setting:
        return run_tiered(setting, ticks, decoder, steps, starts)
    rs, rp, _, _ = sm.rover(
        decoder,
        epistemic=tuple(setting["epistemic"]),
        field_strength=setting["field_strength"],
        grid_shape=setting["grid"],
        planning_belief=setting["planning_belief"],
        energy_target=tuple(setting["energy_logits"]),
        image_preference=setting["image_preference"],
    )
    system, params = couple(
        {"rover": (rs, rp), "world": wd.world_system()}, world_bindings("rover")
    )
    start = setting.get("start_energy")
    rec = run(
        system,
        params,
        ticks,
        jax.random.PRNGKey(setting["seed"]),
        steps={
            "rover": rover_step(setting, steps, starts, "rover.", rp.likelihood["img"]),
            "world": wd.world_step,
        },
        tick_order="sequential",
        carries=None if start is None else {"world": {"energy": start}},
        record_carries=True,
    )
    rows = tick_rows(rec, "rover", ticks)
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


def agent_system(decoder, setting):
    """The two-tier agent, the merge of the scheduler ``mission`` and the rover:
    ``((System, SystemParams), grid)``. The option's action node drives the rover's option control
    node, the field is one parameter variable, and the scheduler's cell modality observes the
    rover's position belief through the cell indicator, realised by ``mode`` (the ``Link`` sugar:
    an emission ``rover.cell_obs`` of the rover, observed by ``mission.cell_obs``). The pinned run
    (``tiers == 1``) is the same agent, its scheduler stepped by a step that always chooses
    ``PINNED_OPTION``."""
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
        durations=True,  # the scheduler decides once per option, its duration read off these
        readings=setting.get("readings", False),
    )
    agent = merge(
        {"mission": ms, "rover": rs},
        {"mission": mp, "rover": rp},
        [
            Identify("mission.option", "rover.u_opt"),
            Identify("rover.field", "mission.field"),
            Link("rover.x", "mission.cell_obs", realise="mode", kernel=grid),
        ],
    )
    return agent, grid


def survey_system(decoder, setting):
    """The closed System of a two-tier run: the agent (``agent_system``) coupled to the world, the
    gauge feeding both tiers. Returns ``(System, SystemParams)``."""
    agent, _ = agent_system(decoder, setting)
    return couple(
        {"agent": agent, "world": wd.world_system()},
        [*world_bindings("agent.rover"), Identify("world.gauge", "agent.mission.gauge")],
    )


def timed(step, seconds):
    """``step`` with the wall time of each of its calls after the first (the reset) appended to
    ``seconds``; ``functools.wraps`` carries the marks ``run`` reads off a ``plan()`` step. The
    page's ``seconds`` of a decision is this time, the ``plan()`` call alone (inference, planning
    and the push of the prior); building the decision's record from the trace is not in it."""

    @functools.wraps(step)
    def wrapper(system, key, obs, carry, params):
        t_start = time.time()
        out = step(system, key, obs, carry, params)
        if obs is not None:
            seconds.append(time.time() - t_start)
        return out

    return wrapper


def pinned_step(name):
    """The pinned run's scheduler: a step that chooses ``PINNED_OPTION`` for the action node
    ``name`` every round it is called, and nothing at the reset."""
    level = jnp.asarray(PINNED_OPTION, jnp.int32)

    def step(system, key, obs, carry, params):
        return ({} if obs is None else {name: level}), carry, params

    return step


def decision_record(carry, option, reading, seconds, pre="agent.mission."):
    """The record of one decision of the scheduler for the page, read from the carry its
    ``plan(trace=True)`` step left: the option sequences it scored (``trace.policies``) with
    their term trees, ``G`` and ``q_pi``, the best ``G`` and the probability of each first option,
    and the posterior over the energy band and the visited cell (``carry.previous``). ``reading``
    is the cell modality's value that round, ``None`` where it was absent."""
    q_pi, tree, sequences = carry.trace.q_pi, carry.trace.tree, np.asarray(carry.trace.policies)
    g = total(tree)
    records = [
        dict(
            options=[int(a) for a in sequences[i].ravel()],
            terms=term_record(tree, i, pre.split(".", 1)[0] + "."),
            G=interval_pair(g, i),
            q_pi=float(q_pi.probs[i]),
        )
        for i in range(sequences.shape[0])
    ]
    midpoint = [float(np.mean(r["G"])) for r in records]
    first = [r["options"][0] for r in records]
    efe = [
        min(m for m, f in zip(midpoint, first, strict=True) if f == o) for o in range(sm.N_OPTIONS)
    ]
    q_option = [
        float(sum(r["q_pi"] for r in records if r["options"][0] == o)) for o in range(sm.N_OPTIONS)
    ]
    chosen = max((i for i, f in enumerate(first) if f == option), key=lambda i: records[i]["q_pi"])
    beliefs = carry.previous
    return dict(
        decision=True,
        option=option,
        cell=reading,
        efe_by_option=efe,
        q_by_option=q_option,
        policies=records,
        chosen=chosen,
        belief=dict(
            energy=[float(p) for p in np.asarray(beliefs[pre + "energy"].probs)],
            visit=[float(p) for p in np.asarray(beliefs[pre + "visit"].probs)],
        ),
        seconds=seconds,
    )


def run_tiered(setting, ticks, decoder, steps=PLANNER_STEPS, starts=PLANNER_STARTS):
    """A run of the two-tier agent coupled to the world: one ``run`` call over the closed System
    steps the scheduler (``plan()``, once per option, its trace recorded), the rover (every round)
    and the world (every round); with ``tiers == 1`` the scheduler always chooses
    ``PINNED_OPTION``."""
    system, params = survey_system(decoder, setting)
    two = setting["tiers"] == 2
    every = setting["slow_ticks"]
    seconds = []
    mission = (
        timed(
            plan(trace=True, selection="deterministic", replace={"policy_len": 2}),
            seconds,
        )
        if two
        else pinned_step("agent.mission.option")
    )
    rover = rover_step(setting, steps, starts, "agent.rover.", params.likelihood["agent.rover.img"])
    rec = run(
        system,
        params,
        ticks,
        jax.random.PRNGKey(setting["seed"]),
        steps={"agent.mission": mission, "agent.rover": rover, "world": wd.world_step},
        clocks={"agent.mission": every / ss.SURVEY_TICKS},
        tick_order="sequential",
        record_carries=True,
    )
    scheduler = rec["agent.mission"]
    options = scheduler.intended["option"]
    decided = [t for t in range(ticks) if bool(scheduler.decided[t])]
    decisions = {}
    if two:
        for t, secs in zip(decided, seconds, strict=True):
            reading = int(scheduler.received["cell_obs"][t])
            present = bool(scheduler.present["cell_obs"][t])
            option = int(options[t])
            carry = scheduler.carries[t + 1]
            decisions[t] = decision_record(carry, option, reading if present else None, secs)

    def extra(t):
        out = dict(option=int(options[t]))
        if two:
            out["slow"] = decisions.get(t, dict(decision=False))
        return out

    rows = tick_rows(rec, "agent.rover", ticks, extra)
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
    if any("tiers" in s for s in settings):
        lines += [
            "Two-tier runs are the merged agent coupled to the world; the scheduler's cell reading "
            "is the rover's posterior of the round before and is absent in round 0.",
            "",
        ]
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
    """Merge the part files into the bundle and write the report. The bundle's model records
    (``graph``, ``mdl``, ``description``) are of the model the page shows: at level 2 the agent's
    joint planner (the scheduler and the rover planned as one model), not the coupled System that
    ``run`` steps, which adds the world, whose hand-written step makes its tables placeholders that
    only type its ports; at level 1 the rover's System."""
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
        #  the page's graph, description length and document are of the agent's joint planner
        #  (each tier's option read as one decision), not of the coupled System that run steps:
        #  the page shows the agent's model, and the world's tables only type its ports
        model = joint_planner(*agent_system(decoder, settings[0])[0])
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
    ap.add_argument(
        "--readings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pool the scheduler's map reading over an option (default: on if available)",
    )
    ap.add_argument("--seed", type=int, default=None, help="the world's key (default: 0)")
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
        if args.seed is not None:
            s["seed"] = args.seed
        s["field_strength"] = args.field_strength
        s["planning_belief"] = args.planning_belief
        s["grid"] = args.grid
        if args.scheduler_logits:
            s["scheduler_logits"] = [float(v) for v in args.scheduler_logits.split(",")]
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
