"""Closed-loop runs of the survey rover: a rover that maps a terrain field by driving over it
while keeping its energy up.

The rover (``survey_model``) infers its state, plans a velocity command by gradient descent from
several warm starts, acts, and learns the terrain field by counting each terrain it sees into the
cells it was in. The true world (``survey_world``) is a separate process wired in with ``couple``.
A setting is the epistemic terms (the field's novelty, the position fix's information gain, both
or neither) and whether the image preference is declared. The runs of a grid are written to one
bundle for the launch page, and a REPORT.md summarises them.

    python examples/survey_run.py --out DIR [--settings id,id] [--ticks 200]
    python examples/survey_run.py --out DIR --assemble     # merge the part files into bundle
    python examples/survey_trim.py DIR/bundle.json OUT_DIR   # the page's bundle.js
"""

import argparse
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
import survey_world as wd

from hybrid_aif import __version__
from hybrid_aif.composition import Wire, couple, run
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


def position_belief(beliefs):
    x = beliefs["x"]
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
        terrain=[float(p) for p in np.asarray(beliefs["terrain"].probs)],
        energy=[float(p) for p in np.asarray(beliefs["energy"].probs)],
        velocity=[float(mean[2]), float(mean[3])],
        mean=[float(mean[0]), float(mean[1])],
    )


def image_rows(flat):
    """A flat pixel vector (row-major, as the renderer lays it out) as uint8 rows of the picture:
    row 0 is the top of the picture, columns run left to right."""
    grid = np.asarray(flat).reshape(wd.IMG_SIDE, wd.IMG_SIDE)
    return [[int(v) for v in row] for row in wd.to_uint8(grid)]


def field_record(params, strength):
    """The field's posterior mean per cell as uint8 probabilities ``(cells, 3)``, the counts it
    has absorbed per cell beyond its prior, the fraction of cells whose top probability exceeds
    ``COVERED_MAX`` and the fraction that has absorbed ``SEEN_COUNTS`` counts."""
    conc = np.asarray(params.transition["terrain"].concentration, np.float64)
    total_conc = conc.sum(axis=0)
    mean = conc / total_conc
    rows = np.asarray(np.round(mean.T * 255), np.uint8)
    return (
        [[int(v) for v in row] for row in rows],
        [float(c) for c in total_conc - strength],
        float(np.mean(mean.max(axis=0) > COVERED_MAX)),
        float(np.mean(total_conc - strength >= SEEN_COUNTS)),
    )


def make_mind(system, params0, log, setting, steps, starts):
    """The rover's response: infer, plan, act, push the prior and learn the field."""
    compiled = system.compiled()
    planner = make_gradient_planner(steps=steps, starts=starts)
    state = dict(params=params0, prev=None)
    decoder = params0.likelihood["img"]

    def respond(key, obs, carry):
        del key
        t_start = time.time()
        params = state["params"]
        beliefs, detail = compiled.infer_states(params, obs, priors=carry.beliefs, detail=True)
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
        q_pi, tree, policies = compiled.infer_policies(
            params, beliefs, planner=planner, metrics=True, return_policies=True
        )
        v_plans = policies  # (1, policy_len, 2): the gradient planner returns its best plan
        best = int(jnp.argmax(q_pi.probs))
        action = {"a_v": v_plans[best, 0]}
        pushed = compiled.update_empirical_prior(params, beliefs, action)
        state["prev"] = (beliefs, action)
        mean = np.asarray(beliefs["x"].mean())
        terrain = int(jnp.argmax(beliefs["terrain"].probs))
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
        table, counts, coverage, seen = field_record(params, setting["field_strength"])
        log.append(
            dict(
                seconds=time.time() - t_start,
                belief=position_belief(beliefs),
                policies=records,
                chosen=best,
                approximate=approximate_record(tree),
                camera=dict(log_evidence=float(detail.log_evidence["img"])),
                free_energy=float(detail.free_energy),
                recon=image_rows(recon),
                map=table,
                counts=counts,
                coverage=coverage,
                seen=seen,
            )
        )
        return action, carry._replace(beliefs=pushed)

    return respond


def run_setting(setting, ticks, decoder, steps=PLANNER_STEPS, starts=PLANNER_STARTS):
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
    world, rover = rec["world"], rec["rover"]
    st = world.states
    rows = []
    for t in range(ticks):
        rl = log[t]
        energy = float(st["energy"][t])
        rows.append(
            dict(
                t=t,
                v=[float(c) for c in rover.actions["a_v"][t]],
                true_x=[float(c) for c in st["x"][t]],
                true_terrain=int(st["terrain"][t]),
                true_energy=dict(band=int(wd.band_of(jnp.asarray(energy))), level=energy),
                stalled=bool(wd.band_of(jnp.asarray(energy)) == 0),
                fix=[float(c) for c in rover.observations["fix"][t]],
                gauge=int(rover.observations["gauge"][t]),
                img=image_rows(rover.observations["img"][t]),
                **rl,
            )
        )
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


def meta_record(ticks, camera_report, grid):
    return dict(
        library_version=__version__,
        ticks=ticks,
        grid=dict(shape=list(grid.shape), low=list(grid.low), high=list(grid.high)),
        field=[list(wd.FIELD[0]), list(wd.FIELD[1])],
        start=list(wd.START),
        world=downsampled_maps(),
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


def run_summary(run_):
    """The figures REPORT.md gives per setting."""
    rows = run_["ticks"]
    n = len(rows)
    err = [
        float(np.hypot(*(np.asarray(r["belief"]["mean"]) - np.asarray(r["true_x"][:2]))))
        for r in rows
    ]
    bands = np.bincount([r["true_energy"]["band"] for r in rows], minlength=4)
    terrain = np.bincount([r["true_terrain"] for r in rows], minlength=3)
    secs = [r["seconds"] for r in rows[1:]] or [rows[0]["seconds"]]
    coverage = {f"t{k}": rows[min(k, n) - 1]["coverage"] for k in (50, 100, 200)}
    return dict(
        coverage=coverage,
        seen={f"t{k}": rows[min(k, n) - 1]["seen"] for k in (50, 100, 200)},
        position_error=float(np.mean(err)),
        ticks_per_band=[int(b) for b in bands],
        ticks_on_ice=int(terrain[2]),
        ticks_on_rough=int(terrain[1]),
        final_band=int(rows[-1]["true_energy"]["band"]),
        mean_seconds=float(np.mean(secs)),
        first_velocities=[[round(c, 3) for c in r["v"]] for r in rows[:10]],
    )


def report_text(settings, runs):
    on = ", ".join(s["id"] for s in settings if s["image_preference"]) or "none"
    off = ", ".join(s["id"] for s in settings if not s["image_preference"]) or "none"
    lines = ["# Survey rover runs", ""]
    lines += [
        f"Image preference on: {on}; off: {off}.",
        "",
    ]
    for s in settings:
        r = run_summary(runs[s["id"]])
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
            f"- final energy band: {sm.ENERGY_NAMES[r['final_band']]}",
            f"- mean seconds per tick after the first: {r['mean_seconds']:.2f}",
            f"- first ten chosen velocities: {r['first_velocities']}",
            "",
        ]
    return "\n".join(lines)


def graph_record(system, params):
    """The model's graph for the page: one tier of factors, modalities and controls, and the
    transition, likelihood and target edges."""
    names = system.names
    all_nodes = (*system.factor_names, *system.action_names)
    tier = dict(name="rover", factors=[], modalities=[], controls=[])
    for f in names.states:
        d = params.initial_state_prior[f]
        if hasattr(d, "logits"):
            rec = dict(name=f, family="categorical", levels=int(d.logits.shape[-1]))
        else:
            rec = dict(name=f, family="gaussian_mixture", dim=int(d.means.shape[-1]))
        if f in dict(system.cells):
            grid = dict(system.cells)[f]
            rec["cells"] = dict(shape=list(grid.shape), axes=list(grid.axes))
        tier["factors"].append(rec)
    families = {
        "LinearGaussianObservation": "linear_gaussian",
        "MLPConditional": "neural",
        "ConditionalCategorical": "categorical",
    }
    for m, parents in zip(names.modalities, system.A_deps, strict=True):
        lik = type(params.likelihood[m]).__name__
        if lik not in families:
            raise ValueError(f"no graph family for the likelihood of {m!r}: {lik}")
        tier["modalities"].append(
            dict(name=m, family=families[lik], parents=[all_nodes[p] for p in parents])
        )
    for node in names.controls:
        box = params.control[node]
        tier["controls"].append(
            dict(
                name=node,
                kind="float",
                box=[[float(b) for b in box.low], [float(b) for b in box.high]],
                durations=None,
            )
        )
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
    return dict(tiers=[tier], edges=edges)


def _entry_size(params, kind, name):
    table = {
        "likelihood": params.likelihood,
        "transition": params.transition,
        "initial_state_prior": params.initial_state_prior,
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
    order = [s["id"] for s in settings_grid() if s["id"] in runs]
    settings = [settings[sid] for sid in order]
    runs = {sid: runs[sid] for sid in order}
    n_grid = settings[0].get("grid", sm.GRID_SHAPE)
    grid, _ = sm.field_grid(n_grid)
    system, params, _, _ = sm.rover(decoder, image_preference=True, grid_shape=n_grid)
    bundle = dict(
        meta=meta_record(ticks.pop(), camera_report, grid),
        graph=graph_record(system, params),
        settings=settings,
        runs=runs,
    )
    try:
        bundle["mdl"] = mdl_record(system, params)
    except Exception as exc:  # the description refuses some models; say so and carry on
        print(f"mdl skipped: {type(exc).__name__}: {exc}")
    text = json.dumps(bundle)
    (out_dir / "bundle.json").write_text(text)
    (out_dir / "REPORT.md").write_text(report_text(settings, runs))
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
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    decoder, _, camera_report = cam.load_or_train(out_dir)
    if args.assemble:
        assemble(out_dir, camera_report, decoder)
        return
    settings = settings_grid()
    if args.settings:
        keep = set(args.settings.split(","))
        settings = [s for s in settings if s["id"] in keep]
    for s in settings:
        s["field_lr"] = args.field_lr
        s["field_strength"] = args.field_strength
        s["planning_belief"] = args.planning_belief
        s["grid"] = args.grid
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
