"""Two-tier rover showcase: a slow mission mind over a fast rover mind, with a camera.

The mission tier (energy, three rocks, zone; visit, sample or charge) is held per option over
the fast rover tier (terrain-switched position and velocity, a position fix with a levelled noise
map, a terrain cue and a camera). The two System declarations are merged; each mind runs on its
own ``subsystem`` view, and the true world is a separate process wired in with ``couple``. A
setting is epistemic terms on or off, a steep or mild energy preference, a seed and a camera
reading (the engine inverts the neural decoder, or a frozen encoder's measurement replaces that
inversion). The runs of a grid are written to one bundle for the launch page.

    python examples/showcase_rover.py --out DIR [--settings id,id] [--ticks 120]
    python examples/showcase_rover.py --out DIR --assemble     # merge the part files into bundle
    python examples/showcase_trim.py DIR/bundle.json docs/site   # the page's bundle.js
"""

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
import showcase_camera as cam
import showcase_mission as mm
import showcase_rover_model as rm
import showcase_world as wd

from hybrid_aif import __version__
from hybrid_aif.composition import Wire, couple, hold, run
from hybrid_aif.composition.merge import ActionControl, merge, subsystem
from hybrid_aif.description import describe
from hybrid_aif.description.mdl import description_length
from hybrid_aif.objectives import total
from hybrid_aif.objectives.base import term_sign
from hybrid_aif.planning import mixed_plan
from hybrid_aif.planning.mixed import _face

PRECISION = 16.0  # the rover's position-target precision (1 / variance)
PLANNER_STEPS = 40
LIVE_WEIGHT = 1e-3
MDL_RESOLUTION = 100.0
NAMES = {
    "option": mm.OPTION_NAMES,
    "sensor": rm.SENSOR_NAMES,
    "terrain": rm.TERRAIN_NAMES,
    "energy": mm.ENERGY_NAMES,
    "rock": mm.ROCK_NAMES,
    "zone": mm.ZONE_NAMES,
}


def settings_grid():
    """Epistemic on/off x steep/mild x two seeds with the inverted camera, and one run with the
    encoder's measurement in place of the inversion."""
    out = []
    for epistemic, sharp, seed in itertools.product((True, False), ("steep", "mild"), (0, 1)):
        sid = f"{'epi' if epistemic else 'pragmatic'}-{sharp}-s{seed}"
        label = (
            f"Epistemic terms {'on' if epistemic else 'off'}, {sharp} energy preference, "
            f"seed {seed}"
        )
        out.append(
            dict(
                id=sid,
                label=label,
                epistemic=epistemic,
                sharpness=sharp,
                seed=seed,
                precision=PRECISION,
                camera="inverted",
            )
        )
    out.append(
        dict(
            id="epi-steep-s0-encoder",
            label="Epistemic terms on, steep energy preference, "
            "seed 0, encoder measurement replaces the image inversion",
            epistemic=True,
            sharpness="steep",
            seed=0,
            precision=PRECISION,
            camera="encoder",
        )
    )
    return out


def build(epistemic, sharp, camera, decoder, encoder, precision=PRECISION):
    """The merged composite, the mission view, and a function from an option level to the rover
    view pinned at that level (built once per level)."""
    ms, mp = mm.mission(sharp, epistemic)
    rs, rp = rm.rover(
        decoder, encoder if camera == "encoder" else None, precision=precision, epistemic=epistemic
    )
    cs, cp = merge(
        {"mission": ms, "rover": rs},
        {"mission": mp, "rover": rp},
        [ActionControl("mission.u_option", "rover.u_opt")],
    )
    mview, mparams = subsystem(cs, cp, "mission")
    mview = dataclasses.replace(mview, policy_len=2, planner="enumeration")
    cache = {}

    def rover_view(level):
        if level not in cache:
            view, vparams = subsystem(cs, cp, "rover", partner_priors={"mission.u_option": level})
            view = dataclasses.replace(view, planner="mixed", num_iters=PLANNER_STEPS, policy_len=3)
            cache[level] = (view, vparams)
        return cache[level]

    return cs, cp, (mview, mparams), rover_view


def interval_pair(interval, i):
    return [float(interval.lower[i]), float(interval.upper[i])]


def strip(name):
    return name.split(".", 1)[1] if "." in name else name


def term_record(tree, i):
    return {
        term: {strip(leaf): interval_pair(term_sign(term) * iv, i) for leaf, iv in leaves.items()}
        for term, leaves in tree.items()
    }


def approximate_record(tree):
    """term -> leaf -> whether the planner reports the leaf as a linearised or quadrature read."""
    return {
        term: {strip(leaf): bool(iv.approximate) for leaf, iv in leaves.items()}
        for term, leaves in tree.items()
    }


def position_belief(beliefs):
    x = beliefs["rover.x"]
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
        terrain=[float(p) for p in np.asarray(beliefs["rover.terrain"].probs)],
        velocity=[float(mean[2]), float(mean[3])],
    )


def image_rows(flat):
    """A flat pixel vector (x-major, as the renderer lays it out) as uint8 rows of the picture:
    row 0 is the largest y, columns run along x."""
    grid = np.asarray(flat).reshape(wd.IMG_SIDE, wd.IMG_SIDE)  # [x cell, y cell]
    return [[int(v) for v in row] for row in wd.to_uint8(grid.T[::-1])]


def camera_record(detail, covered):
    """The camera's contribution to the tick: the engine's log evidence of the image (inverted),
    or the encoder's normaliser and the covered modality's expected cost (encoder)."""
    if covered:
        return dict(
            normaliser=float(detail.recognition_log_evidence["recognition:rover.x"]),
            covered_term=float(detail.covered_terms["rover.img"]),
        )
    return dict(log_evidence=float(detail.log_evidence["rover.img"]))


def make_rover_mind(rover_view, holder, log, decoder, encoder, camera):
    """Infer on the view pinned at the mission's current option, plan a mixed policy, act."""
    captured = {}
    last = {"sensor": 0}  # the sensor level in force before the first round

    def recording_planner(ctx, use_bound="midpoint", metrics=True):
        result = mixed_plan(ctx, steps=PLANNER_STEPS, use_bound=use_bound, metrics=metrics)
        captured["result"] = result
        return _face(result, ctx)

    recording_planner.needs_key = False

    def respond(key, obs, carry):
        del key
        t_start = time.time()
        view, vparams = rover_view(holder["option"])
        previous = {"rover.a_sensor": last["sensor"]}
        beliefs, detail = view.infer_states(
            vparams, obs, priors=carry.beliefs, previous=previous, detail=True
        )
        q_pi, tree = view.infer_policies(vparams, beliefs, planner=recording_planner, metrics=True)
        result = captured["result"]
        best = int(jnp.argmax(q_pi.probs))
        action = {
            "rover.a_v": result.plan[best, 0],
            "rover.a_sensor": result.discrete[best, 0, 0].astype(jnp.int32),
        }
        last["sensor"] = int(action["rover.a_sensor"])
        pushed = view.update_empirical_prior(vparams, beliefs, action)
        mean = np.asarray(beliefs["rover.x"].mean())
        terrain = int(jnp.argmax(beliefs["rover.terrain"].probs))
        recon, _ = decoder.diag_moments((jnp.asarray(mean), jax.nn.one_hot(terrain, 3)))
        enc_mean, enc_scale = encoder.diag_moments((jnp.asarray(obs["rover.img"]),))
        g = total(tree)
        policies = [
            dict(
                id=i,
                start=[float(mean[0]), float(mean[1])],
                sensor_plan=[int(s) for s in result.discrete[i, :, 0]],
                v_plan=[[float(c) for c in step] for step in result.plan[i]],
                terms=term_record(tree, i),
                G=interval_pair(g, i),
                q_pi=float(q_pi.probs[i]),
            )
            for i in range(result.discrete.shape[0])
        ]
        log.append(
            dict(
                seconds=time.time() - t_start,
                belief=position_belief(beliefs),
                policies=policies,
                chosen=best,
                approximate=approximate_record(tree),
                camera=camera_record(detail, camera == "encoder"),
                free_energy=float(detail.free_energy),
                recon=image_rows(recon),
                encoder=dict(
                    mean=[float(enc_mean[0]), float(enc_mean[1])],
                    sd=[float(enc_scale[0]), float(enc_scale[1])],
                ),
            )
        )
        return action, carry._replace(beliefs=pushed)

    return respond


def collapse(probs):
    p = np.asarray(probs)
    out = np.zeros(3)
    for lvl, c in enumerate(mm.ROCK_COLLAPSE):
        out[c] += p[lvl]
    return [float(v) for v in out]


def make_mission_mind(mview, mparams, holder, log):
    """The mission's inference and planning, held for each option's duration."""
    state = dict(decided=False)

    def inner(key, obs, carry):
        beliefs, _ = mview.infer_states(mparams, obs, priors=carry.beliefs)
        action, q_pi, tree = mview.act(
            key, mparams, beliefs, metrics=True, selection="deterministic"
        )
        pushed = mview.update_empirical_prior(mparams, beliefs, action)
        policies = np.asarray(mview.policies(mparams))
        g = total(tree)
        records = [
            dict(
                options=[int(a) for a in policies[i].ravel()],
                terms=term_record(tree, i),
                G=interval_pair(g, i),
                q_pi=float(q_pi.probs[i]),
            )
            for i in range(policies.shape[0])
        ]
        state["decided"] = True
        state["belief"] = dict(
            energy=[float(p) for p in np.asarray(beliefs["mission.energy"].probs)],
            rocks=[collapse(beliefs[f"mission.rock_{k}"].probs) for k in (1, 2, 3)],
        )
        state["policies"] = records
        state["chosen"] = int(np.argmax(np.asarray(q_pi.probs)))
        return action, carry._replace(beliefs=pushed)

    held = hold(mview, mparams, inner)

    def respond(key, obs, carry):
        state["decided"] = False
        action, carry = held(key, obs, carry)
        holder["option"] = int(action["mission.option"])
        entry = dict(decision=state["decided"], belief=state.get("belief"))
        if state["decided"]:
            entry.update(policies=state["policies"], chosen=state["chosen"])
        log.append(entry)
        return action, carry

    return respond


def arrival_summary(rows):
    """Per option (decision to the next decision) whether the rover ended in the target zone."""
    starts = [i for i, r in enumerate(rows) if "slow_policies" in r]
    arrived = []
    if not starts:
        return dict(options=0, arrived=0, rate=None)
    for a, b in zip(starts, [*starts[1:], len(rows)], strict=True):
        if b - a < 2:
            continue
        end = rows[b - 1]
        option = rows[a]["option"]
        arrived.append(end["slow"]["zone"] == mm.OPTION_ZONE[option])
    return dict(
        options=len(arrived),
        arrived=int(sum(arrived)),
        rate=float(np.mean(arrived)) if arrived else None,
    )


def run_setting(setting, ticks, decoder, encoder):
    cs, cp, (mview, mparams), rover_view = build(
        setting["epistemic"],
        setting["sharpness"],
        setting["camera"],
        decoder,
        encoder,
        setting.get("precision", PRECISION),
    )
    holder, rover_log, slow_log = {"option": 0}, [], []
    mission_mind = make_mission_mind(mview, mparams, holder, slow_log)
    rover_mind = make_rover_mind(rover_view, holder, rover_log, decoder, encoder, setting["camera"])
    port_view, port_params = rover_view(0)
    env, env_params = wd.world()
    wiring = couple(
        minds={
            "slow": (mview, mparams, mission_mind),
            "fast": (port_view, port_params, rover_mind),
        },
        processes={"world": (env, env_params)},
        wires=[
            Wire("slow.mission.option", "world.option"),
            Wire("fast.rover.a_v", "world.a_v"),
            Wire("fast.rover.a_sensor", "world.a_sensor"),
            *[Wire(f"world.{m}", f"slow.mission.{m}") for m in mm.MISSION_MODALITIES],
            Wire("world.fix", "fast.rover.fix"),
            Wire("world.cue", "fast.rover.cue"),
            Wire("world.img", "fast.rover.img"),
        ],
    )
    rec = run(wiring, ticks=ticks, key=jax.random.PRNGKey(setting["seed"]))
    world, fast, slow = rec["world"], rec["fast"], rec["slow"]
    st = world.states
    rows = []
    last_belief = None
    for t in range(ticks):
        sl = slow_log[t]
        rl = rover_log[t]
        last_belief = sl["belief"] or last_belief
        rocks_true = [min(int(st[f"rock_{k}"][t]), 2) for k in (1, 2, 3)]
        zone = int(wd.zone_of(jnp.asarray(st["x"][t][:2])))
        row = dict(
            t=t,
            option=int(slow.actions["mission.option"][t]),
            sensor=int(fast.actions["rover.a_sensor"][t]),
            v=[float(c) for c in fast.actions["rover.a_v"][t]],
            true_x=[float(c) for c in st["x"][t]],
            true_terrain=int(st["terrain"][t]),
            belief=rl["belief"],
            fix=[float(c) for c in fast.observations["rover.fix"][t]],
            cue=int(fast.observations["rover.cue"][t]),
            img=image_rows(fast.observations["rover.img"][t]),
            recon=rl["recon"],
            encoder=rl["encoder"],
            camera=rl["camera"],
            free_energy=rl["free_energy"],
            seconds=rl["seconds"],
            policies=rl["policies"],
            chosen=rl["chosen"],
            approximate=rl["approximate"],
            slow=dict(
                energy_true=int(wd.band_of(st["energy"][t])),
                energy_belief=last_belief["energy"],
                rocks_true=rocks_true,
                rocks_belief=last_belief["rocks"],
                zone=zone,
                decision=bool(sl["decision"]),
            ),
        )
        if sl["decision"]:
            row["slow_policies"] = sl["policies"]
            row["slow_chosen"] = sl["chosen"]
        rows.append(row)
    return dict(ticks=rows, summary=arrival_summary(rows))


def graph_record(cs, cp):
    names = cs.names
    all_nodes = (*cs.factor_names, *cs.action_names)  # a modality may read an action node
    tiers = {}

    def tier(n):
        return tiers.setdefault(n.split(".")[0], dict(factors=[], modalities=[], controls=[]))

    for f in names.states:
        d = cp.initial_state_prior[f]
        if hasattr(d, "logits"):
            rec = dict(name=f, family="categorical", levels=int(d.logits.shape[-1]))
        else:
            rec = dict(name=f, family="gaussian_mixture", dim=int(d.means.shape[-1]))
        tier(f)["factors"].append(rec)
    for m, parents in zip(names.modalities, cs.A_deps, strict=True):
        lik = type(cp.likelihood[m]).__name__
        families = {
            "LinearGaussianObservation": "linear_gaussian",
            "MLPConditional": "neural",
            "ConditionalCategorical": "categorical",
        }
        if lik not in families:
            raise ValueError(f"no graph family for the likelihood of {m!r}: {lik}")
        family = families[lik]
        tier(m)["modalities"].append(
            dict(name=m, family=family, parents=[all_nodes[p] for p in parents])
        )
    levels = cs._control_levels(cp)
    durations = {k: [float(x) for x in np.asarray(v)] for k, v in (cp.durations or {}).items()}
    for node, n_levels in zip(names.controls, levels, strict=True):
        if node in cp.control:
            box = cp.control[node]
            rec = dict(
                name=node,
                kind="float",
                box=[[float(b) for b in box.low], [float(b) for b in box.high]],
                durations=None,
            )
        else:
            rec = dict(name=node, kind="level", levels=int(n_levels), durations=durations.get(node))
        tier(node)["controls"].append(rec)
    edges = []
    for child, parents in zip(cs.factor_names, cs.B_deps, strict=True):
        if child in names.controls:
            continue
        for p in parents:
            src = all_nodes[p]
            if src == child:
                continue
            edges.append({"from": src, "to": child, "kind": "transition"})
    for m, parents in zip(names.modalities, cs.A_deps, strict=True):
        for p in parents:
            edges.append({"from": all_nodes[p], "to": m, "kind": "likelihood"})
    for key, parents in cs.C_deps:
        for p in parents:
            edges.append({"from": p, "to": key, "kind": "target"})
    return dict(tiers=[dict(name=n, **v) for n, v in tiers.items()], edges=edges)


def mdl_record(cs, cp):
    document, arrays = describe(cs, cp)
    dl = description_length(document, MDL_RESOLUTION, arrays=arrays)
    unscored = {a for a, link in dl.unscored if link == "family"}
    rows = []
    for address, links in dl.per_address.items():
        kind, _, name = address.partition(":")
        tier_name = name.split(".")[0] if "." in name else "model"
        nats = None if address in unscored else float(links.structure + links.parameters)
        rows.append(
            dict(name=address, tier=tier_name, kind=kind, nats=nats, exact=bool(links.exact))
        )
    scored = [r for r in rows if r["nats"] is not None]
    return dict(
        total=float(sum(r["nats"] for r in scored)),
        total_exact=float(sum(r["nats"] for r in scored if r["exact"])),
        library_total=float(dl.total),
        components=rows,
    )


def dynamics_record():
    """The mean position and velocity dynamics the page rolls a policy under: per sensor level
    the velocity persistence and command gain, and per terrain the velocity drift."""
    drift = []
    for terrain in range(len(rm.TERRAIN_NAMES)):
        _, _, loc, _ = wd.dynamics(terrain, 0)
        drift.append([float(loc[2]), float(loc[3])])
    return dict(
        dt=float(wd.DT),
        keep=[float(k) for k in wd.KEEP],
        gain=[float(g) for g in wd.GAIN],
        drift=drift,
    )


def meta_record(ticks, camera_report):
    return dict(
        library_version=__version__,
        ticks=ticks,
        dynamics=dynamics_record(),
        zones={z: [float(p[0]), float(p[1])] for z, p in mm.ZONES.items()},
        beacon=list(wd.RIDGE),
        field=[list(wd.FIELD[0]), list(wd.FIELD[1])],
        names=NAMES,
        image=dict(
            side=wd.IMG_SIDE,
            layout="rows of uint8 0..255, row 0 at the largest y, columns along x, over the field",
            field=[list(wd.FIELD[0]), list(wd.FIELD[1])],
            camera_report=camera_report,
        ),
        sign_convention=(
            "every leaf is added to G with its registered sign already applied, so the leaves of "
            "a policy sum to its G and an epistemic leaf appears negative"
        ),
    )


def assemble(out_dir, camera_report, decoder, encoder):
    """Merge the part files into the bundle; each part carries its setting and its runs."""
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
    settings = [settings[sid] for sid in runs]
    cs, cp, _, _ = build(True, "steep", "inverted", decoder, encoder)
    graph = graph_record(cs, cp)
    graph["recognition"] = [  # declared in the encoder setting only
        dict(variable=f"rover.{e.variable}", parents=[p.replace(":", ":rover.") for p in e.parents])
        for e in rm.rover(decoder, encoder)[0].recognition
    ]
    bundle = dict(
        meta=meta_record(ticks.pop(), camera_report),
        graph=graph,
        settings=settings,
        runs=runs,
        mdl=mdl_record(cs, cp),
    )
    text = json.dumps(bundle)
    (out_dir / "bundle.json").write_text(text)
    (out_dir / "bundle.js").write_text(f"window.HYBRID_AIF_BUNDLE = {text};\n")
    print(f"bundle: {len(runs)} runs, {len(text) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--out", required=True, help="output directory (also holds the camera cache)")
    ap.add_argument("--settings", default=None, help="comma-separated setting ids")
    ap.add_argument("--assemble", action="store_true", help="merge the part files and stop")
    ap.add_argument("--precision", type=float, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    decoder, encoder, camera_report = cam.load_or_train(out_dir)
    if args.assemble:
        assemble(out_dir, camera_report, decoder, encoder)
        return
    settings = settings_grid()
    if args.settings:
        keep = set(args.settings.split(","))
        settings = [s for s in settings if s["id"] in keep]
    for s in settings:
        if args.precision is not None:
            s["precision"] = args.precision
        t0 = time.time()
        result = run_setting(s, args.ticks, decoder, encoder)
        elapsed = time.time() - t0
        result["seconds"] = elapsed
        part = out_dir / f"part_{s['id']}{args.tag}.json"
        part.write_text(json.dumps(dict(runs={s["id"]: result}, settings={s["id"]: s})))
        secs = sorted(r["seconds"] for r in result["ticks"])
        print(
            f"{s['id']}: {elapsed:.1f}s ({elapsed / args.ticks:.2f}s/tick; median rover tick "
            f"{secs[len(secs) // 2]:.2f}s, max {secs[-1]:.2f}s) arrival {result['summary']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
