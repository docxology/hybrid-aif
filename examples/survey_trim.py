"""Trim a survey bundle for the launch page.

    python examples/survey_trim.py IN.json OUT_DIR [--keep-every K] [--heavy-every H]
        [--play-stride P]

Writes OUT_DIR/bundle.js (``window.HYBRID_AIF_BUNDLE = ...``), the one data file the page loads.
Every float is rounded to four significant digits, each tick (and each slow step of a two-tier
run, under ``slow``) keeps at most ``MAX_POLICIES`` policies (the best by the midpoint of G,
always including the chosen one, in their original order with ``chosen`` re-indexed), and each
image (``img``, ``recon``) and the field's table (``map``) becomes one base64 string of its bytes,
in the layouts ``meta.image.layout`` and ``meta.map_layout`` describe.

For long runs (the large field's): ``--keep-every K`` keeps every K-th tick whole and drops the
others (a tick where the scheduler decides is always kept); ``--heavy-every H`` keeps the heavy
fields (``HEAVY``: the pictures, the plans and their records) on every H-th tick and drops them
from the other kept ticks, whose viewer shows those of the latest tick that has them; a map that
was not recorded at a tick (``map`` null) is dropped with its counts, and where it was recorded
its counts are folded into it: a cell with fewer than ``SEEN_SHOWN`` counts (which the viewer
does not draw) is written as uniform (which it does not draw either) and the counts are dropped;
the per-tick record of which cells are mapped (``mapped``) is dropped. ``meta.trim`` states
what was kept, and ``--play-stride P`` the ticks the viewer advances per step when it plays.
"""

import argparse
import base64
import json
from pathlib import Path

MAX_POLICIES = 12
PACKED = ("img", "recon", "map")
#  the fields a tick drops under ``--heavy-every`` when it is not a heavy tick
HEAVY = ("img", "recon", "camera", "recogniser", "policies", "chosen", "approximate", "offsets")
SEEN_SHOWN = 0.02


def rounded(obj):
    if isinstance(obj, float):
        return float(f"{obj:.4g}")
    if isinstance(obj, list):
        return [rounded(v) for v in obj]
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    return obj


def trim_policies(tick, limit=MAX_POLICIES, per_option=False):
    """Keep the ``limit`` best policies of ``tick`` (a rover tick or a slow step), the chosen one
    among them, in their original order with ``chosen`` re-indexed. With ``per_option`` (a slow
    step) the best sequence of every first option is kept as well."""
    policies = tick.get("policies")
    if not policies or len(policies) <= limit:
        return
    chosen = tick["chosen"]
    order = sorted(range(len(policies)), key=lambda i: sum(policies[i]["G"]) / 2)
    keep = {chosen}
    for i in order:
        if len(keep) >= limit:
            break
        keep.add(i)
    if per_option:  # the best sequence of each first option, for the slow tier's bars
        best = {}
        for i in order:
            best.setdefault(policies[i]["options"][0], i)
        keep |= set(best.values())
    kept = sorted(keep)
    tick["policies"] = [policies[i] for i in kept]
    tick["chosen"] = kept.index(chosen)


def pack(tick):
    for key in PACKED:
        rows = tick.get(key)
        if isinstance(rows, list):
            tick[key] = base64.b64encode(bytes(v for row in rows for v in row)).decode("ascii")


def thin(run, keep_every, heavy_every):
    """``run``'s ticks for a long replay (``--keep-every``, ``--heavy-every``), in place."""
    ticks = [
        t for t in run["ticks"] if t["t"] % keep_every == 0 or t.get("slow", {}).get("decision")
    ]
    for t in ticks:
        t.pop("mapped", None)
        if t.get("map") is None:
            t.pop("map", None)
            t.pop("counts", None)
        elif t.get("counts") is not None:
            for c, n in enumerate(t.pop("counts")):
                if n < SEEN_SHOWN:
                    t["map"][c] = [85, 85, 85]
        if heavy_every > 1 and t["t"] % heavy_every:
            for k in HEAVY:
                t.pop(k, None)
    run["ticks"] = ticks


def main(src, out, keep_every=1, heavy_every=1, play_stride=1):
    bundle = json.loads(Path(src).read_text())
    meta = bundle.setdefault("meta", {})
    long_run = keep_every > 1 or heavy_every > 1
    for run in bundle["runs"].values():
        if long_run:
            thin(run, keep_every, heavy_every)
        for tick in run["ticks"]:
            trim_policies(tick)
            if tick.get("slow", {}).get("decision"):
                trim_policies(tick["slow"], per_option=True)
            pack(tick)
    if long_run or play_stride > 1:
        meta["trim"] = dict(
            keep_every=keep_every,
            heavy_every=heavy_every,
            heavy_fields=list(HEAVY),
            play_stride=play_stride,
            note=(
                "every keep_every-th tick is kept (and every tick where the scheduler decides); "
                "the heavy fields are kept on every heavy_every-th tick, and a tick without them "
                "shows those of the latest tick that has them; the map is kept where it was "
                "recorded; the viewer advances play_stride ticks a step when it plays"
            ),
        )
    meta.setdefault("image", {})["encoding"] = "base64 of the pixel bytes"
    meta["map_encoding"] = "base64 of the cell bytes, three per cell"
    text = json.dumps(rounded(bundle), separators=(",", ":"), allow_nan=False)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bundle.js").write_text("window.HYBRID_AIF_BUNDLE = " + text + ";\n")
    print(f"{Path(src).stat().st_size / 1e6:.2f} MB -> {len(text) / 1e6:.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--keep-every", type=int, default=1)
    ap.add_argument("--heavy-every", type=int, default=1)
    ap.add_argument("--play-stride", type=int, default=1)
    a = ap.parse_args()
    main(a.src, a.out, a.keep_every, a.heavy_every, a.play_stride)
