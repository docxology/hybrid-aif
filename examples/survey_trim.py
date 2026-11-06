"""Trim a survey bundle for the launch page.

    python examples/survey_trim.py IN.json OUT_DIR

Writes OUT_DIR/bundle.js (``window.HYBRID_AIF_BUNDLE = ...``), the one data file the page loads.
Every float is rounded to four significant digits, each tick (and each slow step of a two-tier
run, under ``slow``) keeps at most ``MAX_POLICIES`` policies (the best by the midpoint of G,
always including the chosen one, in their original order with ``chosen`` re-indexed), and each
image (``img``, ``recon``) and the field's table (``map``) becomes one base64 string of its bytes,
in the layouts ``meta.image.layout`` and ``meta.map_layout`` describe.
"""

import base64
import json
import sys
from pathlib import Path

MAX_POLICIES = 12
PACKED = ("img", "recon", "map")


def rounded(obj):
    if isinstance(obj, float):
        return float(f"{obj:.4g}")
    if isinstance(obj, list):
        return [rounded(v) for v in obj]
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    return obj


def trim_policies(tick, limit=MAX_POLICIES):
    """Keep the ``limit`` best policies of ``tick`` (a rover tick or a slow step), the chosen one
    among them, in their original order with ``chosen`` re-indexed."""
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
    kept = sorted(keep)
    tick["policies"] = [policies[i] for i in kept]
    tick["chosen"] = kept.index(chosen)


def pack(tick):
    for key in PACKED:
        rows = tick.get(key)
        if isinstance(rows, list):
            tick[key] = base64.b64encode(bytes(v for row in rows for v in row)).decode("ascii")


def main(src, out):
    bundle = json.loads(Path(src).read_text())
    for run in bundle["runs"].values():
        for tick in run["ticks"]:
            trim_policies(tick)
            if tick.get("slow", {}).get("decision"):
                trim_policies(tick["slow"])
            pack(tick)
    meta = bundle.setdefault("meta", {})
    meta.setdefault("image", {})["encoding"] = "base64 of the pixel bytes"
    meta["map_encoding"] = "base64 of the cell bytes, three per cell"
    text = json.dumps(rounded(bundle), separators=(",", ":"), allow_nan=False)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bundle.js").write_text("window.HYBRID_AIF_BUNDLE = " + text + ";\n")
    print(f"{Path(src).stat().st_size / 1e6:.2f} MB -> {len(text) / 1e6:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
