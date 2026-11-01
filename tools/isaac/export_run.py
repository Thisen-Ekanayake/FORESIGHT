#!/usr/bin/env python
"""Export a recorded object-search run (a `TimedEpisode` pickle from tools/run_object_search.py) to plain JSON for
the Isaac Sim replay (`tools/isaac/render_replay.py`), whose Python can't import habitat or foresight.

Keeps, per tick: sim time, position (Habitat frame: y up, forward = -z), heading, mode, target label, the planned
path; and once: the walls FORESIGHT drew, confirmed objects, "found" events, the mirror-check events.

  python tools/isaac/export_run.py results/runs/object_search/fresh/mirror-sofa-bed/foresight_fs.pkl
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_isaac.json")
    with open(src, "rb") as f:
        ep = pickle.load(f)
    s = ep.steps
    ticks = []
    for k, x in enumerate(s):
        p = ep.positions[k + 1]
        ticks.append({"t": round(float(x["t"] + x["dt"]), 3), "pos": [round(float(v), 4) for v in p],
                      "fwd": [round(float(v), 4) for v in x["forward_xz"]], "mode": x.get("mode"),
                      "target": x.get("target_label"), "safety": x.get("safety_level"),
                      "verifying": bool(x.get("verifying", False)), "event": x.get("event")})
    last = s[-1]
    data = {
        "start": {"t": 0.0, "pos": [float(v) for v in ep.positions[0]]},
        "ticks": ticks,
        "walls": last.get("map_walls", []),
        "memory": [o for o in last.get("memory", []) if o["hits"] >= 3 and not o["rejected"]],
        "found": last.get("found", []),
        "search_events": last.get("search_events", []),
        "crash_at": ep.contact_at,
    }
    out.write_text(json.dumps(data))
    print(f"{len(ticks)} ticks, {ticks[-1]['t']:.1f} s -> {out}")


if __name__ == "__main__":
    main()
