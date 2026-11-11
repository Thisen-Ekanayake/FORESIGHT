#!/usr/bin/env python
"""Mirror-guard diagnostics from `run_pointnav_eval.py` foresight runs on annotated homes: is each confirmed wall on a
real mirror/glass object (`wall_log[].on_reflective`), split by what the check saw, plus encounter outcomes.

  python tools/guard_precision.py results/runs/pointnav/dev_guard_v22 [--map results/runs/pointnav/dev_guard_map]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--map", default=None, help="map-only runs on the same episodes, for crash comparison")
    args = p.parse_args()
    rows = [json.loads(f.read_text()) for f in Path(args.root).rglob("*.json") if f.name != "summary.json"]
    rows = [r for r in rows if r.get("condition") == "foresight" and not r.get("scan_hole")]
    walls = [dict(w, episode=r["episode"], stratum=r.get("stratum")) for r in rows for w in (r.get("wall_log") or [])]
    print(f"{len(rows)} runs, {len(walls)} confirmed walls, on a real mirror/glass: "
          f"{sum(w['on_reflective'] for w in walls)} ({100 * np.mean([w['on_reflective'] for w in walls] or [0]):.0f}%)")
    for name, key, cuts in (("slow-check P (Laya)", "p_laya", (0.6, 0.7, 0.8, 0.9, 0.95)),
                            ("mirror-box P (Qwen yes/no)", "p_box", (0.5, 0.7, 0.9, 0.97))):
        print(f"\n{name}: walls kept if >= cut -> kept, of them on reflective (precision), real walls lost")
        for c in cuts:
            k = [w for w in walls if w[key] >= c]
            tp = sum(w["on_reflective"] for w in k)
            lost = sum(w["on_reflective"] for w in walls) - tp
            print(f"  >= {c:4.2f}: kept {len(k):3d}, on reflective {tp:3d} ({100 * tp / max(len(k), 1):.0f}%), real lost {lost}")
    print("\nby candidate kind:")
    for kind in sorted({w["kind"] for w in walls}):
        k = [w for w in walls if w["kind"] == kind]
        print(f"  {kind:6s}: {len(k):3d} walls, on reflective {sum(w['on_reflective'] for w in k)}")
    print("\nby set (encounter vs other):")
    for st in sorted({str(w["stratum"]) for w in walls}):
        k = [w for w in walls if str(w["stratum"]) == st]
        print(f"  {st:9s}: {len(k):3d} walls, on reflective {sum(w['on_reflective'] for w in k)}")
    enc = [r for r in rows if r.get("stratum") == "encounter"]
    if enc:
        print(f"\nencounters ({len(enc)}): success {100 * np.mean([r['success'] for r in enc]):.0f}%, "
              f"crash {100 * np.mean([r['crash'] for r in enc]):.0f}%, with >= 1 wall on reflective: "
              f"{sum(any(w['on_reflective'] for w in (r.get('wall_log') or [])) for r in enc)}")
        if args.map:
            m = {json.loads(f.read_text())["episode"]: json.loads(f.read_text()) for f in Path(args.map).rglob("*.json")}
            pairs = [(m[r["episode"]], r) for r in enc if r["episode"] in m and not m[r["episode"]].get("scan_hole")]
            print(f"  map-only on the same encounters ({len(pairs)}): success {100 * np.mean([a['success'] for a, _ in pairs]):.0f}%, "
                  f"crash {100 * np.mean([a['crash'] for a, _ in pairs]):.0f}%")
            for a, b in pairs:
                print(f"    {b['episode']:34s} map {'OK' if a['success'] else ('CRASH' if a['crash'] else 'time')} | "
                      f"guard {'OK' if b['success'] else ('CRASH' if b['crash'] else 'time')} walls {len(b.get('wall_log') or [])} "
                      f"on-refl {sum(w['on_reflective'] for w in (b.get('wall_log') or []))} slow {b.get('slow_looks')}")


if __name__ == "__main__":
    main()
