#!/usr/bin/env python
"""Tables from `tools/run_pointnav_eval.py` outputs: one row per (condition, depth source), 95% bootstrap intervals,
and optional paired comparisons against a reference cell on the episodes both ran.

  python tools/summarize_pointnav.py results/runs/pointnav/pilot
  python tools/summarize_pointnav.py results/runs/pointnav/A --ref map:gt --by stratum
Writes <dir>/summary.json and prints a Markdown table.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foresight.eval.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff  # noqa: E402


def load(root: Path):
    rows = []
    for f in root.rglob("*.json"):
        if f.name == "summary.json":
            continue
        r = json.loads(f.read_text())
        if "episode" in r:
            r["cell"] = (r["condition"] + ("" if r.get("floor_scale", True) else "_nofs")
                         + ("_robust" if r.get("robust_scan") else "") + ("_v21" if r.get("guard") == "v21" else "")
                         + ":" + r["depth"])
            rows.append(r)
    return rows


def fmt(m, lo, hi, pct=True):
    if not np.isfinite(m):
        return "–"
    k = 100 if pct else 1
    return f"{m * k:.0f} [{lo * k:.0f}, {hi * k:.0f}]" if pct else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def table(rows, ref=None, title=""):
    cells = defaultdict(list)
    for r in rows:
        cells[r["cell"]].append(r)
    out, lines = {}, []
    lines.append(f"\n{title}\n| cell | n | success % | SPL | crash % | timeout % | progress | depth AbsRel | depth RMSE m |"
                 + (" vs ref: success (McNemar p) | SPL diff [95%] |" if ref else ""))
    lines.append("|---|---|---|---|---|---|---|---|---|" + ("---|---|" if ref else ""))
    refrows = {r["episode"]: r for r in cells.get(ref, [])} if ref else {}
    for cell in sorted(cells):
        rs = cells[cell]
        s = bootstrap_ci([r["success"] for r in rs])
        spl = bootstrap_ci([r["spl"] for r in rs])
        cr = bootstrap_ci([r["crash"] for r in rs])
        to = bootstrap_ci([r["timeout"] for r in rs])
        pr = bootstrap_ci([r["progress"] for r in rs])
        ab = np.nanmean([r["err_absrel"] for r in rs if r["err_absrel"] is not None] or [np.nan])
        rm = np.nanmean([r["err_rmse"] for r in rs if r["err_rmse"] is not None] or [np.nan])
        o = {"n": len(rs), "success": s, "spl": spl, "crash": cr, "timeout": to, "progress": pr, "absrel": ab, "rmse": rm}
        extra = ""
        if ref and cell != ref and refrows:
            common = [r for r in rs if r["episode"] in refrows]
            if common:
                a = [refrows[r["episode"]]["success"] for r in common]
                b = [r["success"] for r in common]
                n10, n01, p = mcnemar_exact(a, b)
                d = paired_bootstrap_diff([refrows[r["episode"]]["spl"] for r in common], [r["spl"] for r in common])
                o["vs_ref"] = {"n": len(common), "ref_only": n10, "cell_only": n01, "mcnemar_p": p, "spl_diff": d}
                extra = f" −{n10}/+{n01} (p={p:.3f}) | {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] |"
        elif ref:
            extra = " (ref) | |"
        out[cell] = o
        lines.append(f"| {cell} | {len(rs)} | {fmt(*s)} | {fmt(*spl, pct=False)} | {fmt(*cr)} | {fmt(*to)} | "
                     f"{fmt(*pr, pct=False)} | {ab:.2f} | {rm:.2f} |{extra}")
    print("\n".join(lines))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--ref", default=None, help="reference cell for paired tests, e.g. map:gt")
    p.add_argument("--by", default=None, help="also split by this record field (e.g. stratum, scene)")
    p.add_argument("--only", nargs="*", default=None, help="cells to include")
    args = p.parse_args()
    root = Path(args.root)
    rows = load(root)
    # A run that met a hole in the scanned mesh is voided on its own: it leaves its cell, and any paired comparison
    # loses that pair. (Voiding the episode everywhere when *any* of ~36 cells met a hole dropped 24 of 120 episodes.)
    holes = [r for r in rows if r.get("scan_hole")]
    if holes:
        print(f"voided {len(holes)} runs that met a hole in the scanned mesh "
              f"({len({r['episode'] for r in holes})} episodes, {len({r['cell'] for r in holes})} cells)")
        rows = [r for r in rows if not r.get("scan_hole")]
    if args.only:
        rows = [r for r in rows if r["cell"] in set(args.only)]
    summary = {"all": table(rows, args.ref, f"## {root} (all episodes)")}
    if args.by:
        for v in sorted({str(r.get(args.by)) for r in rows}):
            summary[f"{args.by}={v}"] = table([r for r in rows if str(r.get(args.by)) == v], args.ref, f"## {args.by} = {v}")
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
