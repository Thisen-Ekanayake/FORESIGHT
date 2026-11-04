#!/usr/bin/env python
"""Deliverable A figures: navigation outcome against injected depth error, one panel per error type.

Reads <root>/summary.json (tools/summarize_pointnav.py). Rows: success rate, crash rate (95% bootstrap intervals).
Series: the map navigator (floor-plane self-calibration on), the same without it (scale panel), the depth-only
heuristic (where run). Depth Anything V2 is drawn as a reference line in each panel.

  python tools/plot_depth_error.py results/runs/pointnav/A --out results/figures/depth_error_A
Writes <out>.png and <out>.svg.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (dataviz skill, light): validated all-pairs for 3 slots; slot 3 is below 3:1 on the surface,
# so every series is also direct-labelled and shape-coded.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"map": ("#2a78d6", "o", "map navigator"),
          "map_nofs": ("#eb6834", "s", "map, no floor calibration"),
          "heuristic": ("#1baf7a", "^", "depth-only heuristic")}
PANELS = [("scale", "global scale ×", 1.0), ("bias_m", "additive bias (m)", 0.0), ("noise", "noise σ (× depth)", 0.0),
          ("noise_corr", "correlated noise σ (16 px)", 0.0), ("holes", "missing pixels (fraction)", 0.0),
          ("quant_m", "quantization step (m)", 0.0)]


def parse(cell):
    cond, depth = cell.split(":", 1)
    if depth == "gt":
        return cond, None, None
    m = re.fullmatch(r"gt:([a-z_]+)=(-?[0-9.]+)(,noise_px=16)?", depth)
    if not m:
        return cond, depth, None
    k = "noise_corr" if m.group(3) else m.group(1)
    return cond, k, float(m.group(2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--out", default="results/figures/depth_error_A")
    args = p.parse_args()
    cells = json.loads((Path(args.root) / "summary.json").read_text())["all"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, len(PANELS), figsize=(16, 6.2), sharey="row", facecolor=SURFACE)
    for j, (key, xlabel, x0) in enumerate(PANELS):
        for i, metric in enumerate(("success", "crash")):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for cond, (color, marker, label) in SERIES.items():
                pts = []
                for cell, v in cells.items():
                    c, k, lvl = parse(cell)
                    if c != cond:
                        continue
                    if k is None and lvl is None and cell.endswith(":gt"):
                        pts.append((x0, v[metric]))
                    elif k == key:
                        pts.append((lvl, v[metric]))
                if len(pts) < 2:
                    continue
                pts.sort()
                xs = [x for x, _ in pts]
                m = [100 * y[0] for _, y in pts]
                lo = [100 * (y[0] - y[1]) for _, y in pts]
                hi = [100 * (y[2] - y[0]) for _, y in pts]
                ax.errorbar(xs, m, yerr=[lo, hi], color=color, marker=marker, ms=6, lw=2, capsize=0, elinewidth=1.2,
                            mec=SURFACE, mew=1.5, label=label)
                if i == 0:
                    ax.annotate(label.split(",")[0] if cond != "map_nofs" else "no floor calib.", (xs[-1], m[-1]),
                                xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
                da = cells.get(f"{cond}:da")
                if da is not None and (key == "scale" or cond == "map"):
                    ax.axhline(100 * da[metric][0], color=color, lw=1, ls=(0, (4, 3)), alpha=0.9)
            if key == "scale":
                ax.axvline(1.0, color=GRID, lw=1)
            ax.set_ylim(-3, 103)
            if j == 0:
                ax.set_ylabel(("Success" if metric == "success" else "Crash") + " rate (%)")
            if i == 1:
                ax.set_xlabel(xlabel)
            else:
                ax.set_title(xlabel.split(" (")[0].replace(" ×", "").capitalize(), fontsize=10, color=INK, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK2, ls=(0, (4, 3)), lw=1))
    labels.append("Depth Anything V2 (real estimator), same navigator")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Point-goal navigation in 10 HM3D homes vs. injected depth error (ground truth + error; ~105 episodes "
                 "per point, 95% intervals)", y=0.935, fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix("." + ext), dpi=160, facecolor=SURFACE)
    print(f"-> {out}.png / .svg")


if __name__ == "__main__":
    main()
