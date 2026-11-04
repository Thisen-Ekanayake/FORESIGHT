# FORESIGHT — current state (2026-11-02, stopped on request)

Read with `PLAN.md` (the plan, revised after the lit review) and `logs/2026-11-02.md` (what was done, point by point).

## Done

| Phase | State | Where |
|---|---|---|
| 0 Housekeeping | ✅ Isaac Go2 friction fix committed (`ade8990`). Team names in PROGRESS.md still empty. | |
| 1 Lit review (Athena) | ✅ Six maps, gaps, idea checks, web cross-check; plan revised from it (`e4eace9`) | `docs/lit_review/`, Athena feedback in `/ml/Athena/docs/feedback/2026-11-02-foresight-lit-review.md` (uncommitted in Athena's repo) |
| 2 Point-goal infrastructure | ✅ Depth sources (da / gt / gt:errors / scale-aligned masked swaps), runner, paired stats, escape manoeuvre, scan-hole voiding (`9e9ff6b`, `042d539`, last commit) | `src/foresight/perception/depth/sources.py`, `tools/run_pointnav_eval.py`, `tools/summarize_pointnav.py`, `src/foresight/eval/stats.py` |
| Episode sets | ✅ Evaluation sets fixed and committed before any run: **A** = `episodes_A_3003.json` (120, 10 homes), **S** = `episodes_S_3004.json` (64, 4 annotated homes, reflective-stratified). Dev sets: `A_3001`, `S_3002` (pilot ran there). | `experiments/configs/pointnav/` |
| 3 Deliverable A | ✅ Sweep complete (5,600 runs, 2026-11-04); results + figure in `docs/results/depth_error_A.md` | `results/runs/pointnav/{A,S}/summary.json` |

## Pilot findings (dev set A_3001, 30 episodes, scan holes voided → 27)

| Cell | Success | SPL | Crash |
|---|---|---|---|
| map + GT depth | 70% [52, 85] | 0.67 | 0% |
| map + Depth Anything V2 | 63% [44, 81] | 0.54 | 4% |
| heuristic + GT / DA (first pilot) | 23% / 20% | 0.22 / 0.17 | 3% / 13% |
| FORESIGHT v2.1 + DA (first pilot) | 33% | 0.31 | 13% |

- The map navigator's ceiling with perfect depth is ~70%. Failures are tight starts and long wanders/timeouts, a
  planner limit. A is measured as degradation relative to GT (the spec's depth-sensor baseline), so this is
  acceptable. The navigator was **not** tuned further.
- **FORESIGHT v2.1 halves point-goal success vs `map`** (11 losses, 1 gain, McNemar p = 0.006): 4–18 slow checks per
  episode and 1–6 confirmed "mirror" walls even in homes without annotated mirrors. Those walls cut routes. This is
  the same over-checking seen on object-search set 3.
- The heuristic (reactive potential field) mostly times out (local minima), even with GT.
- Throughput: GT-based runs ~10 s wall per episode, DA ~50 s (three workers in parallel); FORESIGHT ~105 s and
  7.5 GB GPU peak, so it runs alone.

## Sweep schedule (2026-11-04)

Resumed 02:16 (estimated ~10 h in total). `scripts/stop_sweep_at.sh 09:00` stops it at 09:00 and appends the run count to
`results/runs/pointnav/sweep_A.log`. It stays stopped until resumed with the first command below.

## To resume

```bash
nohup scripts/run_depth_error_sweep.sh > results/runs/pointnav/sweep_A.log 2>&1 &   # resumable: skips finished runs
/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python tools/summarize_pointnav.py results/runs/pointnav/A --ref map:gt
/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python tools/summarize_pointnav.py results/runs/pointnav/S --ref map:gt --by stratum
```

Stages in the script, in order (~6–8 h left on the RTX 4060):
1. `map` × 20 uniform sources + DA (set A);
2. the scale sweep with `--no-floor-scale`;
3. `heuristic` on the key levels;
4. structured errors (reflect, flat_tex, masked swaps) for `map` + `heuristic` on set S.

## Next (as of 2026-11-04 18:30)

Done since: the A analysis and figure (`docs/results/depth_error_A.md`). Headline: error type, not magnitude, predicts
navigation error; reflective error doesn't show up on random episodes, so C needs encounter episodes (PLAN.md Phase 4).
Remaining, in order: v2.2 mirror guard on the dev sets (B); encounter episodes for C; a robust-scan ablation for the
noise collapse (per-column percentile or median-filtered depth) is optional.

## Earlier next-steps list (kept for reference)

1. **Analysis + figures for A** (not written yet): per-error-type curves of success/SPL/crash against depth AbsRel/RMSE
   with CIs; DA-V2 placed on the curve; floor-scale on vs off; share of collisions with structured error before
   the crash (`pre_crash_*` fields in each run's JSON). This tests H-A (PLAN.md). Load the dataviz skill before
   plotting.
2. **v2.2 mirror guard on the dev sets** (B): don't re-check mirrors already walled from another angle; stop only for
   candidates within ~1.5 m. Measure guard precision against semantics on `S_dev` (are confirmed walls on real
   mirror/glass?). Then evaluate on A/S.
3. **Phase 4 (C)**: mirror/glass and textureless strata from the S runs (per-frame `reflective_frac`,
   `textureless_frac` are already logged). Low light and blur are secondary and use RobustNav/NavTrust corruption
   definitions (not implemented yet).
4. Phase 5 (B) tables, then the write-up.

## Loose ends

- PROGRESS.md is git-ignored and was only lightly updated (points to PLAN.md). Milestones #8 and deliverable A
  statuses haven't changed yet.
- `results/runs/pointnav/{smoke,pilot,pilot_escape}` are dev outputs (untracked), safe to delete.
- Athena: the stack and GROBID were stopped and the feed pause was undone. The session cookie the user provided is
  still valid in Athena (only my copy was deleted).
