# Deliverable A: how depth error becomes navigation error

Point-goal navigation in Habitat-Sim on the 10 HM3D minival homes (set A: 120 episodes, seed 3003, fixed and
committed before any run). Every run uses the same latency-honest loop and crash rule; only the depth the navigator is
given changes: simulator ground truth (the spec's depth-sensor baseline), ground truth with one controlled error, or
Depth Anything V2. Runner `tools/run_pointnav_eval.py`, sweep `scripts/run_depth_error_sweep.sh`, tables
`tools/summarize_pointnav.py`, figure `tools/plot_depth_error.py`. Results: `results/runs/pointnav/A/summary.json`.

![Success and crash rate against each injected error](depth_error_A.svg)

## Setup

- **Navigators.** `map`: map-free occupancy mapping from depth + Dijkstra + carrot + reactive safety layer, with
  floor-plane scale self-calibration (`FloorScale`: the camera height is known, so floor pixels have a known depth).
  `map_nofs`: the same with calibration off. `heuristic`: the depth-only potential field (milestone #7), with the same
  calibration. No VLM in this study.
- **Success**: within 0.36 m of the goal before the time budget (30 s + 3× geodesic at 0.35 m/s). **Crash**: driving
  forward with < 0.3 m true clearance ahead (ends the run).
- **Voiding.** A run that met a hole in the scanned mesh (ground-truth depth ≈ 0 ahead) is voided on its own: 435 of
  4,320 runs, concentrated in 24 episodes whose paths pass real mesh holes. Cells keep 104–112 episodes; paired tests
  use only pairs where both runs are valid.
- **Statistics**: 95% bootstrap intervals; vs ground truth, exact McNemar on paired success (losses/gains, p).

## Results (map navigator unless stated)

| Depth given | n | Success % [95%] | SPL | Crash % | Depth AbsRel | vs GT (−/+, p) |
|---|---|---|---|---|---|---|
| Ground truth | 104 | 83 [75, 89] | 0.77 | 0 | 0.00 | ref |
| **Depth Anything V2** | 107 | **79 [71, 86]** | 0.72 | 0 | 1.39 | −10/+8, p = 0.82 |
| GT × 0.5 … × 2.5 (scale) | 104–106 | 80–82 | 0.74–0.76 | 0 | 0.30–1.50 | p ≥ 0.5 at every level |
| GT − 0.3 m (too near) | 110 | 43 [34, 53] | 0.39 | 0 | 0.23 | −38/+0, p < 1e-11 |
| GT + 0.3 m (too far) | 105 | 94 [90, 98] | 0.87 | 2 | 0.22 | −0/+12, p = 0.0005 |
| GT + 0.6 m | 105 | 80 [72, 88] | 0.73 | 10 | 0.45 | −11/+8, p = 0.65 |
| GT + noise σ = 0.1 × depth | 109 | **5 [1, 9]** | 0.04 | 0 | 0.08 | −79/+0, p < 1e-23 |
| GT + noise σ = 0.2 / 0.4 | 110–112 | 0 | 0.00 | 0 | 0.16 / 0.32 | p < 1e-25 |
| GT + correlated noise σ = 0.1 / 0.2 (16 px) | 107–112 | 10 / 1 | 0.10 / 0.01 | 0 | 0.08 / 0.16 | p < 1e-21 |
| GT, 10–50% pixels missing | 104–106 | 83–90 | 0.77–0.84 | 0 | (holes) | p ≥ 0.09 |
| GT quantized to 0.1 / 0.25 / 0.5 m | 104–106 | 88 / 79 / 83 | 0.73–0.80 | 0–4 | 0.02–0.10 | p ≥ 0.07 |
| *No floor calibration:* ground truth | 106 | 91 [85, 95] | 0.85 | 0 | 0.00 | −3/+11, p = 0.06 |
| *No floor calibration:* Depth Anything V2 | 111 | 35 [26, 44] | 0.32 | **34** | 1.42 | −52/+5, p < 1e-10 |
| *No floor calibration:* GT × 0.5 / × 0.7 | 110–111 | 0 / 2 | ~0 | 0 | 0.50 / 0.30 | p < 1e-24 |
| *No floor calibration:* GT × 1.4 | 106 | 89 | 0.80 | 0 | 0.40 | p = 0.17 |
| *No floor calibration:* GT × 2.0 / × 2.5 | 108–112 | 47 / 19 | 0.42 / 0.16 | **21 / 54** | 1.00 / 1.50 | p < 1e-5 |
| Heuristic: GT / DA / noise 0.2 / scale 0.7–2.0 | 110–112 | 44 / 41 / 40 / 40–45 | 0.37–0.42 | 0–7 | | (heuristic ceiling ~44%) |

## What it says

1. **The real estimator costs this navigator almost nothing.** Depth Anything V2 reaches 79% against ground truth's
   83%, not a detectable difference at n ≈ 105 (p = 0.82), despite a mean AbsRel of 1.39 (it reads HM3D ~2× too far).
2. **Because its error is almost all global scale, and the floor calibration removes scale.** With calibration, scale
   errors from 0.5× to 2.5× change nothing (80–82%). Without it the same errors are catastrophic in both
   directions: too near (×0.5, ×0.7) everything looks blocked and the robot never arrives (0–2%, all timeouts); too far
   (×2, ×2.5) it crashes (21%, 54%). Uncalibrated Depth Anything: 35% success, 34% crashes. So the claim that
   navigation tolerates monocular scale error (LG-VLN, see `docs/lit_review/positioning.md`) holds only with a scale
   correction. It is the calibration, not the estimator, that makes monocular depth usable here.
3. **Error magnitude does not predict navigation error; error type does.** Zero-mean noise with σ = 10% of depth (AbsRel
   0.08, RMSE 0.18 m) collapses the map navigator to 5%, while a 2.5× scale error (AbsRel 1.50, RMSE 3.3 m) costs
   nothing. A single depth-RMSE axis would put these in the wrong order. This is the core of hypothesis H-A, though
   the damaging type here is noise, not (yet) the structured reflective error the hypothesis names.
4. **The noise collapse is an interaction with the map representation.** The mapper takes the *nearest* point of
   obstacle height in each image column; on floor pixels, depth noise moves the computed height across the 0.2 m
   obstacle threshold, so zero-mean noise becomes systematic phantom obstacles. Offline on saved frames, σ = 0.1 makes
   13–36% of columns read ≥ 0.5 m nearer than truth (σ = 0.2: 71–72%). The depth-only heuristic, which uses no height
   filter, is barely affected (40% vs 44%). Robust per-column statistics are the obvious remedy; that ablation is not
   run yet.
5. **Error direction matters: too near costs more than too far, until too far starts crashing.** −0.3 m halves success
   (43%: the robot treats doorways as blocked); +0.3 m *improves* it (94%: it commits to narrow gaps), and +0.6 m brings
   crashes (10%).
6. **Missing data and quantization are harmless** for this mapper (missing pixels are skipped, and a column still finds
   its nearest valid point); 50% holes slightly *help* (fewer spurious obstacles).
7. **The planner, not perception, sets the ceiling.** Even with ground truth the map navigator succeeds 83% of the
   time (91% without the floor calibration, whose own estimates are noisy on beds and tables: p = 0.06); the reactive
   heuristic stays near 44% under every error. Perception error is measured against these ceilings.

## Limits

- One navigator family (map-based + reactive), 10 minival homes, simulated depth error on ground truth. The noise
  model (independent or 16-px-correlated Gaussian ∝ depth) is harsher than the estimator's spatially smooth error.
- HM3D lighting and mirrors are baked; reflective and textureless structured errors are measured on set S (below,
  pending).
- Scan holes remove ~10% of runs; voiding is per run, so cells differ slightly in which episodes they keep.

## Robust depth scan (ablation, 2026-11-10)

The noise collapse comes from the map builder's scan (nearest obstacle-height point per image column). The ablation
`depth_scan(robust=True)` median-filters the depth (5 x 5 at scan resolution) and needs 3 supporting points within 0.2 m
per column. Set A, map navigator, ~105 episodes per cell:

| Depth given | Plain scan | Robust scan | Robust vs GT (−/+, p) |
|---|---|---|---|
| Ground truth | 83% | 84% | −1/+2, p = 1 |
| Depth Anything V2 | 79% | 78% | −9/+6, p = 0.61 |
| Noise σ = 0.1 / 0.2 / 0.4 | 5% / 0% / 0% | **81% / 75%** / 1% | p = 1 / 0.18 / < 1e-24 |
| Correlated noise σ = 0.1 / 0.2 (16 px) | 10% / 1% | 66% / 34% | p < 0.001 |
| Bias −0.3 / +0.3 m | 43% / 94% | 43% / 92% | unchanged |
| Scale × 2.5 | 81% | 84% | unchanged |

The robust scan removes the collapse for iid noise up to σ = 0.2 at no cost elsewhere on set A; spatially correlated
noise (smoother than the filter) is only partly fixed, and σ = 0.4 is beyond it. The trade-off shows on mirror
encounters (below): the smoothing also erases the thin real frame around a mirror, and Depth Anything + robust scan
crashes into mirrors more often (41% vs 25%). So the collapse was a property of how the mapper used depth, and the fix
is not free.

## Structured errors (set S, 4 annotated homes)

64 episodes in the 4 homes with semantic annotations, half of them per home the shortest paths that see the most
mirror/glass (reflective pixels along the path: exposed 4–16% mean, random 0–7%); set S seed 3004, fixed before any run.
151 runs voided for scan holes (12 episodes); cells keep 54–57 episodes.

**Correction (2026-11-10).** The reflective category list matched names exactly and missed "closet mirror wall" (7 in
00808, including the hallway mirror and the closet mirrors behind every object-search mirror crash), "exhibition window"
and "door window" (00803). The reflective cells below (reflect ×, and the two reflective swaps) were re-run with the fixed
list; the earlier runs are kept in `results/runs/pointnav/S_oldmask_*`. The conclusion did not change in aggregate, but
the map navigator now shows crashes at × 2–3 (2–5%, previously 0%). Set S's exposure stratification was computed with the
old list, so its "exposed" stratum under-represents 00808's closet mirrors.

| Depth given (map navigator) | n | Success % [95%] | Crash % | Depth AbsRel | vs GT (−/+, p) |
|---|---|---|---|---|---|
| Ground truth | 57 | 89 [81, 96] | 0 | 0.00 | ref |
| Depth Anything V2 | 55 | 71 [58, 82] | 4 | 1.41 | −11/+1, p = 0.006 |
| GT, mirror/glass pixels read × 1.5 / 2 / 3 too deep | 56–57 | 89 / 89 / 86 | 0 / 2 / 5 | 0.02–0.12 | −1/+1 at most, p ≥ 0.5 |
| GT, textureless regions flattened | 57 | 86 | 0 | 0.01 | −3/+1, p = 0.63 |
| GT everywhere, DA's own (scale-aligned) error on mirror/glass only | 57 | 86 | 2 | 0.02 | −2/+0, p = 0.50 |
| GT everywhere, DA's error on textureless regions only | 55 | 93 | 0 | 0.04 | −0/+2, p = 0.50 |
| DA everywhere, mirror/glass pixels fixed (GT) | 53 | 81 | 0 | 1.38 | vs DA: +10 points (n.s.) |
| DA everywhere, textureless regions fixed (GT) | 54 | 80 | 0 | 1.36 | vs DA: +9 points (n.s.) |
| Heuristic: GT / reflect × 3 | 57 | 35 / 25 | 0 / **12** | | reflect × 3 adds crashes |

Exposed vs random stratum: same pattern (map + reflect × 3: 96% = GT on exposed episodes).

What it says:

1. **On these homes Depth Anything does cost the map navigator** (89% → 71%, p = 0.006), unlike on the 10-home set A
   (83% → 79%, n.s.). Fixing DA's mirror/glass pixels recovers ~10 of the 18 points and fixing its textureless regions
   ~9, neither significant at n ≈ 55.
2. **Injected reflective error barely moves the map navigator on random point-goal episodes, even exposed ones**
   (89% → 86% at ×3, with 5% crashes; n.s.). Mirror/glass is a few percent of the pixels seen, the occupancy map fuses many views
   (a mirror seen obliquely or from its side is mapped correctly), and random start/goal pairs rarely require driving
   *at* a mirror. The reactive heuristic, which has no map, does start crashing (×3: 11% crashes vs 0%).
3. **So H-A's structured half is not supported in aggregate**: reflective failure is a *tail* event that a random-episode
   benchmark dilutes. Where mirrors are actually approached it dominates (encounters, next section).
4. **Textureless regions are harmless for both navigators** in HM3D (flattened texture or DA's error there: no effect).

## Mirror/glass encounters (set E, deliverable C's mirror stratum)

33 encounters (`tools/make_encounter_episodes.py`, seed 3005, fixed before any run): the robot starts 1.5–4.5 m from a
large mirror or glass pane, facing it, and the goal lies straight through it (21 closet mirror walls incl. the 00808
hallway mirror, 4 mirrors, 8 windows). 44 runs voided for scan holes; cells keep 27–32.

| Depth given (map navigator) | n | Success % [95%] | Crash % [95%] | vs GT (−/+, p) |
|---|---|---|---|---|
| Ground truth | 28 | 82 [68, 96] | 0 | ref |
| **Depth Anything V2** | 28 | 64 [46, 82] | **25 [11, 43]** | −6/+2, p = 0.29 |
| DA, mirror/glass pixels fixed (GT, scale-aligned) | 27 | **89 [74, 100]** | **0** | −1/+4, p = 0.38 |
| GT, DA's own error on mirror/glass pixels only | 28 | 71 [54, 86] | 11 [0, 25] | −3/+0, p = 0.25 |
| GT, mirror/glass read × 2 / × 3 too deep | 28 | 43 / 18 | **43 / 75** | p = 0.003 / < 0.001 |
| DA, robust scan | 29 | 48 [31, 69] | 41 [24, 59] | −11/+2, p = 0.02 |
| Heuristic: GT / DA / GT reflect × 3 | 31–32 | 34 / 29 / 16 | 0 / 26 / 25 | |

By pane: every DA crash is at a closet mirror wall (7 of 21); at × 3 all 21 closet-mirror encounters crash; windows
and the smaller mirrors rarely fail (their reflection is usually the room's own far wall or outdoors, read as "far" but
behind a frame the scan still sees).

- **Depth Anything's encounter crashes come entirely from its mirror/glass pixels**: replace only those pixels with
  ground truth and the crashes vanish (25% → 0%, success 64% → 89%, at the GT level).
- **The effect is dose-dependent**: reading the mirror 2× too deep crashes 43% of encounters, 3× crashes 75%.
- So H-A's structured half holds **where reflective surfaces are actually approached**: a small fraction of pixels
  causes most of the failures, invisible in aggregate and decisive in encounters. A mirror detector that fixes those
  pixels would be worth more than any improvement in average depth accuracy; that is FORESIGHT's mirror guard's job
  (deliverable B, to be run on this set).

## Summary for H-A

| Error type | Map navigator | Reactive heuristic |
|---|---|---|
| Global scale (the estimator's dominant error) | harmless *with* floor calibration; catastrophic without | harmless (calibrated) |
| Additive bias | too near halves success; too far helps until it crashes | – |
| Zero-mean noise | catastrophic at σ = 10% with the plain scan; fixed up to σ = 0.2 by the robust scan | harmless |
| Missing data, quantization | harmless | harmless |
| Reflective (structured) | harmless in aggregate; **25–75% crashes in mirror encounters** | crashes at × 3 |
| Textureless (structured) | harmless | harmless |

Error **type** predicts navigation error far better than error **magnitude**, and the damaging types depend on how the
navigator uses depth (calibration, per-column nearest points, map vs reactive). The real estimator's error is mostly
the benign kind once scale is calibrated.

## Concurrent work: *From Depth to Safe Depth* (arXiv 2611.13769, 10 Nov 2026)

The closest published result, found in the 2026-11-12 literature update (`docs/lit_review/update_2026-11-12.md`).
Nav2 in Gazebo procedural worlds, 120 paired episodes: Depth Anything V2 metric collides in 63% (37% success, GT 93%);
a trained 0.58 M correction head plus conformal uncertainty reaches 70% / 3%. They reject a constant scale because
the predicted/true ratio varies 0.8–2 across scenes, but do not try a per-scene online scale, which is what
FloorScale is.

Their depth metric, near over-estimation (fraction of readings closer than 3 m that the model over-estimates by more
than 0.2 m), on our set-A traces (forward ray, every tick with true distance 0.1–3 m; theirs is per pixel, so the two
are not identical):

| Set A (120 episodes, map navigator) | Near over-estimation | Success / crash |
|---|---|---|
| Depth Anything V2, no FloorScale | 98% | 35% / 34% |
| Depth Anything V2 + FloorScale (training-free) | 11% | 79% / 0% (n.s. vs GT) |
| GT depth (definition floor: ray vs mesh sampling) | 2% | 83% / 0% |
| *Safe Depth* (Gazebo): DA-V2 raw → trained head µ | 59.8% → 12.0% | 37% / 63% → 60% / 15% (70% / 3% with σ) |

On scanned HM3D homes a per-scene floor-based scale removes about as much near over-estimation as their trained
head does in procedural worlds, without training. This is a contrast across simulators and planners, not a
head-to-head. What remains after FloorScale is structured (mirrors, glass), as H-A says, and as their glass group
(37% at best vs 94% with GT) also shows.
