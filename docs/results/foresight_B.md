# Deliverable B: does the on-board VLM earn its cost? (point-goal, FORESIGHT v2.3)

Same navigator, same episodes, paired: the map navigator with Depth Anything V2 (`map:da`) against the same navigator
plus FORESIGHT's mirror guard (`foresight_v23:da`), plus the depth-only reactive heuristic and ground-truth depth as
references. Run 2026-11-12/13 with `scripts/run_b_eval.sh` (≈ 5 h 20 min on the RTX 4060). Guard version, threshold
and every setting were fixed on the dev sets (A_3001, S_3002, E_9001) before these runs (commit 70f1140).

## Setup

- **Sets** (all fixed by seeded scripts before any run): **A** 120 random episodes over the 10 minival homes; **S**
  64 episodes in the 4 annotated homes, half "exposed" (route passes a mirror/glass object) and half random; **E** 33
  mirror/glass encounters (start 1.5–4.5 m from a large pane, facing it, goal straight through). Runs that touch a
  hole in the scanned mesh are voided per run (both cells), so n is 27–108.
- **Guard v2.3** (`planning/fast_slow.py`): every background look asks Qwen3-VL-2B "is there a mirror" (yes/no
  probability) and for a mirror box; a box with P ≥ 0.9 becomes a wedge candidate. Only when the planned route or the
  heading enters a wedge within 1.5 m does the robot stop, face it and run the slow check (a reflection-aware Qwen
  description, read by Laya); P(mirror) > 0.6 writes a wall into the occupancy map. At least 5 s between checks; a
  candidate within 0.8 m of an existing wall is already handled.
- **Timing is latency-honest**: sim time advances by max(0.1 s, the tick's measured compute); VLM work runs as a
  background worker whose results arrive after their measured latency. GPU contention between the two is not modelled.
- Success within 0.36 m, crash = camera within 0.3 m of a surface while driving forward; McNemar exact tests on the
  paired outcomes, bootstrap 95% intervals.

## Operating point

| Loop | Model | Rate / latency (RTX 4060 laptop, median) |
|---|---|---|
| Control tick (depth → floor scale → scan → map → plan) | Depth Anything V2 Metric Indoor Small | 45 ms compute (depth 36 ms), run at the 10 Hz control period |
| Fast look (mirror yes/no + box) | Qwen3-VL-2B | 0.18 s per look, background |
| Slow check (description + Laya) | Qwen3-VL-2B (96 tokens) + Laya | robot held 4–5 s per check (median; p10–p90 2.7–7.0 s), incl. turning to face |

## Results

| Set | Cell | n | Success % [95%] | Crash % | Timeout % | SPL | vs map:da (−/+, p) |
|---|---|---|---|---|---|---|---|
| **E** (encounters) | map:gt | 28 | 82 [68, 96] | 0 | 18 | 0.79 | |
| | map:da | 28 | 64 [46, 82] | **25** | 11 | 0.59 | ref |
| | **foresight_v23:da** | 27 | 48 [30, 67] | **15** | 37 | 0.45 | −10/+6, p = 0.45 |
| | map:da, mirror pixels fixed (oracle) | 27 | 89 [74, 100] | 0 | 11 | 0.81 | −1/+8, p = 0.04 |
| | heuristic:da | 31 | 29 [13, 45] | 26 | 45 | 0.29 | −9/+0, p = 0.004 |
| **S** (annotated homes) | map:gt | 57 | 89 [81, 96] | 0 | 11 | 0.83 | |
| | map:da | 55 | 71 [58, 82] | 4 | 25 | 0.65 | ref |
| | **foresight_v23:da** | 54 | 67 [54, 80] | **0** | 33 | 0.62 | −5/+2, p = 0.45 |
| | heuristic:da | 57 | 44 [32, 56] | 5 | 51 | 0.38 | −17/+4, p = 0.007 |
| **A** (10 homes, random) | map:gt | 104 | 83 [75, 89] | 0 | 17 | 0.77 | |
| | map:da | 107 | 79 [71, 86] | 0 | 21 | 0.72 | ref |
| | **foresight_v23:da** | 108 | **70 [61, 79]** | 0 | 30 | 0.65 | **−10/+2, p = 0.039** |
| | heuristic:da | 111 | 41 [32, 50] | 7 | 52 | 0.38 | −50/+10, p < 0.001 |

S by stratum: exposed 74% vs 78% (−2/+1), random 59% vs 64% (−3/+1). Pooled over the three sets the guard loses 25
paired episodes the map navigator solves and wins 10 (p = 0.017).

### Trigger costs (the slow module's measured price)

| Set | Episodes with ≥ 1 slow check | Slow checks / episode | Walls / episode | Wall on a real mirror/glass | Robot held (share of sim time) | Time to goal when both succeed |
|---|---|---|---|---|---|---|
| E | 92% | 6.7 | 3.1 | 44 / 84 (52%) | 22% | 67 s vs 54 s |
| S | 68% | 3.4 | 0.8 | 14 / 45 (31%) | 12% | 43 s vs 38 s |
| A | 68% | 3.4 | 1.1 | (6 of 10 homes unannotated) | 12% | 44 s vs 38 s |

On S, 51 of 91 slow checks were dismissed (E: 17 of 90), so the slow check does reject most false candidates where
mirrors are rare, but the walls it keeps are still mostly wrong (31%).

## What it says

1. **Where mirrors are approached, the guard trades crashes for timeouts.** On E it removes 6 of the map navigator's
   7 crashes at the 00808 closet mirror walls (crash 25% → 15%), but adds 3 crashes and 7 timeouts, so success falls
   (64% → 48%, n.s.). The oracle row shows what is on the table: fixing only the mirror pixels gives 89% / 0%.
2. **Elsewhere it is a cost with nothing to buy.** On A the map navigator with calibrated DA already never crashes,
   and the guard loses 10 episodes (p = 0.039), every one a timeout. In those episodes the robot was held 7–34 s, but
   it typically ended 4–11 m from the goal with time left: **false walls on the route**, not the waiting, cost the
   episodes. With ~1 wall per episode in homes with few mirrors, and 31% of walls on a real mirror where it could be
   checked, the detector's precision is the limiting factor.
3. **The remaining encounter crashes are a geometry failure, not a detection one.** All four guard crashes on E were
   at long closet mirror walls that already had a confirmed wall on them (≤ 0.31 m from the mirror). The robot hit
   the mirror 0.5–1.4 m beyond the wall's end: a wedge wall spans only the box's bearing range (+ 0.2 m), and the mirror
   continued out of view (the pre-crash view was > 90% mirror; two of the four had a later slow check dismissed at
   P = 0.27 and 0.45, the MIRA-Bench failure of VLMs on views that are mostly reflection).
4. **A box detector is enough, if its box fills depth instead of drawing walls, and if it still fires when the mirror
   fills the view.** The oracle ablation and the depth fill (next sections) show the guard's walls, not detection,
   cap v2.3; the v3 run shows the 2B VLM's boxes fail exactly when the mirror is closest.
5. **Against the spec's question** ("does a compact VLM earn its cost over a depth heuristic"): FORESIGHT beats the
   reactive depth heuristic by 19–29 points on every set, but that gain is the map navigator's, not the VLM's. With
   the navigator held fixed, the VLM guard does not pay for itself in v2.3: a negative result, reported as such.

## Detector-quality ablation: the walls, not the detector

`tools/run_detector_ablation.py` (queued by `scripts/run_det_ablation.sh`) drives the same v2.3 guard with a
simulated detector instead of the VLM: each look's mirror boxes are the semantic annotation's connected mirror/glass
regions (≥ 1% of the frame), each kept with probability *recall*, plus one random false box per look with probability
*fp*; the slow check always confirms, so the detector alone decides the walls. Looks are charged 0.17 s and slow-check
descriptions 3 s of sim time (the VLM's measured look latency; the description time is an assumption). Same 33
encounters; 26 runs common to every cell after scan-hole voiding.

| E, map navigator + … (n = 26) | Success % | Crash % | Timeout % | Walls / ep | Walls on the mirror |
|---|---|---|---|---|---|
| nothing (map:da) | 65 | 27 | 8 | – | – |
| v2.3 guard, VLM detector (as run in B) | 50 | 15 | 35 | 3.1 | 53% |
| v2.3 guard, **oracle detector** (recall 1, no false boxes) | **15** | **0** | **85** | 6.7 | 41% |
| v2.3 guard, oracle at recall 0.5 per look | 23 | 0 | 77 | 6.1 | 37% |
| v2.3 guard, oracle + a false box on 30% of looks | 0 | 27 | 73 | 9.2 | 38% |
| depth re-filled on the oracle **mask** (no walls; `fill:da@reflective`) | **88** | 8 | 4 | – | – |
| depth re-filled on the oracle mask's **bounding box** (`fill:da@reflective_box`) | **85** | 4 | 12 | – | – |
| DA with GT depth pasted on the mask (scale-aligned) | 92 | 0 | 8 | – | – |
| GT depth | 81 | 0 | 19 | – | – |

- **A perfect detector makes the v2.3 guard worse, not better**: no crashes, but 85% timeouts. Every box becomes a
  wall at one depth (the nearest reading inside the box, clamped 0.6–2.0 m), so 59% of the oracle's walls stand in
  front of or beside the mirror rather than on it, and they wall off the way round. The guard's geometry, not the VLM's
  precision, is the binding constraint; the measured 53% wall precision of the VLM run is partly the geometry's.
- **v2.4** (walls from boxes touching the image border extended by 1 m, after the four E crashes beyond a wall's end)
  changed nothing with the oracle on the dev encounters (E_dev, 9 runs: 33% / 0% / 67% for both): not pursued.
- **Fixing the depth instead of the map works**: re-fill each mirror region column by column from the depth just
  below it (a mirror is part of a vertical plane, whose depth is constant down a column: the floor where it meets the
  mirror, or the wall under it; `perception/depth/sources.py:MirrorFillDepth`). With the oracle mask this takes the
  encounters from 65% / 27% to 88% / 8%, close to pasting ground truth (92% / 0%), using only Depth Anything's own
  values. With only the bounding box, 85% / 4%: **a box detector is enough, if its boxes fill depth rather than draw
  walls.**

## FORESIGHT v3: VLM boxes re-fill the depth (`--guard fill`)

`planning/fast_slow.py:BoxFillGuard`: no slow check, no walls. The most recent look with P(mirror) ≥ 0.9 within
`hold_s` gives boxes; each tick they are shifted across the image by the robot's turn since that look and their
columns re-filled before mapping. A false box bends the depth only while held; free-space carving undoes it.

Two things found while bringing it up (one E episode, e-00808-closetmirrorwall527-0, 10 s replay):
- a look that answers "mirror" also runs grounding and takes 1.5–2.7 s (0.17 s for "no"), so boxes are old on arrival;
- **Qwen3-VL-2B answers "no mirror" once the closet mirror fills the view** (P ≈ 0.001 on every look from 2.7 s to
  7.3 s, after 0.97 on the approach): the MIRA-Bench failure. The box from the approach must be held.

A first E run with a 0.6 s age limit never filled a single tick (bug: every box was older than that; results kept
only as `E/foresight_fill_bug_noop`, identical to map:da). The hold time is chosen on the dev encounters (E_dev, 3 s
vs 8 s) before the E run.

| E (n = 26 common) | Success % | Crash % | Timeout % |
|---|---|---|---|
| map:da | 65 | 27 | 8 |
| v2.3 guard (VLM, walls) | 50 | 15 | 35 |
| **v3 fill guard (VLM boxes, 3 s hold)** | **62** | **31** | 8 |
| oracle box fill (`fill:da@reflective_box`) | 85 | 4 | 12 |

E_dev (hold chosen there, n = 10): map 40% / 40% crash, v3 60% / 20% with 3 s and with 8 s (identical outcomes),
v2.3 50% / 10%.

**v3 does not carry over to E** (−2/+1 against the map navigator, p = 1). The fill fires on 41% of ticks in the
episodes that crash (16% in successes), but every one of the 8 crashes happens with the view > 90% mirror, the
phase where Qwen3-VL-2B says "no mirror" and a box held from further back no longer covers what the robot sees (the
hold ignores translation). The oracle box fill shows the mechanism works when the box is right *in the last metre*;
the 2B VLM cannot supply that box. The dev-set gain (2 of 10) was noise.

## What would change the answer

- A detector that keeps firing at close range: a dedicated RGB mirror segmenter (GDNet / HetNet class, ~30 fps per
  Athena's mirror map) feeding `MirrorFillDepth`, or the VLM asked a close-range question ("is the whole view a
  reflection?"). Not run: weights need a download over the slow link.
- Tracking a mirror box in the world (lift it to the floor plane once, then re-project it every tick with the full
  pose) instead of holding an image box with a turn-only shift.
- The same evaluation on S and A for any fill variant, to price its false boxes outside encounters (not run: v3 did
  not pay on E).

## Limits

- One simulator with baked mirrors (Habitat renders reflections as textures; depth sensors see them as surfaces).
  Only 00808 has large, approachable mirrors, so E is dominated by its closet mirror walls.
- Sim time does not charge GPU contention between the VLM worker and the depth model.
- Wall precision is measured only in the 4 annotated homes.
