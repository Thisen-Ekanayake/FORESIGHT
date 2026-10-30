# `foresight.planning.object_search`

Language-goal object search: "find the sofa, then go back to the bed", with no map and no goal coordinates.
The robot explores its own occupancy map (built from zero-shot monocular depth, [`local_map`](../../../../src/foresight/planning/local_map.py)),
and a background Qwen3-VL-2B worker grounds the object names in what the camera sees. Run it with
[`tools/run_object_search.py`](../../../tools/run_object_search.md).

## Pieces

| Piece | What it does | Why |
|---|---|---|
| `PerceptionWorker` | One "look" per cycle, all VLM calls back-to-back on the frame seen at the look's start, delivered (measured latency) sim-seconds later: optional scene note (FORESIGHT), then per label a yes/no presence check (`option_probs`, one forward pass) and grounding (`locate`) only if P(yes) > 0.5, then a left/centre/right search hint for the current target. | One on-board GPU runs everything, so a look costs the sum of its calls; FORESIGHT's looks (~3.8 s) are slower than the baseline's (~0.5 s). |
| `box_to_xz` | Lifts a box onto the map: the box's front depth layer (30th percentile, ±0.35 m band) back-projected through the pinhole camera, scaled by `FloorScale`. | Boxes include background; the near layer is the object's front face. |
| `ObjectMemory` | Sightings merged within 1.2 m; an object is a goal after 3 sightings (distinct looks) with more hits than misses. A look that has a remembered object in view (±38°, 0.6–4 m, not occluded on the scan) and doesn't box it counts a miss. | The 2B model over-reports: grounding boxed an object on 16–34% of frames without it ([probe](#probes)). |
| Close-up check | On arrival (≤ 0.8 m, or ≤ 1.4 m and blocked) the robot faces the object and waits for a look that started after it faced it; found only if that look boxes it again within 1.5 m. Two failed checks reject the object. | A final, near-range confirmation before declaring success. |
| Frontier explorer | Known-free cells next to never-observed ones, clustered; cost = path cost − 0.15·size − `prior_weight_m`·(hint value − 1); hysteresis keeps the current frontier unless a new one is ≥ 3 cost units better; reached frontiers are blacklisted. Planning treats unmapped cells as cost 3 (vs 1 for seen-free). | Standard frontier exploration; the hint is a soft bias, never a steering command. |
| Search hint | Each look paints P(left/centre/right) from `DIRECTION_PROMPT` into the map sectors (±45° view, ≤ 8 m), running mean capped at 4 looks; reset when the target changes. | Probe: 52% argmax accuracy vs 33% chance — weak, so it only discounts frontier costs. |
| `add_reflection_wall` | On a confirmed reflection flag (from `laya_fusion.ForesightNavigator`), for each mirror box Qwen gave on that view: grow the "phantom" scan region from the box while it stays about as deep, and if it reads ≥ 1.3× deeper than the real surfaces framing it, join those two surfaces with a virtual wall. | The first version walled the whole view centre at ±35°, and false flags ("a window on the right…") put walls across real doorways. A box-only wall was too short (Qwen boxes only part of a mirror) and the planner drove round its ends. |
| Reflection filter | A sighting whose line of sight crosses a virtual wall is ignored. | Objects seen *in* a mirror lift to positions behind the glass. |
| Memory for later legs | Every look grounds all instruction labels, so a later "go back to the bed" can be served from the map at once. | |

Conditions in the runner share all of the above. FORESIGHT adds `ForesightNavigator` around it (scene notes →
Laya → double-check → wall), with `in_path_only=True` (flags about views the route doesn't enter are skipped),
and also grounds `mirror` on every look (`extra_labels`) so walls can be placed.

## Probes

All under `results/runs/object_search/` (untracked); spot checks, not results.

- `probe_grounding.py` (55 frames, 00808/00800, GT masks): one category per query → recall 0.8–1.0, frame-level
  false-box rate 0.16–0.34 for targets (0.56 mirror, 0.63 chair), ~0.3 s median. Asking for several categories at
  once made the model emit a box for every listed category; an open "list the objects" query recalled ≤ 0.4 in ~3.4 s.
  Many "false" boxes were real but unlabelled objects (a TV the annotation missed) or objects inside a mirror.
- `probe_direction.py` (50 poses): left/centre/right hint matches the shortest path's sector 52% (chance 33%),
  mean P(correct) 0.43; it almost never answers "centre".
- Presence yes/no (12 + 20 frames, plus the bench the grounding called a bed): P(yes) ≈ 1 on 11/12 frames with the
  object, ≈ 0 on 19/20 without; the black bench scored 0.00 as "bed" and 0.32 as "sofa".

## Known limits

- Exploration is slow (0.35 m/s, a 90° camera, a VLM look every ~4 s): budgets are 180 s per object.
- Many navmesh scrape ticks while turning near walls the narrow camera hasn't mapped yet (the forward-cone safety
  layer can't see sideways); reported, not hidden. A crash is the stricter rule (driving forward with < 0.3 m true
  clearance).
- The search hint is weak; one scene note can still mislead the mirror logic. Walls are permanent.

## Results (2026-10-30, `results/runs/object_search/eval/eval.json`)

16 episodes fixed by `tools/make_object_search_episodes.py` (seed 2026) and committed (`ab10b5b`) before any
evaluation run; 180 s per object; RTX 4060. `00800-tv-1` is voided for both conditions: both runs met a hole in
the scanned mesh 10 s in (GT depth 0 ahead), which the contact rule counts as a crash.

| 15 scored episodes | Baseline | FORESIGHT |
|---|---|---|
| Found (stopped within 1 m) | 2 | 0 |
| Stopped within 1.5 m | 3 | 4 |
| Ran out of time | 2 | 7 |
| Runs ending in a crash | 5 | 1 |
| …at a mirror | 5 (all the 00808 hallway mirror, from 5 different starts) | 1 (a 00808 closet mirror) |
| Look latency (median) | 0.34 s | 2.64 s |

Demo episode `mirror-sofa-bed` (two legs, not in the table): both found the sofa and the bed; FORESIGHT in 190 s,
SPL 0.42 / 0.38 (bed from memory, no re-search), baseline 244 s, SPL 0.17 / 0.18. The baseline's safety layer
(stop 0.4 m) caught the mirror frame at 0.58 m true distance in that run.

Reading it honestly:
- **Neither agent is a good object-finder yet.** Most "found" calls are the right object stopped 1.0–1.7 m away
  (the arrival rule accepts ≤ 1.4 m when blocked, and the object's estimated front face is ~0.3 m off), plus a
  few wrong objects (2.5–6 m: a sofa-like chair, a TV across the room).
- **FORESIGHT's reflection reasoning removes most crashes** (5 → 1, every baseline crash at a mirror) **but costs
  search time**: its looks are ~8× slower, so object sightings arrive later and more runs time out; double-checks
  hold the robot still (66 flags: 31 confirmed, 29 dismissed, 32 skipped as off-route, 34 skipped as known).
- FORESIGHT's one crash: Laya flagged a closet mirror at P = 0.94, the in-path rule judged the view off-route while
  the robot was turning, and it drove in 1.2 s later. Fixes to try next (not run): evaluate the in-path rule
  against the heading as well as the route; let a high-P flag stop forward motion until checked.
- Walls drawn: 0–1 per run outside 00808, 1–5 inside it (00808 has 18 annotated mirrors).
