# Changelog

Notable changes to FORESIGHT, newest first. Day-by-day working notes live in `logs/` (untracked);
milestone status in `PROGRESS.md`.

## [Unreleased] — Promo video "One Camera, No Map": live SONIC runs and compositor (set up 2026-11-21, produced 2026-11-23)

### Produced (2026-11-23)
- The master, `results/isaac/promo/foresight_promo.mp4`: 1920², 30 fps, 2:08, Kokoro af_heart narration plus music,
  loudness-normalized. Seven live SONIC runs (all passed `check_runs.py`), 11 path-traced shots at 32 spp.
- Search segment uses episode s2-00808-sofa-1 ("Find the sofa") after mirror-sofa-bed crashed into the mirror; the
  no-calibration robot timed out rather than crashing ("never arrives"). Both are recorded in the docs.
- Compositor fixes from the full render: the depth grid holds each robot's 7.2 s frame (the moving cursor landed on a
  wall), its labels sit at the tile bottoms; DEPTH ONLY fades when COLLISION appears; the mirror check's labels are
  anchored on the robot (the mirror is behind that camera); `sys_shoulder` uses a fixed camera; audio loudnorm.

### Added
- `tools/isaac/sonic_walker.py`: `SonicWorld`, SONIC's G1 in a plain Isaac Lab sim driven by FORESIGHT's
  (forward, turn) command (slow_walk at speed × gain, idle + facing angle to turn on the spot, heading lead clamp);
  `robot_cfg()` shared with `g1_sonic.py`.
- `tools/isaac/g1_live.py`: `--walker sonic` (+ `--sonic-speed-gain`, `--sonic-cpu`, `--sonic-seed`, `--wbc`),
  `--task search --instructions ...` (object search; ends when the brain reports done), `--depth`, `--no-floor-scale`,
  `--depth-seed` (the A study's depth sources and seeding, passed to the brain). Defaults unchanged.
- `tools/isaac/foresight_brain.py`: depth sources other than Depth Anything (`gt`, `gt:noise=0.1`, ...) on Habitat's
  ground truth, FloorScale off, and object search (baseline / FORESIGHT v1, as `tools/run_object_search.py`).
- `tools/isaac/replay_render.py`: `"ghosts": [{motion, tint, opacity, offset}]` for several translucent robots per
  shot (one env each); frames.json logs every robot's pose. The single `"ghost"` still works.
- `tools/isaac/promo/`: `live_runs.sh` (resumable live runs by stage, GPU check), `check_runs.py` (gait, speed gain,
  each run vs its segment), `make_shots.py` (shot list timed from the runs' events, render estimate), `render.sh`
  (32 vs 64 spp stills, resumable per-shot full render).
- `tools/make_promo_video.py` + `foresight.viz.promo_video`: the narrated 1920² compositor (point cloud and growing
  map from the brain's own depth, hidden behind the render's depth; plan path; VLM label; four-robot depth grid;
  mirror check; time-lapse search with the spoken goal as on-screen words; end card with the SONIC credit;
  `--preview T`, `--scale 0.5`).

## [Unreleased] — G1 walking-quality fixes for Isaac renders (2026-11-20)

### Added
- NVIDIA GEAR-SONIC for the G1 (motion-tracking whole-body controller + kinematic walking planner; NVIDIA Open Model
  License, `/ml/Models/gear_sonic`): `tools/isaac/sonic.py` (Python/onnxruntime port of the C++ reference deploy)
  and `tools/isaac/g1_sonic.py` (SONIC's G1 in Isaac Lab, scripted mode/speed/heading profile → motion.json).
  Natural gait: ~0.5–0.7 m/s at ~1–1.3 steps/s with 45–60 cm steps (the Isaac Lab pretrained policies take ~5–6
  steps/s with 6–15 cm steps at any speed). `replay_render.py --robot sonic` renders it on Isaac Sim's G1
  (29 DoF + Dex3), joints matched by name. `g1_live.py --policy rough` for the other Isaac Lab checkpoint.
- `tools/isaac/g1_live.py --profile '[[s, v, w], ...]'`: scripted command profile, no brain (gait tests, clean demo
  walks). `--cmd-tau S`: optional first-order smoothing of the (v, w) command before the walking policy (off by
  default, so no recorded or evaluated run changes).
- `tools/isaac/replay_render.py`: `rig` camera mode (offset in the robot's frame); per-shot `rtx` renderer settings,
  `updates` and `preroll`; `--iso N` fixes exposure at launch (histogram off, film ISO N). Turning the histogram off
  at runtime through carb settings does not stick (exposure still stepped 218 → 193 → 134 within one shot).
- `replay_render.py --deterministic` (on by default): synchronous geometry streaming and no path-tracing / ray-tracing
  caches or spatial light cache (NVIDIA's synthetic-data settings). Without it the first 35–130 frames after each
  launch came out 2–2.5× too bright with a random whole-room colour cast (yellow, blue, pink, green), then snapped
  to the right look; no renderer setting changed at the snap. `--tex-streaming 0` and `--debug-settings` (per-frame
  /rtx settings diff) added while diagnosing.
- `tools/isaac/deflicker.py`: temporal deflicker for rendered frame sequences. Each frame's per-channel tone curve is
  matched to a running median of its neighbours', which removes the path tracer's occasional 1–10-frame whole-frame
  flashes (+25 to +40 luma, lifted blacks) and keeps slow, real lighting changes.

### Changed
- `replay_render.py`: poses are interpolated between the 20 ms control steps (lerp + slerp) instead of snapping to
  the nearest step; cameras track the root path smoothed over ±0.6 s (`"raw": true` keeps the old behaviour).

### Found (why the "Through the Glass" G1 looked shaky)
- Not the policy or the scan floor: steady walking/turning commands give ~1.1° body roll on flat ground and on the
  00808 floor alike. The live run's commands did it: turn command flipping ±0.7 rad/s (48 sign flips), ~6 changes/s,
  30% of the time turning in place (body roll up to ±7°).
- Replay: shots at 1.6–3× speed rendered at 15 fps (4–6 frames per step) and nearest-step poses.
- Path-traced frame flicker at 24 spp with auto exposure (orbit shot: whole-frame tint changes every frame).
  64 spp + fixed exposure removes it at the same cost (~8.3 s/frame at 1440²). Even with exposure fixed at launch
  (ISO 14), the first 35–130 frames after launch were wrong (renderer caches, fixed by `--deterministic`); a few
  later frames flashed, which `deflicker.py` removes. Real cost is ~23 s/frame at 1440² (8.3 s was for stills).

## [Unreleased] — The Mirror Test: Isaac Sim humanoid demo (2026-11-19)

### Added
- `tools/isaac/house_stage.py`: shared Isaac stage builder (HM3D house, mirror material, lights, floor collider,
  cameras) with a scan-texture emission scale and dollhouse cuts (height cut, footprint box).
- `tools/isaac/mirror_test.py`: a Unitree H1 humanoid walks up to the 00808 hallway mirror, path traced (its head
  camera sees its own reflection); saves chase + head camera + Isaac ground-truth depth + mirror mask per frame.
  `--chase side|behind|corner`, `--robot h1|go2`, `--emission`.
- `tools/isaac/mirror_test_perception.py`: DA V2 (× FloorScale) and Qwen3-VL-2B on those frames afterwards; the
  walk-in self-motion test (box growth vs a reflection and a robot standing still); raw and mirror-filled point clouds.
- `tools/isaac/phantom_room.py`: path-traced dollhouse orbit with DA's mirror depth as a point cloud (the phantom
  room) folding onto the glass under `fill_from_border`.
- `foresight.viz.mirror_test_video`, `tools/make_mirror_test_video.py`: narrated 1920×1920 video
  (`results/isaac/foresight_mirror_test.mp4`); every number computed from the run files.

### Results (one scripted encounter, not an evaluation; `docs/tools/isaac.md`)
- DA V2 reads the mirror too far on all 751 frames (+0.9 to +1.5 m, median +1.3 m).
- Qwen3-VL-2B P(mirror) 0.91 beyond 2.1 m, 0.37 within 1.5 m (71% of looks below 0.5); sees a robot (P 0.97) but
  never judges it to be itself (P(self) ≤ 0.38).
- Walk-in test: Qwen's robot box grew ×3.0 from 2.3 m to 0.8 m from the glass; reflection predicts ×2.7, a robot
  standing still ×1.5.
- The mirror fill lands on the glass while the mirror's base is in view; within ~1.5 m it lands ~0.8 m behind it.

## [Unreleased] — B results, detector ablation, mirror depth fill, FORESIGHT v3 (2026-11-13)

### Results (`docs/results/foresight_B.md`)
- B evaluation of v2.3 (E 33, S 64, A 120 episodes, paired with the map navigator on Depth Anything V2): success
  E 48% vs 64% (crash 15% vs 25%), S 67% vs 71% (crash 0% vs 4%), A 70% vs 79% (p = 0.039); pooled −25/+10,
  p = 0.017. Losses are timeouts from false walls on the route; 3.4 slow checks per episode outside encounters.
- Detector ablation on E: with an oracle mirror detector the v2.3 guard succeeds in 15% (0% crashes, 85% timeouts):
  its walls, not the VLM, are the bottleneck. Re-filling the mirror's depth from the depth below it on the oracle
  mask gives 88% / 8% crashes, on the oracle box 85% / 4% (map alone 65% / 27%).
- FORESIGHT v3 (VLM boxes re-fill the depth) on E: 62% / 31% crashes, no better than the map navigator; all crashes
  with the view > 90% mirror, where Qwen3-VL-2B answers "no mirror". A first v3 run never filled (0.6 s box-age limit
  vs 1.5–2.7 s grounding looks); kept as `E/foresight_fill_bug_noop`, not a result.

### Added
- `perception/depth/sources.py`: `MirrorFillDepth` / `fill_from_border` and depth specs `fill:da@reflective`,
  `fill:da@reflective_box` (column-wise re-fill of mirror regions from their lower/upper border, DA's own values).
- `planning/fast_slow.py:BoxFillGuard`, `run_pointnav_eval.py --guard fill --fill-hold-s`: FORESIGHT v3, VLM mirror
  boxes re-fill the depth (held 3 s, shifted by the robot's turn); no slow check, no walls.
- `FastSlowMirrorGuard(edge_extend_m=)` / `--guard v24`: walls from boxes at the image border extended; no effect
  with the oracle on E_dev, kept off.
- `run_detector_ablation.py`: `--guard v23|v24`, `--max-sim-s`; `scripts/run_det_ablation.sh`.

## [Unreleased] — Literature update, B evaluation, detector ablation (2026-11-12)

### Added
- `docs/lit_review/update_2026-11-12.md`: second Athena pass (MCP tools) on the questions the A/C/v2.3 results raise.
  Main finding: *From Depth to Safe Depth* (arXiv 2611.13769, 10 Nov 2026) is concurrent work on deliverable A
  (Nav2 + Depth Anything V2 collides in 63%; trained correction head + conformal uncertainty 70% / 3%); it never tries
  per-scene scale calibration, which FloorScale is. Also MIRA-Bench (VLMs can't tell reflections from objects),
  slow-module scheduling as a decision problem, and a LiDAR-only literature on mirrors in navigation maps.
- `docs/results/depth_error_A.md`: concurrent-work section with Safe Depth's near-range over-estimation metric on our
  set-A traces: Depth Anything V2 98% raw, 11% with FloorScale (theirs: 59.8% → 12.0% with a trained head).
- `scripts/run_b_eval.sh`: deliverable B, FORESIGHT v2.3 on E, S, A (resumable).
- `tools/run_detector_ablation.py`: the v2.3 guard driven by a simulated mirror detector (semantic boxes, controlled
  recall / false-box rate, always-confirming slow check), to bound what any mirror detector can buy.

## [Unreleased] — FORESIGHT v2.3: wedge-only mirror walls (2026-11-11)

### Added
- Per-wall guard log (`FastSlowMirrorGuard.wall_log`: Laya P, mirror-box P, candidate kind; on annotated homes the
  distance to the nearest mirror/glass object) and `tools/guard_precision.py` (wall precision by confidence cut, by
  candidate kind, and encounter outcomes against the map navigator).
- `FastSlowMirrorGuard(gap_candidates=False)`, `run_pointnav_eval.py --guard v21|v22|v23`; dev encounter set
  `episodes_E_9001.json` (12, seed 9001, dev only); `scripts/run_guard_dev.sh`, `scripts/run_guard_dev_v23.sh`.

### Results (`docs/src/foresight/planning/fast_slow.md`, v2.3 section)
- v2.2 walls on real mirror/glass: 20/45 (44%); confidence cuts don't separate them (41–60%), candidate kind does
  (depth-gap walls 6/26, wedge walls 14/19).
- v2.3 (no depth-gap candidates) on dev: 21/33 walls (64%) on real mirrors; `A_3001` 57% / 0% crash vs map 61% / 4%
  (p = 1) and v2.2 54%; dev encounters 50% / 10% crash vs map 40% / 40% and v2.2 33% / 11%. v2.3 is the configuration
  for the B evaluation run.

## [Unreleased] — Robust scan, mirror encounters, FORESIGHT v2.2 (2026-11-10)

### Added
- `local_map.depth_scan(robust=True)` / `MapNavigator(robust_scan=)` / `run_pointnav_eval.py --robust-scan`:
  median-filtered depth and a supported nearest point per column, against the phantom obstacles that zero-mean depth
  noise made in the plain scan.
- Encounter set E (`tools/make_encounter_episodes.py`, `episodes_E_3005.json`, 33 encounters, committed before any run):
  start 1.5–4.5 m facing a mirror/glass pane, goal straight through it. Deliverable C's mirror/glass stratum.
- `FastSlowMirrorGuard` v2.2 options `near_m`, `walled_m`, `gap_s` (off by default; `run_pointnav_eval.py --guard v22`,
  the point-goal default) against over-checking; wall-precision diagnostic (`walls_on_reflective`) on annotated homes.
- `scripts/run_fix_sweep.sh`, `scripts/run_b_dev.sh`.

### Fixed
- `REFLECTIVE_CATEGORIES` matched names exactly and missed "closet mirror wall" (00808, 7, incl. the hallway mirror),
  "exhibition window" and "door window" (00803). Deliverable A's reflective cells on set S re-run with the fixed list
  (old runs kept as `results/runs/pointnav/S_oldmask_*`).

### Results (`docs/results/depth_error_A.md`, `docs/src/foresight/planning/fast_slow.md`)
- Robust scan, set A: 10% / 20% noise 5% / 0% → 81% / 75% success (GT 84%), no cost elsewhere on set A; correlated
  noise partly fixed; on mirror encounters it crashes more (DA: 41% vs 25%).
- Corrected reflective masks, set S: same aggregate conclusion (map navigator 86–89% at × 1.5–3), now with 2–5% crashes.
- Mirror encounters (set E): GT 82% / 0% crashes; Depth Anything 64% / 25%; DA with only its mirror/glass pixels
  fixed 89% / 0%; mirror read × 2 / × 3 too deep: 43% / 75% crashes. H-A's structured half holds where mirrors are
  approached.
- FORESIGHT v2.2 on dev: 58% / 0% crashes vs the navigator alone 58% / 4% and v2.1 38%; slow checks 9.3 → 5.2 per
  episode. Open: only 3 of 19 confirmed walls lie on real mirrors/glass.

## [Unreleased] — Showcase video (2026-11-08)

### Added
- `tools/make_showcase_video.py`, `viz/showcase_video.py` and `viz/showcase_assets.py` build a narrated square
  1920x1920 project overview for LinkedIn, modelled on the Habitat platform promo. It runs about 3:24.
  - Segments: logo, how it works, HM3D fly-throughs and a 3D floor-plan mosaic, point-goal and object-search clips,
    "same robot, different depth" (set-A episode `a-00800-00` with GT / DA-V2 / ×2.5 scale / 10% noise), the mirror
    encounter, the set-A result bars, and what's next.
  - New assets: photographic cutaways of all 13 HM3D homes and tilted "dollhouse" floor plans extruded from them.
  - Docs: `docs/tools/make_showcase_video.md`.

## [Unreleased] — Deliverable A: depth-error propagation (2026-11-02 – 2026-11-04)

### Added
- Point-goal evaluation: depth sources (`perception/depth/sources.py`), `tools/run_pointnav_eval.py`,
  `tools/summarize_pointnav.py`, `eval/stats.py`, `tools/make_pointnav_episodes.py` (sets A 3003 / S 3004 + dev sets),
  `scripts/run_depth_error_sweep.sh`, `scripts/stop_sweep_at.sh`, `tools/plot_depth_error.py`.
- `MapNavigator`: `floor_scale` switch, opt-in escape manoeuvre, `reflection_segments`/`through_wall` moved down from
  `ObjectSearchNavigator`. `timed_episode`: DepthSource, semantic sensor, frame_stats, on_end.

### Results
- `docs/results/depth_error_A.md`: 5,600 runs. DA-V2 vs ground truth, map navigator: 79% vs 83% (10 homes, n.s.),
  71% vs 89% (4 annotated homes, p = 0.006). Scale error harmless with floor calibration, catastrophic without; 10%
  zero-mean noise collapses the map navigator to 5%; injected reflective error has no aggregate effect on random
  episodes.

### Changed
- `.gitignore`: `results/` anchored to the repo root so `docs/results/` is tracked.

## [Unreleased] — Literature review and next-phase plan (2026-11-02)

### Added
- `PLAN.md`: next phase, closing the spec's deliverables A → C → B on point-goal; revised after the lit review.
- `docs/lit_review/`: six Athena research maps (VLN, zero-shot ObjectNav, monocular metric depth, mirror/glass/
  transparent perception, embodied-navigation robustness, dual-system VLAs). Field evolution, open-problem themes,
  closest prior work (Athena + web), positioning of each contribution; Athena's milestone and reading-path exports.

### Changed (plan)
- Deliverable A sharpened into hypothesis H-A (uniform vs structured depth error), with NavTrust's depth corruptions
  added for comparability. Deliverable C narrowed to mirror/glass and textureless as primary axes (low light and
  motion blur are already covered by RobustNav/NavTrust). B framed against Slow Brain/Fast Planner, AsyncVLA and
  TIC-VLA.

### Fixed
- Isaac replay: ground physics material (friction 1.0, restitution 0) to match the Go2 walking policy's training.

## FORESIGHT v2 / v2.1 and the Isaac Sim replay (2026-11-01)

### Added
- `planning/fast_slow.py` `FastSlowMirrorGuard` (FORESIGHT v2): yes/no mirror check + mirror box every look; the slow
  Qwen scene note + Laya only when a candidate mirror wall (box + depth gap) lies on the route or heading; forward
  motion vetoed until checked. Runner condition `foresight_fs`.
- v2.1: candidates "known" only if they lie along an existing wall/dismissed candidate; "wedge" candidates from
  confident mirror boxes without gap geometry.
- Close-in arrival (`ObjectSearchNavigator(close_in=True)`), `MapNavigator.step(creep_mps=…)`, approach patience.
- `PerceptionWorker(describe_when=…)`: on-demand, describe-only looks.
- Second and third evaluation sets (`episodes_2027.json`, `episodes_2028.json`, `--seed/--set fresh`), each committed
  before any run on it; `run_object_search.py --episodes fresh|set3`, `HF_HUB_OFFLINE=1` default.
- Isaac Sim replay: `tools/export_hm3d_for_isaac.py`, `tools/isaac/export_run.py`, `tools/isaac/render_replay.py`,
  `viz/isaac_replay.py`; docs `docs/tools/isaac.md`.
- `viz/object_search_v2_video.py`, `run_object_search.py --phase render_v2`.

### Changed
- `ForesightNavigator` (v1): off-path flags stay pending 10 s and are checked if the route or heading turns into them.
- `ObjectSearchNavigator.reflection_segments` split from `add_reflection_wall`; walls joined to a much nearer "frame"
  surface (an occluder) use the farther depth for both ends.

### Results
- Second set (16 episodes): found within 1 m — baseline 5, v1 7, v2 7; SPL 0.17 / 0.23 / 0.30; looks 0.34 / 2.6 /
  0.5 s; crashes 0 / 1 / 1 (v2's into a closet mirror, again on the demo start).
- Third set (15 scored): baseline 8 found, 2 mirror crashes; v2.1 6 found, 0 crashes, 6 timeouts.
- Details: `docs/src/foresight/planning/object_search.md`, `docs/src/foresight/planning/fast_slow.md`.

### Fixed
- A slanted mirror wall across the 00808 bedroom doorway (frame depth fix); an approach that pushed at a blocked target
  for 214 s (approach patience); a hidden "confirmed" event when a new suspect started on the same tick.

### Removed
- 3.4 GB of stale mirror-trap pickles (`results/runs/mirror_trap_demo/hm3d-hallway-mirror/*.pkl`, untracked).

## Object search demo ("find the sofa") — 2026-10-30

### Added
- `CHANGELOG.md`.
- Language-goal object search, `planning/object_search.py`: frontier exploration on the self-built map,
  `PerceptionWorker` (one background VLM worker per look: optional scene note, yes/no presence gate, grounding,
  direction hint), `ObjectMemory` (3 sightings + more hits than misses; looks that should have seen an object
  but didn't count against it), close-up verification before declaring "found", grounded mirror walls
  (`add_reflection_wall` spans only the boxed mirror, and only over a too-deep gap), sightings through a
  reflection wall ignored, later instructions served from memory.
- `Qwen3VLSceneReasoner.locate` (grounding, 0-1000 boxes), `option_probs` (one-forward-pass answer
  probabilities), `GROUNDING_PROMPT`, `PRESENT_PROMPT`, `DIRECTION_PROMPT`.
- `local_map.cost_grid`, `frontiers`, `distances_from`; `OccupancyMap.add_segment`.
- `sim/objectnav_eval.py`: ObjectNav-style scoring (success within 1 m of an annotated instance, SPL via
  multi-goal shortest path) from HM3D semantics.
- `tools/run_object_search.py` (run / eval / assets / render; `--oracle` harness with label-based grounding),
  `tools/make_object_search_episodes.py` (seeded evaluation set), `viz/object_search_video.py`.
- Probes in `results/runs/object_search/`: grounding (single-category recall 0.8-1.0 at ~0.3 s; multi-category
  prompts hallucinate every listed class), direction hint (52% vs 33% chance on 50 poses), presence yes/no.

### Changed
- `make_sim_config` loads the semantic mesh only when a semantic sensor is requested (saves ~0.6 GB GPU; needed
  to fit Qwen3-VL + Depth Anything + Laya + renderer on 8 GB next to a browser).
- `ForesightNavigator`: `in_path_only` (skip double-checks on flagged views the route doesn't enter), no
  re-check of views already resolved without a wall, falls back to the first flagged view when the confirming
  view is too close to show a gap.
- `MapNavigator`: `replan_look`, `plan_kw`, `last_path`; `BackgroundDescriber`: `min_latency_s`, `_deliver` hook;
  `timed_episode`: `episode_done` diagnostic ends a run, `oracle_semantic` for harnesses.

### Results
- Evaluation of the fixed set (15 scored, 1 voided for a scan hole): found within 1 m — baseline 2, FORESIGHT 0;
  crashes 5 vs 1 (all baseline crashes at the 00808 hallway mirror). Demo episode: both complete sofa → bed,
  FORESIGHT 190 s / SPL 0.42, 0.38 vs 244 s / 0.17, 0.18. Details: `docs/src/foresight/planning/object_search.md`.
- Narrated 1920×1920 video, 3:47 (`tools/run_object_search.py --phase render`): instruction hook, mirror lure, side-by-side search with each robot's revealed map, how-it-works diagram, scoreboard.

### Fixed
- Scoring: a "crash" whose ground-truth depth ahead is ~0 is a hole in the scanned mesh, reported as
  `ended_by_scan_hole` and voided for every condition.
- HM3D semantic `aabb` is stretched to the world origin in habitat-sim 0.3.3; scoring uses `obb`, which is only
  populated when the semantic mesh is loaded together with an RGB sensor.

## 2026-10-30

### Added
- Out-of-process Laya decision worker (`tools/laya_worker.py`) and client (`planning/laya_client.py`) — `c136e0c`.
- Latency-honest mirror-trap demo — `e2023c8`: `sim/timed_episode.py`, `planning/laya_fusion.py`,
  `planning/local_map.py`, `planning/map_navigator.py`, `viz/mirror_trap_video.py`,
  `tools/run_mirror_trap_demo.py`, `tools/kokoro_tts.py`; reflection-aware Qwen3-VL prompt;
  `ReactiveSafetyLayer(clearance_quantile=...)`.

## 2026-10-19 and earlier

See `git log` (Habitat setup, sensor capture, Depth Anything V2, Qwen3-VL, action fusion, depth-only
baseline, PointNav RL, vlm_memory experiments).
