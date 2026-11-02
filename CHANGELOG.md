# Changelog

Notable changes to FORESIGHT, newest first. Day-by-day working notes live in `logs/` (untracked);
milestone status in `PROGRESS.md`.

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
