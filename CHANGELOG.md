# Changelog

Notable changes to FORESIGHT, newest first. Day-by-day working notes live in `logs/` (untracked);
milestone status in `PROGRESS.md`.

## [Unreleased] — object search demo ("find the sofa")

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

### Fixed
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
