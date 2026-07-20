# `src/foresight/sim/sensors.py`

RGB + ground-truth depth + semantic sensor capture from a single Habitat-Sim agent pose. Library code — no file I/O, no CLI; consumed by [`scripts/capture_sensors.py`](../../../scripts/capture_sensors.md).

## `Observation`

```python
@dataclass
class Observation:
    rgb: np.ndarray       # (H, W, 3) uint8
    depth: np.ndarray     # (H, W) float32, meters, simulator ground truth
    semantic: np.ndarray  # (H, W) uint32, per-pixel object instance ID
```

One camera frame's worth of sensor output. `depth` is exact simulator ground truth, not an estimate — this is what the depth-error-propagation study (`PROGRESS.md` deliverable A) injects noise against. `semantic` is `0` everywhere if the loaded scene has no semantic annotations.

## `make_sim_config(dataset_config, scene_id, width, height) -> habitat_sim.Configuration`

Builds a `habitat_sim.Configuration` with one agent carrying three co-located `CameraSensorSpec`s (`rgb`/COLOR, `depth`/DEPTH, `semantic`/SEMANTIC), all at eye height (`position = [0, 1.5, 0]`) so the three sensors share one optical center and are pixel-aligned by construction.

- `dataset_config` — path to a `*.scene_dataset_config.json` (e.g. the HM3D minival one under `data/processed/scene_datasets/hm3d/minival/`).
- `scene_id` — scene ID as declared in that config, e.g. `00800-TEEsavR23oF`.
- `width`, `height` — sensor resolution. Habitat's `CameraSensorSpec.resolution` takes `[height, width]` (numpy/image convention, not `(x, y)`) — verified empirically against the installed `habitat_sim` 0.3.3 (non-square 480×640 capture came back with shape `(480, 640)`, matching this ordering).

## `capture_scene_observation(dataset_config, scene_id, width=640, height=480, seed=0) -> Observation`

1. Builds the config via `make_sim_config` and constructs `habitat_sim.Simulator(cfg)`.
2. Seeds `sim.pathfinder` and spawns the agent at `sim.pathfinder.get_random_navigable_point()` — a uniformly random point on the scene's navmesh, not a fixed pose. Same `seed` reproduces the same point for a given scene/navmesh.
3. Calls `sim.get_sensor_observations()` once (single frame, agent does not move) and packs the three channels into an `Observation`.
4. Always closes the simulator (`try`/`finally`), even on error.

Raises whatever `habitat_sim` raises if `dataset_config`/`scene_id` don't resolve, or if the scene has no navmesh (`get_random_navigable_point` needs one — all HM3D `.basis` scenes ship with one).

## Gotchas

- Doesn't check whether the scene has semantic annotations loaded — call site decides whether to warn (see `SEMANTIC_SCENES` in the CLI script).
- One observation per call; each call spins up and tears down a full `Simulator`. Fine for one-off captures — if this gets reused for batch capture across many scenes/poses, hoist the `Simulator` construction out of the per-frame loop instead.
