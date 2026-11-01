# Isaac Sim replay (`tools/export_hm3d_for_isaac.py`, `tools/isaac/`)

Replays a recorded FORESIGHT run in NVIDIA Isaac Sim 6.0.1 (`/ml/isaac-sim`): the same HM3D home, RTX-rendered, with
a Unitree Go2 walking the recorded route. The navigation decisions were all made in Habitat-Sim
([`run_object_search.py`](run_object_search.md)); Isaac Sim adds a photoreal render, real mirror reflections and a
physically simulated legged body, not new decisions.

| Step | Env | What it does |
|---|---|---|
| `tools/export_hm3d_for_isaac.py` | habitat | HM3D's GLB textures are Basis-compressed (`image/x-basis`), which Isaac's importer can't read and no `basisu` tool is installed; Magnum's `BasisImporter` (bundled with habitat-sim) transcodes them to RGBA8. Writes `meshes.npz` (positions, UVs, indices, texture id, node name per mesh) and `tex_NNN.jpg`. No download needed. |
| `tools/isaac/export_run.py` | habitat | A run pickle → JSON (per tick: time, position, heading, mode, mirror-check state; once: walls, finds, events), because Isaac's Python can't import habitat or foresight. |
| `tools/isaac/render_replay.py` | Isaac (`/ml/isaac-sim/python.sh`) | Builds the stage (one `UsdGeom.Mesh` per GLB mesh, `UsdPreviewSurface` with the scan texture as diffuse + emission, the 18 HM3D mirror meshes as a perfect mirror material, a dome light and warm ceiling lights, an invisible flat floor collider), spawns `Go2FlatTerrainPolicy`, steers it along the recorded route, renders a chase camera and a head camera. |
| `foresight.viz.isaac_replay` | habitat | Composes 1920×1920 frames (chase view, head camera, the run's state at that time, route minimap) for the video. |
| `tools/isaac/make_clip.py` | habitat | A standalone silent clip of a replay at constant speed. |

```bash
PY=/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python
$PY tools/export_hm3d_for_isaac.py --scene 00808-y9hTuugGdiq --out results/isaac/00808
$PY tools/isaac/export_run.py results/runs/object_search/fresh/mirror-sofa-bed/foresight_fs.pkl results/isaac/v2_run.json
/ml/isaac-sim/python.sh tools/isaac/render_replay.py --run results/isaac/v2_run.json --house results/isaac/00808 \
    --out results/isaac/replay_v2
/ml/isaac-sim/python.sh tools/isaac/render_replay.py --run results/isaac/v2_run.json --out results/isaac/check --pose-check
$PY tools/isaac/make_clip.py results/isaac/replay_v2 results/isaac/v2_run.json \
    results/runs/object_search/set3/mirror-sofa-bed results/isaac/foresight_isaac_replay.mp4 --speed 2
```

Rendered for the v2 video (2026-11-01): the v2.1 demo run's first 160 s (mirror check, sofa found, bed rejected),
5,334 frames at 0.04 s/frame on the RTX 4060 (real-time RTX mode), Go2 tracking error median 4.7 cm, max 11.5 cm.

Nothing else may hold the GPU while Isaac Sim runs (8 GB).

## Frames and checks

- HM3D GLBs are stored z-up (the dataset config's stage `up` is `[0, 0, 1]`; Habitat rotates them to y-up on load), so the
  raw mesh coordinates are already Isaac's z-up frame: Habitat `(x, y, z)` = Isaac `(x, -z, y)`. An earlier version
  rotated the house a second time and put the camera outside it.
- `--pose-check` renders the head camera at the run's start pose with Habitat's camera height (1.5 m) and 90° view:
  it matched Habitat's start frame, except that the hallway "doorway" is now a real mirror reflecting the room behind
  the robot (in the scan it is a textured phantom hallway).
- The Go2 is driven by a tracking controller on body-frame velocity commands (feed-forward from the recorded route +
  proportional correction, yaw from the recorded heading); smoke test: median tracking error 5-9 cm, max 20 cm.
  The scan mesh is visual only (its floor is uneven), physics uses a flat floor at the scan's floor height.
- Chase camera: on the robot's own earlier route, at the furthest point from which it walked a straight line to where
  it is now (free line of sight by construction; a point merely on the route looked through a door after a turn).
  When there is none (the opening spin), a point it walks to later in a straight line, looking back.

## Limits

- A replay: the Go2's body plays no part in the decisions, and the head camera is at Go2 height (~0.4 m), not the
  1.5 m camera the run was decided with.
- Scan artifacts (holes, stretched texels) show more under RTX lighting than in Habitat's flat shading.
