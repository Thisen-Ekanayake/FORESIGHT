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

# The Mirror Test demo (2026-11-19)

A Unitree H1 humanoid (Isaac Sim's `H1FlatTerrainPolicy`) walks up to the full-height hallway mirror of 00808
(mirror115/116, the plane x ≈ -0.1 in Isaac's frame) and moves in front of it, path traced, so its head camera sees its
own reflection, something Habitat can't show (no robot body, baked mirrors). Scripted, not a closed loop: the models run
afterwards on the saved frames, because Isaac and the models don't fit in 8 GB together.

| Step | Env | What it does |
|---|---|---|
| `tools/isaac/house_stage.py` | Isaac | Shared stage building (house meshes, mirror material, lights, floor collider, cameras), with a scan-texture emission scale (0.25 here: the 2026-11-01 replay's 1.0 flattened the lighting) and dollhouse cuts (height cut, footprint box). |
| `tools/isaac/mirror_test.py` | Isaac | The choreography (approach from 2.5 m, look, turn ±28°, side-step ±0.35 m, walk in to 0.4 m where the mirror fills the view). Saves the chase view, the 640×640 90° level head camera (1.52 m, near Habitat's 1.5 m), Isaac's ground-truth depth and the mirror mask (instance segmentation) per frame. |
| `tools/isaac/mirror_test_perception.py` | habitat | DA V2 (×FloorScale at the head height) vs ground truth; every 3rd frame Qwen3-VL-2B: P(mirror) with FORESIGHT's yes/no prompt, P(robot), P("that robot is me, in a mirror"), grounding boxes for robot and mirror, a description per phase. The self-motion test: the image position a reflection would have vs a second robot standing still, from the camera poses, against Qwen's robot box. Back-projects DA depth (raw and mirror-filled with `fill_from_border` on the oracle mask) into `phantom.npz`. |
| `tools/isaac/phantom_room.py` | Isaac | Path-traced orbit of a cut-away of the room with the point cloud: the mirror pixels as DA reads them (a room behind the wall, red), then sliding to the filled depth (cyan). |
| `tools/make_mirror_test_video.py` | habitat | Narrated 1920×1920 video (`foresight.viz.mirror_test_video`); every spoken number is computed from the files above. |

```bash
/ml/isaac-sim/python.sh tools/isaac/mirror_test.py --out results/isaac/mirror_test --renderer PathTracing --spp 32
# the walk-up again for the chase view only (the side camera starts inside a wall before ~5.6 s)
/ml/isaac-sim/python.sh tools/isaac/mirror_test.py --out results/isaac/mirror_test_corner --renderer PathTracing \
    --spp 32 --chase corner --t-to 5.7
$PY tools/isaac/mirror_test_perception.py results/isaac/mirror_test
/ml/isaac-sim/python.sh tools/isaac/phantom_room.py --run results/isaac/mirror_test --out results/isaac/phantom
$PY tools/make_mirror_test_video.py --run results/isaac/mirror_test --phantom results/isaac/phantom \
    --out results/isaac/foresight_mirror_test.mp4
```

Rendered 2026-11-19: 751 frames of the choreography (H1 tracking error median 12 cm), 191 corner-camera frames of the
walk-up (7 cm), a 456-frame Phantom Room orbit (24 fps, 1.5-7 s/frame as the points appear), and the video:
`results/isaac/foresight_mirror_test.mp4`, 2:13, 1920×1920, Kokoro am_michael voice-over.

Findings and limits (one scripted encounter, not an evaluation):
- DA V2 reads the mirror too far on every frame (+0.9 to +1.5 m, median +1.3 m).
- Qwen3-VL-2B: P(mirror) 0.91 beyond 2.1 m, 0.37 within 1.5 m; P(robot) 0.97; "is that robot me, in a mirror?"
  never above 0.38. Close up it describes the reflection as "a black sculpture with 'Unitree' written on it".
- Walk-in test: Qwen's robot box grows ×3.0 from 2.3 m to 0.8 m from the glass; a reflection predicts ×2.7, a robot
  standing still ×1.5 (median error 5% vs 19%). Turning is no position cue: under pure rotation a reflection and a
  still object shift alike.
- The mirror fill (oracle mask) lands within ~0.15 m of the glass while the mirror's base is in view; within ~1.5 m
  the mirror reaches the image bottom and the fill lands ~0.8 m behind it. The Phantom Room uses frames 0–100 only.
- The H1 policy barely side-steps (planned ±0.35 m, got 11 cm): no lateral test. The perception frames come from the
  main render; the walk-up chase view from a second run of the same script (same choreography, a few cm apart).
- Isaac can hang on exit holding ~5 GB of GPU memory: kill the `kit/python` process by PID before the next job.

Cost on the RTX 4060 laptop: path tracing at 32 spp, 1920×1080 chase + 640×640 head camera, ~4.3 s per frame
(675 frames for the 22.5 s choreography). Real-time mode (RaytracedLighting) renders the reflected robot as a black
silhouette (reflections get no lighting there); path tracing lights it.
