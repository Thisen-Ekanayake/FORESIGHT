# `tools/make_showcase_video.py`

Builds the FORESIGHT showcase: a narrated, square 1920x1920 overview of the whole project for LinkedIn. Its structure
follows the Habitat platform promo: a logo open, the question, how the system works, captioned fly-throughs of the
HM3D homes and a mosaic of tilted 3D floor plans, a task montage, the depth-error finding, the mirror encounter, the
set-A results, what's next and a close. The video is about 3:24 long.

Rendering lives in [`foresight.viz.showcase_video`](../../src/foresight/viz/showcase_video.py). The cutaways,
dollhouse floor plans and fly-through paths come from [`foresight.viz.showcase_assets`](../../src/foresight/viz/showcase_assets.py).
The voice-over is Kokoro-82M, run through `tools/kokoro_tts.py` (default voice `am_michael`).

## Usage

```bash
P=/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python
$P tools/make_showcase_video.py --phase record   # 4 point-goal runs, ~1 min (DA on the GPU)
$P tools/make_showcase_video.py --phase assets   # 13 cutaways + 4 fly-through paths, ~1 min
$P tools/make_showcase_video.py --phase render   # the video, ~1 h on the RTX 4060
$P tools/make_showcase_video.py --phase render --preview 115   # one frame at video second 115 -> PNG
```

Every phase skips outputs that already exist. Narration is cached in `<out>/narration/` and keyed on the exact text.
The render phase prints the segment timeline: start, length, and when each voice line is cued.

## Phases

- **record**: re-runs set-A episode `a-00800-00` with the map navigator. It uses each of four depth sources: `gt`,
  `da`, `gt:scale=2.5` and `gt:noise=0.1`. The loop, crash rule, floor calibration and noise seed are the same as
  `tools/run_pointnav_eval.py`. Each run also keeps what the evaluation drops: the full per-tick log, the depth the
  navigator was given (160 px, every tick), its occupancy map and its planned route. All four runs reproduce the
  sweep:
  - `gt`, `da` and `gt:scale=2.5` each arrive at 43.7 s.
  - `gt:noise=0.1` times out at 122.7 s and never gets more than 0.31 m from its start.
- **assets**: renders a photographic cutaway of every home (10 minival + 3 example). An orthographic camera 2 m above
  the main floor looks straight down, so ceilings are culled. The phase also computes four smooth fly-through paths.
  Walls in the dollhouse plans are extruded from the cutaway's dark gaps, plus a rim around the house. Habitat's
  orthographic depth sensor returns only 0/1, so the cutaway can't supply a height map.
- **render**: re-renders every robot view in Habitat at the pose interpolated to each frame's sim time. On screen,
  playback speed is badged whenever it isn't 1×.

## Sources of what's shown

| Segment | Data |
|---|---|
| How it works | Real frames from the `gt` run; the Qwen3-VL note and P(mirror) from the mirror-trap run's first flag |
| Point-goal | `gt` run, sim 8–44 s at 2.5× |
| Object search | `results/runs/object_search/eval/mirror-sofa-bed/foresight.pkl`, sim 117–137.5 s at 3× and 160–189.5 s at 4× |
| Same robot, different depth | the four recorded runs |
| Mirror encounter | `results/runs/mirror_trap_demo_v2/hm3d-hallway-mirror/{baseline,laya}.pkl` (one staged encounter, labelled as a demo) |
| Results | `results/runs/pointnav/A/summary.json` (map navigator), and S's DA vs GT in the footnote |

## Output: `results/runs/showcase/`

| File | Contents |
|---|---|
| `foresight_showcase.mp4` | the video (H.264 CRF 18, AAC voice) |
| `pointnav/*.pkl` | the four recorded runs |
| `assets/cut_<home>.{png,json}`, `assets/fly_<home>.npz` | cutaways and fly-through paths |
| `narration/` | Kokoro clips and their durations |
