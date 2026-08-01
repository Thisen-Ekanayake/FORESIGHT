#!/usr/bin/env python
"""Record the Demo-1 navigation clip: an agent walking the navmesh shortest path through an HM3D house,
first-person feed on the left, top-down floor plan with the traced path on the right (see docs/demos/demo_1.md).

No trained policy — the agent follows Habitat's built-in greedy geodesic follower. Once the VLM planner and
depth-only baseline exist, re-run with those action sources on the same episode for the real comparison.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foresight.sim.navigate import run_shortest_path_episode
from foresight.viz.render import compose_frame, render_topdown_panel

DEFAULT_DATASET_CONFIG = (
    "data/processed/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json"
)


def encode_video(frames_dir: Path, out_path: Path, fps: int):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot encode the video")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-preset", "slow",       # better compression/quality at these frame counts
        "-crf", "17",            # visually lossless
        "-pix_fmt", "yuv420p",   # broad player/LinkedIn compatibility
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--scene", default="00800-TEEsavR23oF", help="Scene ID, e.g. 00800-TEEsavR23oF")
    parser.add_argument("--out", default="results/runs/demo_1", help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920, help="First-person capture width")
    parser.add_argument("--height", type=int, default=1080, help="First-person capture height")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Target clip length in seconds at normal speed; frames = duration * fps "
                             "(the discrete path is interpolated to hit it). 0 = one frame per follower step.")
    parser.add_argument("--mpp", type=float, default=0.05, help="Top-down metres per pixel")
    parser.add_argument("--panel", type=int, default=900, help="Top-down panel size and body height, px")
    parser.add_argument("--min-path-dist", type=float, default=10, help="Min geodesic start->goal distance, m")
    parser.add_argument("--goal-radius", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--keep-frames", action="store_true", help="Keep the intermediate PNG frames")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.scene
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    n_frames = round(args.duration * args.fps) if args.duration else None

    print(f"Running shortest-path episode in {args.scene} (seed {args.seed})...")
    ep = run_shortest_path_episode(
        dataset_config=args.dataset_config,
        scene_id=args.scene,
        width=args.width,
        height=args.height,
        seed=args.seed,
        min_path_dist=args.min_path_dist,
        goal_radius=args.goal_radius,
        max_steps=args.max_steps,
        meters_per_pixel=args.mpp,
        n_frames=n_frames,
    )
    n = len(ep.rgb_frames)
    print(
        f"  {ep.n_keyframes} follower steps -> {n} smooth frames | "
        f"planned geodesic {ep.geodesic_distance:.2f} m | reached goal: {ep.success}"
    )

    print(f"Compositing {n} frames -> {frames_dir}")
    for i in range(n):
        panel = render_topdown_panel(
            topdown=ep.topdown,
            bounds_min=ep.bounds_min,
            mpp=ep.meters_per_pixel,
            planned_path=ep.planned_path,
            traveled=ep.positions[: i + 1],
            start=ep.start,
            goal=ep.goal,
            panel_w=args.panel,
            panel_h=args.panel,
        )
        frame = compose_frame(
            rgb=ep.rgb_frames[i],
            topdown_panel=panel,
            left_caption="First-person RGB",
            right_caption="Top-down map  ·  shortest path (navmesh)",
        )
        frame.save(frames_dir / f"frame_{i:05d}.png")

    out_video = out_dir / "demo_1.mp4"
    print(f"Encoding -> {out_video}")
    encode_video(frames_dir, out_video, args.fps)

    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    dur = n / args.fps
    print(f"Done: {out_video} ({n} frames, {dur:.1f}s @ {args.fps} fps)")


if __name__ == "__main__":
    main()
