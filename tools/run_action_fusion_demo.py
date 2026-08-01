#!/usr/bin/env python
"""Record a demo of the action fusion controller (PROGRESS.md milestone #6) driving an agent
closed-loop: zero-shot depth + a Qwen3-VL-2B heading proposal (budgeted 0.5-2 Hz) fused with the
goal bearing every control tick, wrapped by a per-tick reactive safety layer. Same side-by-side
first-person / top-down layout as demo_1, via src/foresight/viz/render.py.

No smoothing/interpolation here (unlike demo_1/demo_2) — each rendered frame is a real control
tick, so the clip plays back at (roughly) the achieved control rate, honestly showing how fast
the stack actually runs on this hardware (see Project_FORESIGHT.pdf's "report the operating
point actually achieved" note).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foresight.sim.action_fusion_episode import run_action_fusion_episode
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
        "-preset", "slow",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--scene", default="00800-TEEsavR23oF")
    parser.add_argument("--out", default="results/runs/action_fusion_demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640, help="First-person capture width")
    parser.add_argument("--height", type=int, default=480, help="First-person capture height")
    parser.add_argument("--playback-fps", type=int, default=2, help="Video fps (1 frame = 1 control tick)")
    parser.add_argument("--mpp", type=float, default=0.05, help="Top-down metres per pixel")
    parser.add_argument("--panel", type=int, default=700, help="Top-down panel size and body height, px")
    parser.add_argument("--min-path-dist", type=float, default=4.0, help="Min geodesic start->goal distance, m")
    parser.add_argument("--goal-radius", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=30, help="Max control ticks")
    parser.add_argument("--time-step", type=float, default=1.0, help="Kinematic integration step, s")
    parser.add_argument("--vlm-hz", type=float, default=1.0, help="VLM heading-query budget, 0.5-2 Hz")
    parser.add_argument("--max-linear-mps", type=float, default=0.25)
    parser.add_argument("--max-angular-degps", type=float, default=10.0)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.scene
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running action-fusion episode in {args.scene} (seed {args.seed})...")
    t0 = time.monotonic()
    ep = run_action_fusion_episode(
        dataset_config=args.dataset_config,
        scene_id=args.scene,
        width=args.width,
        height=args.height,
        seed=args.seed,
        min_path_dist=args.min_path_dist,
        goal_radius=args.goal_radius,
        max_steps=args.max_steps,
        time_step=args.time_step,
        vlm_hz=args.vlm_hz,
        max_linear_mps=args.max_linear_mps,
        max_angular_degps=args.max_angular_degps,
    )
    wall_s = time.monotonic() - t0
    n_ticks = len(ep.steps)
    vlm_queries = sum(1 for s in ep.steps if s["vlm_queried"])
    achieved_hz = n_ticks / wall_s if wall_s > 0 else float("nan")
    print(
        f"  {n_ticks} control ticks | reached goal: {ep.success} | collisions: {ep.collided_steps} | "
        f"planned geodesic {ep.geodesic_distance:.2f} m | wall time {wall_s:.1f}s "
        f"(achieved {achieved_hz:.2f} Hz, {vlm_queries} VLM queries)"
    )

    print(f"Compositing {len(ep.rgb_frames)} frames -> {frames_dir}")
    for i, rgb in enumerate(ep.rgb_frames):
        traveled = ep.positions[: i + 2]
        step = ep.steps[i]
        panel = render_topdown_panel(
            topdown=ep.topdown,
            bounds_min=ep.bounds_min,
            mpp=ep.meters_per_pixel,
            planned_path=ep.planned_path,
            traveled=traveled,
            start=ep.start,
            goal=ep.goal,
            panel_w=args.panel,
            panel_h=args.panel,
        )
        frame = compose_frame(
            rgb=rgb,
            topdown_panel=panel,
            left_caption=f"First-person RGB  ·  tick {i}  ·  {step['safety_level']}",
            right_caption=f"Action fusion  ·  vlm_queried={step['vlm_queried']}  ·  "
                          f"clearance={step['min_clearance_m']:.2f}m",
        )
        frame.save(frames_dir / f"frame_{i:05d}.png")

    out_video = out_dir / "action_fusion_demo.mp4"
    if ep.rgb_frames:
        print(f"Encoding -> {out_video}")
        encode_video(frames_dir, out_video, args.playback_fps)
    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    metrics = {
        "scene": args.scene,
        "success": ep.success,
        "collided_steps": ep.collided_steps,
        "control_ticks": n_ticks,
        "geodesic_distance_m": ep.geodesic_distance,
        "final_dist_to_goal_m": ep.steps[-1]["dist_to_goal_m"] if ep.steps else None,
        "wall_time_s": wall_s,
        "achieved_hz": achieved_hz,
        "vlm_queries": vlm_queries,
        "vlm_hz_budget": args.vlm_hz,
        "steps": ep.steps,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Done: {out_video if ep.rgb_frames else '(no frames)'}; metrics -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
