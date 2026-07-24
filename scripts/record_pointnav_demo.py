#!/usr/bin/env python
"""Roll out a trained continuous-control PointGoal policy and record a smooth demo clip + eval metrics.

Loads a habitat-baselines checkpoint, runs the policy on a held-out (val) episode, and renders the demo_1
side-by-side layout (first-person left, top-down map right) reusing src/foresight/viz/render.py — the RL
trajectory is the traced path, the geodesic-optimal route is drawn faintly for reference, red marker = goal.
Motion is smoothed the same way as demo_1: keyframe poses are interpolated (lerp + slerp) and re-rendered.

  DISPLAY=:1 <py> scripts/record_pointnav_demo.py --ckpt results/runs/pointnav/checkpoints/latest.pth
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import foresight.rl  # noqa: F401  registers the reward measure + continuous velocity action

import habitat
import habitat_sim
from habitat.config import read_write
from habitat.gym.gym_wrapper import continuous_vector_action_to_hab_dict, create_action_space
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat_baselines.utils.common import batch_obs

from foresight.sim.navigate import _resample_track
from foresight.viz.render import compose_frame, render_topdown_panel

DISCRETE_ACTIONS = ["stop", "move_forward", "turn_left", "turn_right"]


def load_policy(ckpt_path, observation_space, action_space, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    policy_cls = baseline_registry.get_policy(config.habitat_baselines.rl.policy.main_agent.name)
    policy = policy_cls.from_config(config, observation_space, action_space)
    # This habitat-baselines version saves the actor-critic state_dict directly (keys net.*/critic.*/
    # action_distribution.*); older ones prefix with "actor_critic." — strip it if present.
    state = {
        (k[len("actor_critic."):] if k.startswith("actor_critic.") else k): v
        for k, v in ckpt["state_dict"].items()
    }
    policy.load_state_dict(state)
    policy.to(device).eval()
    hidden_size = config.habitat_baselines.rl.ppo.hidden_size
    return policy, hidden_size


def geodesic_path(pathfinder, start, goal):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(start, np.float32)
    sp.requested_end = np.asarray(goal, np.float32)
    pathfinder.find_path(sp)
    return np.asarray(sp.points, np.float32)


def rollout(env, policy, hidden_size, device, deterministic, max_steps):
    """Run the policy for one episode; return keyframe positions, rotations, and end-of-episode metrics."""
    orig_action_space = env.action_space
    flat_space = create_action_space(orig_action_space)
    num_actions = flat_space.shape[0]

    obs = env.reset()
    hidden = torch.zeros(1, policy.net.num_recurrent_layers, hidden_size, device=device)
    prev_actions = torch.zeros(1, num_actions, device=device)
    not_done = torch.zeros(1, 1, dtype=torch.bool, device=device)

    positions = [np.asarray(env.sim.get_agent_state().position, np.float32)]
    rotations = [env.sim.get_agent_state().rotation]
    for _ in range(max_steps):
        batch = batch_obs([obs], device=device)
        with torch.no_grad():
            ad = policy.act(batch, hidden, prev_actions, not_done, deterministic=deterministic)
        hidden = ad.rnn_hidden_states
        prev_actions.copy_(ad.actions)
        not_done.fill_(True)
        act_vec = np.asarray(ad.env_actions[0].cpu().numpy(), np.float32)
        obs = env.step(continuous_vector_action_to_hab_dict(orig_action_space, flat_space, act_vec))
        st = env.sim.get_agent_state()
        positions.append(np.asarray(st.position, np.float32))
        rotations.append(st.rotation)
        if env.episode_over:
            break
    return positions, rotations, env.get_metrics()


def encode_video(frames_dir, out_path, fps):
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%05d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="habitat-baselines checkpoint (.pth)")
    p.add_argument("--config-name", default="rl/pointnav_continuous.yaml")
    p.add_argument("--configs-dir", default="experiments/configs")
    p.add_argument("--modality", choices=["rgb", "depth", "rgbd"], default="rgb")
    p.add_argument("--split", default="val")
    p.add_argument("--episode-index", type=int, default=0, help="Which episode in the split to record")
    p.add_argument("--out", default="results/runs/pointnav/demo")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of taking the mean")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration", type=float, default=0.0,
                   help="If >0, interpolate the trajectory to duration*fps smooth frames (like demo_1)")
    p.add_argument("--panel", type=int, default=720)
    p.add_argument("--mpp", type=float, default=0.05)
    p.add_argument("--keep-frames", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_config(args.config_name, configs_dir=args.configs_dir)
    with read_write(cfg):
        cfg.habitat.dataset.split = args.split
        for a in DISCRETE_ACTIONS:
            cfg.habitat.task.actions.pop(a, None)
        sensors = cfg.habitat.simulator.agents.main_agent.sim_sensors
        if args.modality == "rgb":
            sensors.pop("depth_sensor", None)
        elif args.modality == "depth":
            sensors.pop("rgb_sensor", None)

    env = habitat.Env(config=cfg.habitat)
    policy, hidden_size = load_policy(args.ckpt, env.observation_space, create_action_space(env.action_space), device)

    # Advance the episode iterator to the requested index.
    for _ in range(args.episode_index):
        env.reset()
    positions, rotations, metrics = rollout(
        env, policy, hidden_size, device, deterministic=not args.stochastic, max_steps=args.max_steps
    )
    episode = env.current_episode
    start = np.asarray(episode.start_position, np.float32)
    goal = np.asarray(episode.goals[0].position, np.float32)
    planned = geodesic_path(env.sim.pathfinder, start, goal)
    topdown = np.asarray(env.sim.pathfinder.get_topdown_view(args.mpp, float(start[1])), dtype=bool)
    bmin, _ = env.sim.pathfinder.get_bounds()
    bmin = np.asarray(bmin, np.float32)

    success = float(metrics.get("success", 0.0))
    spl = float(metrics.get("spl", 0.0))
    collisions = int(metrics.get("collisions", {}).get("count", 0)) if isinstance(metrics.get("collisions"), dict) else 0
    print(f"episode {episode.episode_id}: success={success:.0f} spl={spl:.3f} collisions={collisions} "
          f"steps={len(positions) - 1} geodesic={metrics.get('distance_to_goal'):.2f}m-to-goal-at-end")

    # Smooth (optional): interpolate keyframe poses and re-render RGB at each, like demo_1.
    if args.duration > 0:
        n_frames = round(args.duration * args.fps)
        frame_pos, frame_rot = _resample_track(positions, rotations, n_frames)
    else:
        frame_pos, frame_rot = positions, rotations

    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    n = len(frame_pos)
    print(f"rendering {n} frames -> {frames_dir}")
    for i, (pos, rot) in enumerate(zip(frame_pos, frame_rot)):
        obs = env.sim.get_observations_at(pos, rot, keep_agent_at_new_pose=False)
        rgb = np.asarray(obs["rgb"][..., :3], dtype=np.uint8)
        panel = render_topdown_panel(
            topdown=topdown, bounds_min=bmin, mpp=args.mpp,
            planned_path=planned, traveled=np.stack(frame_pos[: i + 1]),
            start=start, goal=goal, panel_w=args.panel, panel_h=args.panel,
        )
        frame = compose_frame(rgb, panel, "First-person RGB",
                              "Top-down map  ·  RL PointGoal policy (blue) vs geodesic-optimal")
        frame.save(frames_dir / f"frame_{i:05d}.png")

    out_video = out_dir / f"pointnav_{episode.episode_id}.mp4"
    encode_video(frames_dir, out_video, args.fps)
    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    (out_dir / "metrics.json").write_text(json.dumps(
        {"episode_id": episode.episode_id, "success": success, "spl": spl,
         "collisions": collisions, "steps": len(positions) - 1}, indent=2))
    env.close()
    print(f"Done: {out_video} ({n} frames, {n / args.fps:.1f}s @ {args.fps} fps)")


if __name__ == "__main__":
    main()
