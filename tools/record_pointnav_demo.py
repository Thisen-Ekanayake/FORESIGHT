#!/usr/bin/env python
"""Roll out a trained continuous-control PointGoal policy and record a smooth demo clip + eval metrics.

Loads a habitat-baselines checkpoint, runs the policy on a held-out (val) episode, and renders the demo_1
side-by-side layout (first-person left, top-down map right) reusing src/foresight/viz/render.py — the RL
trajectory is the traced path, the geodesic-optimal route is drawn faintly for reference, red marker = goal.
Motion is smoothed the same way as demo_1: keyframe poses are interpolated (lerp + slerp) and re-rendered.

Every value lives in experiments/configs/rl/record_demo.yaml — override with `--set KEY=VALUE`.

  DISPLAY=:1 <py> tools/record_pointnav_demo.py --ckpt results/runs/pointnav/checkpoints/latest.pth \
      --set episode_index=0 --set render.duration=15 --set render.fps=50
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
from foresight.rl.config import choice, get, load_config, override

import habitat
import habitat_sim
from habitat.config import read_write
from habitat.gym.gym_wrapper import continuous_vector_action_to_hab_dict, create_action_space
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat_baselines.utils.common import batch_obs

from foresight.sim.navigate import _resample_track
from foresight.viz.render import compose_frame, render_topdown_panel

# Only literal in this script: where its config lives (relative to the repo root).
SCRIPT_CONFIG = "experiments/configs/rl/record_demo.yaml"


def load_policy(ckpt_path, observation_space, action_space, device, state_dict_prefix):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    policy_cls = baseline_registry.get_policy(config.habitat_baselines.rl.policy.main_agent.name)
    policy = policy_cls.from_config(config, observation_space, action_space)
    # This habitat-baselines version saves the actor-critic state_dict directly (keys net.*/critic.*/
    # action_distribution.*); older ones prefix with state_dict_prefix — strip it if present.
    state = {
        (k[len(state_dict_prefix):] if k.startswith(state_dict_prefix) else k): v
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


def encode_video(frames_dir, input_pattern, out_path, fps, ffmpeg):
    cmd = [ffmpeg["binary"], "-y", "-framerate", str(fps), "-i", str(frames_dir / input_pattern),
           "-c:v", ffmpeg["codec"], "-preset", ffmpeg["preset"], "-crf", str(ffmpeg["crf"]),
           "-pix_fmt", ffmpeg["pix_fmt"], "-movflags", ffmpeg["movflags"], str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=SCRIPT_CONFIG, help="Script config (YAML) holding all defaults")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override script-config keys, e.g. render.duration=15 render.fps=50")
    p.add_argument("--ckpt", default=None, help="habitat-baselines checkpoint (.pth)")
    p.add_argument("--modality", default=None, help="Visual input (see `modalities` in the config)")
    p.add_argument("--split", default=None)
    p.add_argument("--episode-index", type=int, default=None, help="Which episode in the split to record")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    override(cfg, "ckpt", args.ckpt)
    override(cfg, "modality", args.modality)
    override(cfg, "split", args.split)
    override(cfg, "episode_index", args.episode_index)
    override(cfg, "out", args.out)

    modality = choice(cfg, "modality", get(cfg, "modality"), get(cfg, "modalities"))
    render_cfg = get(cfg, "render")
    fps = render_cfg["fps"]
    mpp = render_cfg["meters_per_pixel"]

    device_cfg = get(cfg, "device")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_cfg == "auto" else device_cfg
    )

    habitat_cfg = get(cfg, "habitat_config")
    hab = get_config(habitat_cfg["config_name"], configs_dir=habitat_cfg["configs_dir"])
    with read_write(hab):
        hab.habitat.dataset.split = get(cfg, "split")
        for action in get(cfg, "discrete_actions"):
            hab.habitat.task.actions.pop(action, None)
        sensors = hab.habitat.simulator.agents.main_agent.sim_sensors
        for sensor in get(cfg, "modalities")[modality]:
            sensors.pop(sensor, None)

    env = habitat.Env(config=hab.habitat)
    policy, hidden_size = load_policy(
        get(cfg, "ckpt"), env.observation_space, create_action_space(env.action_space),
        device, get(cfg, "state_dict_prefix"),
    )

    # Advance the episode iterator to the requested index.
    for _ in range(get(cfg, "episode_index")):
        env.reset()
    positions, rotations, metrics = rollout(
        env, policy, hidden_size, device,
        deterministic=get(cfg, "rollout.deterministic"), max_steps=get(cfg, "rollout.max_steps"),
    )
    episode = env.current_episode
    start = np.asarray(episode.start_position, np.float32)
    goal = np.asarray(episode.goals[0].position, np.float32)
    planned = geodesic_path(env.sim.pathfinder, start, goal)
    topdown = np.asarray(env.sim.pathfinder.get_topdown_view(mpp, float(start[1])), dtype=bool)
    bmin, _ = env.sim.pathfinder.get_bounds()
    bmin = np.asarray(bmin, np.float32)

    success = float(metrics.get("success", 0.0))
    spl = float(metrics.get("spl", 0.0))
    collisions = int(metrics.get("collisions", {}).get("count", 0)) if isinstance(metrics.get("collisions"), dict) else 0
    print(f"episode {episode.episode_id}: success={success:.0f} spl={spl:.3f} collisions={collisions} "
          f"steps={len(positions) - 1} geodesic={metrics.get('distance_to_goal'):.2f}m-to-goal-at-end")

    # Smooth (optional): interpolate keyframe poses and re-render RGB at each, like demo_1.
    duration = render_cfg["duration"]
    if duration > 0:
        n_frames = round(duration * fps)
        frame_pos, frame_rot = _resample_track(positions, rotations, n_frames)
    else:
        frame_pos, frame_rot = positions, rotations

    out_dir = Path(get(cfg, "out"))
    frames_dir = out_dir / render_cfg["frames_dirname"]
    frames_dir.mkdir(parents=True, exist_ok=True)

    n = len(frame_pos)
    print(f"rendering {n} frames -> {frames_dir}")
    for i, (pos, rot) in enumerate(zip(frame_pos, frame_rot)):
        obs = env.sim.get_observations_at(pos, rot, keep_agent_at_new_pose=False)
        rgb = np.asarray(obs["rgb"][..., :3], dtype=np.uint8)
        panel = render_topdown_panel(
            topdown=topdown, bounds_min=bmin, mpp=mpp,
            planned_path=planned, traveled=np.stack(frame_pos[: i + 1]),
            start=start, goal=goal, panel_w=render_cfg["panel_w"], panel_h=render_cfg["panel_h"],
        )
        frame = compose_frame(rgb, panel, render_cfg["left_label"], render_cfg["right_label"])
        frame.save(frames_dir / render_cfg["frame_name_format"].format(index=i))

    out_video = out_dir / get(cfg, "video.filename_format").format(episode_id=episode.episode_id)
    encode_video(frames_dir, get(cfg, "video.input_pattern"), out_video, fps, get(cfg, "video.ffmpeg"))
    if not render_cfg["keep_frames"]:
        shutil.rmtree(frames_dir)

    (out_dir / get(cfg, "metrics_file")).write_text(json.dumps(
        {"episode_id": episode.episode_id, "success": success, "spl": spl,
         "collisions": collisions, "steps": len(positions) - 1}, indent=2))
    env.close()
    print(f"Done: {out_video} ({n} frames, {n / fps:.1f}s @ {fps} fps)")


if __name__ == "__main__":
    main()
