"""Latency-honest closed-loop episode: simulated time advances by what each decision actually cost.

`closed_loop_episode.run_closed_loop_episode` integrates a fixed `time_step` per tick, so the world
stands still while models think and a 2 s VLM call costs the agent nothing. Here every control tick
advances sim time by max(control_period_s, measured compute of that tick), so a slow decision means
the agent travels further on its previous command before the next one lands.

Work a controller runs "in the background" (e.g. a VLM describing the scene at ~0.5 Hz, see
`foresight.planning.laya_fusion.BackgroundDescriber`) is computed synchronously for simplicity but
reported via `diagnostics["background_ms"]`; that time is excluded from the tick and instead
delivered `latency` sim-seconds later. This models a dedicated parallel worker — it does not model
GPU contention between the two, which a real single-GPU robot would pay.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import habitat_sim
import numpy as np
import quaternion
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image

from foresight.perception.depth.depth_anything import ZeroShotDepthEstimator
from foresight.sim.pose import goal_bearing_rad
from foresight.sim.sensors import make_sim_config
from foresight.sim.velocity_step import integrate_velocity_step

# rgb, depth (m), goal_bearing_rad, sim_time_s, pose {"position", "forward_xz"} (odometry)
#   -> (linear m/s, angular rad/s, diagnostics)
TimedDecideFn = Callable[[np.ndarray, np.ndarray, float, float, dict], tuple]


@dataclass
class TimedEpisode:
    times: np.ndarray  # (T,) sim time at which each tick's frame was observed, s
    rgb_frames: list  # (H, W, 3) uint8 per tick
    positions: np.ndarray  # (T+1, 3), includes the start pose
    collided: np.ndarray  # (T,) bool
    steps: list  # per-tick diagnostics, including tick_compute_ms / depth_ms
    start: np.ndarray
    goal: np.ndarray
    topdown: Optional[np.ndarray] = None  # (rows, cols) bool navmesh raster at the start height
    bounds_min: Optional[np.ndarray] = None
    meters_per_pixel: float = 0.05
    reached_goal_at: Optional[float] = None  # sim time, when stop_radius_m was given and reached
    contact_at: Optional[float] = None  # sim time the camera came within stop_on_contact_m of a surface


def run_timed_episode(
    dataset_config: str,
    scene_id: str,
    decide: TimedDecideFn,
    depth_estimator: ZeroShotDepthEstimator,
    start_position: np.ndarray,
    start_yaw_rad: float,
    goal_position: np.ndarray,
    max_sim_s: float = 12.0,
    control_period_s: float = 0.1,
    width: int = 640,
    height: int = 480,
    model_size: Optional[tuple] = None,
    meters_per_pixel: float = 0.05,
    stop_radius_m: Optional[float] = None,
    stop_on_contact_m: Optional[float] = None,
    keep_frames: bool = True,
    oracle_semantic: bool = False,
) -> TimedEpisode:
    """width/height: sensor resolution (what the stored frames are, e.g. for a video). model_size:
    optional (w, h) the frame is resized to before the depth model and `decide` see it, so a
    high-resolution render doesn't change model latency. stop_radius_m: end the episode once the agent is
    this close to the goal (horizontal distance); otherwise it runs for max_sim_s. stop_on_contact_m: end
    it (as a crash) once the sim-GT depth straight ahead drops below this while the agent is driving forward
    (turning in place next to a wall is not a crash) — the same rule for every agent. keep_frames=False
    drops the per-tick RGB frames (a 90 s run at 1024x1024 is ~2.8 GB) when the poses are enough, e.g. to
    re-render the views for a video. oracle_semantic: test harnesses only — adds the sim's instance-id image as
    pose["oracle_semantic"] (and the simulator as pose["oracle_sim"]) so a scripted stand-in can replace a model."""
    uuids = ("rgb", "depth", "semantic") if oracle_semantic else ("rgb", "depth")
    sim = habitat_sim.Simulator(make_sim_config(dataset_config, scene_id, width, height, sensor_uuids=uuids))
    try:
        agent = sim.get_agent(0)
        start = np.asarray(sim.pathfinder.snap_point(start_position), np.float32)
        goal = np.asarray(goal_position, np.float32)
        state = habitat_sim.AgentState()
        state.position = start
        state.rotation = quat_from_angle_axis(start_yaw_rad, np.array([0.0, 1.0, 0.0]))
        agent.set_state(state)

        times, frames, positions, collided, steps = [], [], [np.asarray(start, np.float32)], [], []
        t, reached, contact = 0.0, None, None
        while t < max_sim_s:
            agent_state = agent.get_state()
            to_goal = goal - np.asarray(agent_state.position)
            if stop_radius_m is not None and np.hypot(to_goal[0], to_goal[2]) < stop_radius_m:
                reached = t
                break
            fwd = quaternion.rotate_vectors(agent_state.rotation, np.array([0.0, 0.0, -1.0]))
            obs = sim.get_sensor_observations()
            rgb = np.array(obs["rgb"][..., :3], dtype=np.uint8)
            model_rgb = rgb if model_size is None else np.asarray(Image.fromarray(rgb).resize(model_size, Image.BILINEAR))
            # Simulator ground-truth depth: logged for evaluation/overlays only, never passed to `decide`.
            h, w = obs["depth"].shape
            gt_ahead = float(np.median(obs["depth"][int(0.4 * h):int(0.9 * h), int(0.25 * w):int(0.75 * w)]))
            bearing = goal_bearing_rad(agent_state.position, agent_state.rotation, goal)

            t0 = time.perf_counter()
            depth = depth_estimator.estimate(model_rgb)  # returns a CPU array, so this is GPU-synchronised
            depth_ms = (time.perf_counter() - t0) * 1000
            pose = {"position": np.asarray(agent_state.position, float), "forward_xz": [float(fwd[0]), float(fwd[2])]}
            if oracle_semantic:
                pose["oracle_semantic"], pose["oracle_sim"] = obs["semantic"], sim
            linear, angular, diagnostics = decide(model_rgb, depth, bearing, t, pose)
            wall_ms = (time.perf_counter() - t0) * 1000
            tick_ms = wall_ms - diagnostics.get("background_ms", 0.0)
            dt = max(control_period_s, tick_ms / 1000)

            hit = integrate_velocity_step(sim, agent, linear, angular, dt)
            times.append(t)
            if keep_frames:
                frames.append(rgb)
            positions.append(np.asarray(agent.get_state().position, np.float32))
            collided.append(hit)
            steps.append({"t": t, "dt": dt, "tick_compute_ms": tick_ms, "depth_ms": depth_ms,
                          "goal_bearing_rad": bearing, "forward_xz": [float(fwd[0]), float(fwd[2])], "gt_depth_ahead_m": gt_ahead, "linear_velocity": linear,
                          "angular_velocity": angular, "collided": hit, **diagnostics})
            t += dt
            if diagnostics.get("episode_done"):  # the agent declared its task finished (or gave up)
                break
            if stop_on_contact_m is not None and gt_ahead < stop_on_contact_m and linear > 0.02:
                contact = t
                break

        topdown = np.asarray(sim.pathfinder.get_topdown_view(meters_per_pixel, float(start[1])), dtype=bool)
        bounds_min = np.asarray(sim.pathfinder.get_bounds()[0], np.float32)
        return TimedEpisode(np.asarray(times), frames, np.stack(positions), np.asarray(collided), steps, start, goal,
                            topdown, bounds_min, meters_per_pixel, reached, contact)
    finally:
        sim.close()
