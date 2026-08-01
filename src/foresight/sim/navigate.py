"""Drive a Habitat-Sim agent along the navmesh shortest path between two points, capturing first-person RGB.

No learned policy is involved: the agent follows Habitat's built-in greedy geodesic follower over the
navmesh (milestone #2), which is the honest stand-in for a trained navigator until the VLM planner (#5, #6)
and depth-only baseline (#7) exist. Once they do, swap the action source and the same recording layout
yields the real planner-vs-baseline comparison.

Motion is produced in two phases so the output is smooth at high fps:
  1. Plan  — run the discrete follower (move 0.25 m / turn 10 deg per action), collecting the keyframe
             poses only, via `agent.act` (no rendering, so it's cheap).
  2. Render — resample those keyframes into a continuous pose track (lerp position + slerp rotation) of an
             arbitrary frame count, and render RGB at each. This decouples the smoothness/duration of the
             clip from the follower's coarse action granularity.
"""
from dataclasses import dataclass, field

import habitat_sim
import numpy as np
import quaternion  # provided by habitat_sim's dependency (numpy-quaternion)

from foresight.sim.sensors import make_sim_config

try:
    from habitat_sim.errors import GreedyFollowerError
except Exception:  # pragma: no cover - fallback if the symbol moves between versions
    GreedyFollowerError = RuntimeError


@dataclass
class Episode:
    """Everything needed to render the demo, decoupled from the (now-closed) simulator."""

    rgb_frames: list  # list of (H, W, 3) uint8 first-person frames, one per rendered (possibly interpolated) frame
    positions: np.ndarray  # (T, 3) float32 world (x, y, z) of the camera at each rendered frame
    start: np.ndarray  # (3,) float32 world start
    goal: np.ndarray  # (3,) float32 world goal
    planned_path: np.ndarray  # (N, 3) float32 geodesic waypoints from start to goal
    geodesic_distance: float
    success: bool  # follower reached within goal_radius before max_steps
    n_keyframes: int  # discrete follower steps before interpolation (for reporting)
    # Navmesh raster + georeferencing, so the top-down view is a pure function of arrays afterwards.
    topdown: np.ndarray  # (rows, cols) bool, True = navigable; row indexes z, col indexes x
    bounds_min: np.ndarray  # (3,) float32 navmesh AABB min
    bounds_max: np.ndarray  # (3,) float32 navmesh AABB max
    meters_per_pixel: float = field(default=0.05)


def _sample_start_goal(pathfinder, min_dist: float, max_tries: int):
    """Sample two navigable, geodesically connected points at least `min_dist` apart along the navmesh.

    Returns (start, goal, path_points, geodesic_distance). Uses the pathfinder's own RNG, so seed it first.
    """
    for _ in range(max_tries):
        start = pathfinder.get_random_navigable_point()
        goal = pathfinder.get_random_navigable_point()

        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = goal
        if (
            pathfinder.find_path(path)
            and np.isfinite(path.geodesic_distance)
            and path.geodesic_distance >= min_dist
        ):
            return (
                np.asarray(start, np.float32),
                np.asarray(goal, np.float32),
                np.asarray(path.points, np.float32),
                float(path.geodesic_distance),
            )
    raise RuntimeError(
        f"could not sample a start/goal pair at least {min_dist} m apart after {max_tries} tries "
        "(scene navmesh may be small or fragmented into disconnected islands)"
    )


def _plan_poses(follower, agent, goal, max_steps):
    """Walk the discrete follower to the goal, returning the list of keyframe poses (position, rotation).

    Uses `agent.act` (no sensor render) so planning is cheap; the frames are rendered later at interpolated poses.
    """
    positions = [np.asarray(agent.get_state().position, np.float32)]
    rotations = [agent.get_state().rotation]
    success = False
    for _ in range(max_steps):
        try:
            action = follower.next_action_along(goal)
        except GreedyFollowerError:
            break
        if action is None or action == "stop":
            success = True
            break
        agent.act(action)
        st = agent.get_state()
        positions.append(np.asarray(st.position, np.float32))
        rotations.append(st.rotation)
    return positions, rotations, success


def _resample_track(positions, rotations, n_frames):
    """Resample discrete keyframe poses into `n_frames` smooth poses (linear position, slerp rotation).

    Frames are spread uniformly across keyframe index, so translation and rotation each play at a constant
    cadence -> constant-velocity, jitter-free motion.
    """
    k = len(positions)
    if k == 1 or n_frames <= 1:
        return [positions[0]] * max(n_frames, 1), [rotations[0]] * max(n_frames, 1)

    out_pos, out_rot = [], []
    for j in range(n_frames):
        u = j * (k - 1) / (n_frames - 1)
        i0 = min(int(np.floor(u)), k - 2)
        frac = u - i0
        out_pos.append((1.0 - frac) * positions[i0] + frac * positions[i0 + 1])
        out_rot.append(quaternion.slerp_evaluate(rotations[i0], rotations[i0 + 1], frac))
    return out_pos, out_rot


def run_shortest_path_episode(
    dataset_config: str,
    scene_id: str,
    width: int = 640,
    height: int = 480,
    seed: int = 0,
    min_path_dist: float = 5.0,
    goal_radius: float = 0.25,
    max_steps: int = 500,
    meters_per_pixel: float = 0.05,
    max_sample_tries: int = 200,
    n_frames: int = None,
) -> Episode:
    """Sample a start/goal, walk the navmesh shortest path, and render first-person RGB.

    If `n_frames` is given, the discrete path is interpolated to exactly that many smooth frames
    (set it to duration_seconds * fps for a clip of a target length). Otherwise one frame per follower step.
    """
    # RGB-only config: the demo doesn't use the depth/semantic passes, and skipping them speeds up rendering.
    cfg = make_sim_config(dataset_config, scene_id, width, height, sensor_uuids=("rgb",))
    sim = habitat_sim.Simulator(cfg)
    try:
        sim.pathfinder.seed(seed)
        agent = sim.get_agent(0)

        start, goal, planned_path, geo_dist = _sample_start_goal(
            sim.pathfinder, min_path_dist, max_sample_tries
        )

        state = habitat_sim.AgentState()
        state.position = start
        agent.set_state(state)

        follower = habitat_sim.nav.GreedyGeodesicFollower(
            sim.pathfinder,
            agent,
            goal_radius=goal_radius,
            stop_key="stop",
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )

        # Phase 1: plan the discrete keyframe poses (cheap, no rendering).
        key_pos, key_rot, success = _plan_poses(follower, agent, goal, max_steps)
        n_keyframes = len(key_pos)

        # Phase 2: resample to a smooth pose track of the requested length.
        target = n_frames if n_frames else n_keyframes
        frame_pos, frame_rot = _resample_track(key_pos, key_rot, target)

        # Phase 3: render RGB at each pose.
        rgb_frames, positions = [], []
        render_state = habitat_sim.AgentState()
        for pos, rot in zip(frame_pos, frame_rot):
            render_state.position = pos
            render_state.rotation = rot
            agent.set_state(render_state)
            obs = sim.get_sensor_observations()
            rgb_frames.append(np.array(obs["rgb"][..., :3], dtype=np.uint8))
            positions.append(np.asarray(pos, np.float32))

        # Navmesh raster sliced at the agent's floor height, for the top-down panel.
        bmin, bmax = sim.pathfinder.get_bounds()
        topdown = sim.pathfinder.get_topdown_view(meters_per_pixel, float(start[1]))

        return Episode(
            rgb_frames=rgb_frames,
            positions=np.stack(positions).astype(np.float32),
            start=start,
            goal=goal,
            planned_path=planned_path,
            geodesic_distance=geo_dist,
            success=success,
            n_keyframes=n_keyframes,
            topdown=np.asarray(topdown, dtype=bool),
            bounds_min=np.asarray(bmin, np.float32),
            bounds_max=np.asarray(bmax, np.float32),
            meters_per_pixel=meters_per_pixel,
        )
    finally:
        sim.close()
