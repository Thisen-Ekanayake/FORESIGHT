"""FORESIGHT drives a Unitree G1 live in Isaac Sim (Isaac Lab): pass 1 of the "Through the Glass" demo.

The G1 walks with NVIDIA's pretrained Isaac Lab policy (Isaac-Velocity-Flat-G1, rsl_rl), in its exact training
configuration (Isaac Lab's own env cfg). Every decision comes from FORESIGHT (tools/isaac/foresight_brain.py,
habitat env, over a local socket): each tick the world renders the G1's head camera (640x640, 90 deg, stabilized
level at head height), sends it with the pose, and walks the returned (forward, turn) velocity command for
max(0.1 s, the tick's measured compute) before the next one, the point-goal evaluation's timing. The world is paused
while the models run.

Records everything pass 2 (the path-traced cinematic re-render) needs: root pose + all joint positions every 20 ms
physics control step, the camera pose and the brain's answer every tick.

  # terminal 1 (habitat env):   python tools/isaac/foresight_brain.py
  # terminal 2 (Isaac Lab env): /ml/IsaacLab/env_isaaclab/bin/python tools/isaac/g1_live.py \
  #       --scene office --start 1.0 2.0 90 --goal 6.0 2.0 --cond foresight --out results/isaac/live/office_fs
  # --probe: no brain; the G1 walks a fixed command (forward/turn) and saves head-camera stills (walking check).

Coordinates: Isaac world, z up, metres; yaw in degrees (0 = +x). The brain works in Habitat's frame
(y up, forward -z): Habitat (x, y, z) = Isaac (x, z, -y).
"""
import argparse
import json
import math
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--assets", default=os.path.join(ROOT, "isaac_assets"), help="local mirror of the asset root")
parser.add_argument("--scene", default="office", choices=["office", "hospital", "warehouse", "dtw", "hm3d", "none"])
parser.add_argument("--start", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "YAW_DEG"))
parser.add_argument("--goal", nargs=2, type=float, default=[4.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--cond", default="foresight", choices=["map", "foresight"], help="map = depth-only (the ghost)")
parser.add_argument("--labels", nargs="*", default=["glass wall", "glass door", "mirror"],
                    help="what FORESIGHT's guard grounds and fills (foresight only)")
parser.add_argument("--max-sim-s", type=float, default=90.0)
parser.add_argument("--success-m", type=float, default=0.5)
parser.add_argument("--contact-m", type=float, default=0.3, help="crash rule: GT depth ahead below this while walking")
parser.add_argument("--after-contact-s", type=float, default=2.0, help="keep simulating after a crash (the bump)")
parser.add_argument("--cam-height", type=float, default=1.22, help="head camera height over the floor, m")
parser.add_argument("--max-linear", type=float, default=0.45)
parser.add_argument("--out", required=True)
parser.add_argument("--probe", nargs="*", type=float, default=None, metavar="V W SECONDS",
                    help="no brain: walk v m/s, w rad/s for SECONDS (default 0.5 0.0 6) and save stills")
parser.add_argument("--port", type=int, default=47011)
parser.add_argument("--realtime-render", action="store_true", help="RTX real-time for the head camera (default)")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import wire  # noqa: E402

A = os.path.abspath(args.assets)
SCENES = {  # usd, (x, y, z) offset applied to the scene
    "office": (f"{A}/Isaac/Environments/Office/office.usd", (0, 0, 0)),
    "hospital": (f"{A}/Isaac/Environments/Hospital/hospital.usd", (0, 0, 0)),
    "warehouse": (f"{A}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd", (0, 0, 0)),
    "dtw": (f"{A}/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd", (0, 0, 0)),
    "hm3d": (os.path.join(ROOT, "results/isaac/00808_collision.usd"), (0, 0, 0)),
}
TASK = "Isaac-Velocity-Flat-G1-Play-v0"
CKPT = f"{A}/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt"
os.makedirs(args.out, exist_ok=True)


def yaw_quat_xyzw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


# ---------------------------------------------------------------- env config: Isaac Lab's own G1 flat (play) env
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 0.0
cfg.scene.robot.spawn.usd_path = f"{A}/Isaac/IsaacLab/Robots/Unitree/G1/g1.usd"
x0, y0, yaw0 = args.start[0], args.start[1], math.radians(args.start[2])
cfg.scene.robot.init_state.pos = (x0, y0, 0.74)
cfg.scene.robot.init_state.rot = yaw_quat_xyzw(yaw0)
# deterministic start: no random base pose / joint offsets / pushes, no resets
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
for name in ("push_robot", "base_external_force_torque"):
    if hasattr(cfg.events, name):
        setattr(cfg.events, name, None)
cfg.terminations.time_out = None
cfg.terminations.base_contact = None
cfg.episode_length_s = 1.0e6
cmd = cfg.commands.base_velocity
cmd.resampling_time_range = (1.0e6, 1.0e6)
cmd.heading_command = False
cmd.rel_standing_envs = 0.0
cmd.rel_heading_envs = 0.0
cmd.debug_vis = False
cfg.observations.policy.enable_corruption = False
cfg.sim.render_interval = cfg.decimation  # render once per control step
# the scene: NVIDIA's USD (with its own colliders); the terrain plane stays as the physics floor
if args.scene != "none":
    usd, off = SCENES[args.scene]
    cfg.scene.stage_set = AssetBaseCfg(prim_path="/World/Set", spawn=sim_utils.UsdFileCfg(usd_path=usd),
                                       init_state=AssetBaseCfg.InitialStateCfg(pos=off))
# head camera: a level, stabilized camera at head height (FORESIGHT assumes a level pinhole camera)
cfg.scene.head_cam = CameraCfg(
    prim_path="/World/HeadCam", update_period=0.0, height=640, width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=10.5, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)))

env = ManagerBasedRLEnv(cfg)
if args.scene != "none":  # hide the terrain plane's visual where a scene provides its own floor
    from pxr import UsdGeom
    stage = env.sim.stage
    for p in stage.Traverse():
        if str(p.GetPath()).startswith("/World/ground") and p.IsA(UsdGeom.Gprim):
            UsdGeom.Imageable(p).MakeInvisible()
venv = RslRlVecEnvWrapper(env)
agent_cfg = load_cfg_from_registry(TASK.replace("-Play", ""), "rsl_rl_cfg_entry_point")
runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
runner.load(CKPT)
policy = runner.get_inference_policy(device="cuda:0")

robot = env.scene["robot"]
cam = env.scene["head_cam"]
term = env.command_manager.get_term("base_velocity")
dt_ctrl = env.step_dt  # 0.02 s
print(f"G1: {len(robot.joint_names)} joints, control dt {dt_ctrl:.3f} s, scene {args.scene}", flush=True)


def root_pose():
    p = robot.data.root_link_pose_w.torch[0].cpu().numpy().astype(float)  # x y z qx qy qz qw
    qx, qy, qz, qw = p[3:]
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return p[:3], yaw, p[3:]


log = {"args": vars(args), "joint_names": list(robot.joint_names), "dt": dt_ctrl, "steps": [], "ticks": []}
t_sim = [0.0]


def control_step(v, w, obs):
    term.vel_command_b[:, 0] = v
    term.vel_command_b[:, 1] = 0.0
    term.vel_command_b[:, 2] = w
    with torch.inference_mode():
        act = policy(obs)
        obs, _, _, _ = venv.step(act)
    t_sim[0] += dt_ctrl
    pos, yaw, q = root_pose()
    log["steps"].append({"t": round(t_sim[0], 4), "pos": pos.round(5).tolist(), "quat_xyzw": q.round(6).tolist(),
                         "joints": robot.data.joint_pos.torch[0].cpu().numpy().round(5).tolist(), "cmd": [v, w]})
    return obs


def look():
    """Place the stabilized head camera, render, return (rgb uint8, GT depth, cam pos, yaw)."""
    pos, yaw, _ = root_pose()
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    cpos = np.array([pos[0], pos[1], args.cam_height]) + f * 0.10
    cam.set_world_poses(torch.tensor([cpos], dtype=torch.float32, device="cuda:0"),
                        torch.tensor([yaw_quat_xyzw(yaw)], dtype=torch.float32, device="cuda:0"), convention="world")
    env.sim.render()
    cam.update(0.0, force_recompute=True)
    rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
    depth = cam.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze().astype(np.float32)
    return rgb, depth, cpos, yaw


def to_habitat(p):
    return [float(p[0]), float(p[2] if len(p) > 2 else 0.0), float(-p[1])]


obs = venv.get_observations()
for _ in range(50):  # settle: stand for 1 s
    obs = control_step(0.0, 0.0, obs)

if args.probe is not None:
    v, w, secs = (list(args.probe) + [0.5, 0.0, 6.0][len(args.probe):])[:3]
    n = int(secs / dt_ctrl)
    for k in range(n):
        obs = control_step(v, w, obs)
        if k % 50 == 0:
            rgb, depth, cpos, yaw = look()
            Image.fromarray(rgb).save(os.path.join(args.out, f"probe_{k:04d}.jpg"), quality=92)
            pos, yaw, _ = root_pose()
            print(f"  t {t_sim[0]:5.2f} pos {pos.round(2)} yaw {math.degrees(yaw):6.1f} "
                  f"depth ahead {np.median(depth[260:380, 220:420]):.2f} m", flush=True)
    pos, yaw, _ = root_pose()
    print(f"probe done: z {pos[2]:.2f} m (standing ~0.7), moved to {pos.round(2)}", flush=True)
    json.dump(log, open(os.path.join(args.out, "motion.json"), "w"))
    env.close()
    app.close()
    sys.exit(0)

# ---------------------------------------------------------------- closed loop with the brain
goal = np.array(args.goal, float)
sock = socket.create_connection(("127.0.0.1", args.port))
start_pos, _, _ = root_pose()
wire.send(sock, {"type": "reset", "cond": args.cond, "start_xz": [start_pos[0], -start_pos[1]],
                 "goal_xz": [goal[0], -goal[1]], "cam_height_m": args.cam_height, "hazard_labels": args.labels,
                 "max_linear_mps": args.max_linear, "robust_scan": False, "guard": "fill",
                 "log_path": os.path.join(os.path.abspath(args.out), "brain.pkl")})
wire.recv(sock)
result, contact_t, w0 = "timeout", None, time.time()
while t_sim[0] < args.max_sim_s:
    pos, yaw, _ = root_pose()
    if np.hypot(*(goal - pos[:2])) < args.success_m:
        result = f"reached at {t_sim[0]:.1f} s"
        break
    if pos[2] < 0.45:
        result = f"fell at {t_sim[0]:.1f} s"
        break
    rgb, depth, cpos, yaw = look()
    fwd_h = [math.cos(yaw), -math.sin(yaw)]  # Isaac (cos, sin) -> Habitat xz (x, -y)
    to_goal = goal - pos[:2]
    right = np.array([math.sin(yaw), -math.cos(yaw)])
    bearing = float(math.atan2(to_goal @ right, to_goal @ np.array([math.cos(yaw), math.sin(yaw)])))
    wire.send(sock, {"type": "tick", "t": t_sim[0], "position": to_habitat([pos[0], pos[1], cpos[2]]),
                     "forward_xz": fwd_h, "bearing": bearing, "h": rgb.shape[0], "w": rgb.shape[1]}, rgb.tobytes())
    r, _ = wire.recv(sock)
    h, w = depth.shape
    gt_ahead = float(np.median(depth[int(.4 * h):int(.9 * h), int(.25 * w):int(.75 * w)]))
    log["ticks"].append({"t": round(t_sim[0], 4), "cam_pos": cpos.round(4).tolist(), "yaw": round(yaw, 5),
                         "gt_ahead_m": round(gt_ahead, 3), **{k: r.get(k) for k in ("linear", "angular", "tick_ms",
                                                                                     "safety", "event")}})
    if contact_t is None and gt_ahead < args.contact_m and r["linear"] > 0.02:
        contact_t = t_sim[0]
        result = f"crash at {t_sim[0]:.1f} s"
    if len(log["ticks"]) % 25 == 0:
        print(f"  t {t_sim[0]:5.1f} pos {pos[:2].round(2)} goal {np.hypot(*to_goal):.2f} m  v {r['linear']:.2f} "
              f"w {r['angular']:+.2f} tick {r['tick_ms']:.0f} ms {r.get('safety')} {r.get('event') or ''}"
              f"  wall {time.time() - w0:.0f} s", flush=True)
    dt = max(0.1, r["tick_ms"] / 1000.0)
    for _ in range(max(1, int(round(dt / dt_ctrl)))):
        obs = control_step(r["linear"], r["angular"], obs)
    if contact_t is not None and t_sim[0] - contact_t > args.after_contact_s:
        break

pos, _, _ = root_pose()
log["result"] = result
log["final_pos"] = pos.round(4).tolist()
wire.send(sock, {"type": "end", "result": result})
wire.recv(sock)
json.dump(log, open(os.path.join(args.out, "motion.json"), "w"))
print(f"{args.scene} {args.cond}: {result}; {len(log['ticks'])} ticks, {len(log['steps'])} control steps", flush=True)
env.close()
app.close()
