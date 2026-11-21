"""G1 walking with NVIDIA GEAR-SONIC in Isaac Lab (physics only): pass 1 of a SONIC demo shot.

The robot is SONIC's own G1 (29 DoF + Dex3 hands as fixed links; URDF, gains, armature and default pose from
gear_sonic/envs/manager_env/robots/g1.py), on 200 Hz physics with the 50 Hz SONIC planner + tracking policy
(tools/isaac/sonic.py). A scripted profile sets the planner's locomotion mode, speed and heading over time; the run
writes motion.json (root pose + joints per 20 ms step) for tools/isaac/replay_render.py.

    tools/isaac/isaaclab_run.sh tools/isaac/g1_sonic.py --out results/isaac/sonic/flat \\
        --profile '[[2, "idle", -1, 0], [8, "slow_walk", 0.5, 0, -45], [3, "idle", -1, -45]]'

Profile segments: [seconds, mode, speed (-1 = mode default), heading_deg (world yaw), optional heading_deg_end]
(the heading ramps linearly within a segment). Modes: idle, slow_walk, walk, run.
"""
import argparse
import json
import math
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--profile", required=True, help="JSON list of segments (file or inline)")
parser.add_argument("--scene", default="none", choices=["none", "office"])
parser.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="x y yaw_deg")
parser.add_argument("--assets", default=os.path.join(os.path.dirname(__file__), "../../isaac_assets"))
parser.add_argument("--wbc", default="/ml/GR00T-WholeBodyControl")
parser.add_argument("--seed", type=int, default=1234, help="planner random seed")
parser.add_argument("--replan", type=int, default=5, help="replan every N control steps (5 = 10 Hz, as deployed)")
parser.add_argument("--cpu-onnx", action="store_true")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sonic as S  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
A = os.path.abspath(args.assets)
os.makedirs(args.out, exist_ok=True)
segs = json.load(open(args.profile)) if os.path.exists(args.profile) else json.loads(args.profile)


def sonic_robot_cfg():
    from sonic_walker import robot_cfg
    return robot_cfg(args.wbc, ROOT)


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, render_interval=4, device="cuda:0"))
from isaaclab.terrains import TerrainImporter  # noqa: E402

TerrainImporter(TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1, num_envs=1,
                                   env_spacing=0.0,
                                   physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0,
                                                                                  dynamic_friction=1.0, restitution=0.0)))
sim_utils.DomeLightCfg(intensity=1500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.0))
if args.scene == "office":
    sim_utils.UsdFileCfg(usd_path=f"{A}/Isaac/Environments/Office/office.usd").func(
        "/World/Set", sim_utils.UsdFileCfg(usd_path=f"{A}/Isaac/Environments/Office/office.usd"))
rcfg = sonic_robot_cfg()
x0, y0, yaw0 = args.start[0], args.start[1], math.radians(args.start[2])
rcfg.init_state.pos = (x0, y0, 0.76)
rcfg.init_state.rot = (0.0, 0.0, math.sin(yaw0 / 2), math.cos(yaw0 / 2))  # Isaac Lab 3: x y z w
robot = Articulation(rcfg)
sim.reset()
names = list(robot.joint_names)
assert sorted(names) == sorted(S.ISAAC_NAMES), f"joint set mismatch: {names}"
SIM_OF_ISAAC = np.array([names.index(n) for n in S.ISAAC_NAMES])  # sim column of each IsaacLab-order joint
print(f"SONIC G1: {len(names)} joints; sim order == IsaacLab order: {list(SIM_OF_ISAAC) == list(range(29))}", flush=True)

# start in SONIC's default standing pose
jp = torch.zeros(1, 29, device="cuda:0")
jp[0, torch.as_tensor(SIM_OF_ISAAC, device="cuda:0")] = torch.as_tensor(S.DEFAULT, dtype=torch.float32, device="cuda:0")
robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
robot.set_joint_position_target(jp)
robot.write_data_to_sim()
robot.reset()


def state():
    d = robot.data
    p = d.root_link_pose_w.torch[0].cpu().numpy().astype(float)  # x y z qx qy qz qw
    quat = np.array([p[6], p[3], p[4], p[5]])  # -> w x y z
    w_b = d.root_ang_vel_b.torch[0].cpu().numpy().astype(float)
    q = d.joint_pos.torch[0].cpu().numpy().astype(float)[SIM_OF_ISAAC]
    dq = d.joint_vel.torch[0].cpu().numpy().astype(float)[SIM_OF_ISAAC]
    return p[:3], quat, w_b, q, dq


def command(t):
    a = 0.0
    for s in segs:
        d, mode, vel, h0 = s[0], s[1], float(s[2]), float(s[3])
        h1 = float(s[4]) if len(s) > 4 else h0
        if t < a + d or s is segs[-1]:
            u = min(max((t - a) / d, 0.0), 1.0)
            return mode, vel, math.radians(h0 + u * (h1 - h0))
        a += d


T_total = sum(s[0] for s in segs)
planner = S.Planner(seed=args.seed, gpu=not args.cpu_onnx)
tracker = S.Tracker(gpu=not args.cpu_onnx)
planner.initialize(S.DEFAULT_MJ)
_, quat0, _, _, _ = state()
tracker.start(quat0, planner.motion.quat[planner.frame])
inv_delta = S.qconj(tracker.apply_delta)  # world -> planner frame (yaw only)

log = {"args": vars(args), "joint_names": names, "dt": 0.02, "robot": "sonic_g1", "steps": [], "ref": []}
n_steps = int(round(T_total / 0.02))
w0, fell = time.time(), None
for k in range(n_steps):
    t = k * 0.02
    mode, vel, head = command(t)
    if k % args.replan == 0:
        if mode == "idle":
            move, face = [0.0, 0.0, 0.0], S.qrot(inv_delta, [math.cos(head), math.sin(head), 0.0]).tolist()
        else:
            face = S.qrot(inv_delta, [math.cos(head), math.sin(head), 0.0]).tolist()
            move = face
        planner.replan(S.MODES[mode], vel, move, face)
    pos, quat, w_b, q, dq = state()
    tgt = tracker.step(planner.motion, planner.frame, quat, w_b, q, dq)
    jt = torch.zeros(1, 29, device="cuda:0")
    jt[0, torch.as_tensor(SIM_OF_ISAAC, device="cuda:0")] = torch.as_tensor(tgt, dtype=torch.float32, device="cuda:0")
    robot.set_joint_position_target(jt)
    for _ in range(4):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(0.005)
    planner.advance()
    pos, quat, _, _, _ = state()
    log["steps"].append({"t": round(t + 0.02, 4), "pos": pos.round(5).tolist(),
                         "quat_xyzw": [round(float(v), 6) for v in (quat[1], quat[2], quat[3], quat[0])],
                         "joints": robot.data.joint_pos.torch[0].cpu().numpy().round(5).tolist(),
                         "cmd": [mode, vel, round(math.degrees(head), 2)]})
    f = planner.frame
    log["ref"].append({"q": planner.motion.q[f].round(4).tolist(), "root_quat": planner.motion.quat[f].round(5).tolist()})
    if fell is None and pos[2] < 0.45:
        fell = t
        print(f"  FELL at t {t:.2f} s (pelvis z {pos[2]:.2f})", flush=True)
    if k % 100 == 0:
        print(f"  t {t:5.2f} {mode:9s} v {vel:4.1f} head {math.degrees(head):6.1f}  pos {pos.round(2)} "
              f"yaw {math.degrees(S.qyaw(quat)):6.1f}  {(time.time() - w0) / (k + 1) * 1000:.0f} ms/step", flush=True)
log["result"] = {"fell_at": fell, "final_pos": log["steps"][-1]["pos"]}
json.dump(log, open(os.path.join(args.out, "motion.json"), "w"))
print(f"done: {n_steps} steps, fell {fell}; {args.out}/motion.json", flush=True)
app.close()
