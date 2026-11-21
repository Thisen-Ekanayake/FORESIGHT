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
parser.add_argument("--scene", default="office", choices=["office", "hospital", "warehouse", "dtw", "hm3d", "cube", "hm3d_nocoll", "hm3d_nomat", "none"])
parser.add_argument("--start", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "YAW_DEG"))
parser.add_argument("--goal", nargs=2, type=float, default=[4.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--cond", default="foresight", choices=["map", "foresight", "foresight_v1"], help="map = depth-only (the ghost)")
parser.add_argument("--labels", nargs="*", default=["glass wall", "glass door", "mirror"],
                    help="what FORESIGHT's guard grounds and fills (foresight only)")
parser.add_argument("--max-sim-s", type=float, default=90.0)
parser.add_argument("--success-m", type=float, default=0.5)
parser.add_argument("--contact-m", type=float, default=0.3, help="crash rule: GT depth ahead below this while walking")
parser.add_argument("--after-contact-s", type=float, default=2.0, help="keep simulating after a crash (the bump)")
parser.add_argument("--cam-height", type=float, default=1.22, help="head camera height over the floor, m")
parser.add_argument("--max-linear", type=float, default=0.45)
parser.add_argument("--out", required=True)
parser.add_argument("--policy", default="flat", choices=["flat", "rough"],
                    help="NVIDIA pretrained G1 policy: flat (default) or rough (height-scan obs; run on a plane here)")
parser.add_argument("--probe", nargs="*", type=float, default=None, metavar="V W SECONDS",
                    help="no brain: walk v m/s, w rad/s for SECONDS (default 0.5 0.0 6) and save stills")
parser.add_argument("--profile", default=None,
                    help="no brain: JSON list of [seconds, v, w] segments (a file or an inline string), walked in order")
parser.add_argument("--cmd-tau", type=float, default=0.0,
                    help="first-order smoothing of the (v, w) command before the policy, seconds (0 = off)")
parser.add_argument("--port", type=int, default=47011)
parser.add_argument("--chase", nargs=6, type=float, default=None, metavar=("EX", "EY", "EZ", "TX", "TY", "TZ"),
                    help="probe: also render a 1920x1080 still from this eye toward this target (world coords)")
parser.add_argument("--pt", action="store_true", help="path tracing for the stills (slow); default RTX real-time")
parser.add_argument("--spp", type=int, default=32)
parser.add_argument("--rendering-mode", default="quality", choices=["performance", "balanced", "quality"])
parser.add_argument("--iso", type=float, default=0.0, help="film ISO (exposure); 0 = auto exposure")
parser.add_argument("--extra-lights", default="auto", help="'auto' (per-scene set), 'none'")
parser.add_argument("--light-scale", type=float, default=1.0)
parser.add_argument("--pt-updates", type=int, default=6, help="app updates per path-traced still")
parser.add_argument("--set-mode", default="cfg", choices=["cfg", "ref"],
                    help="cfg: scene via Isaac Lab's scene config; ref: referenced onto the stage after env creation")
parser.add_argument("--warmup", type=int, default=300, help="render this many frames first (RTX streams big scenes in)")
parser.add_argument("--perception", default="isaac", choices=["isaac", "habitat"],
                    help="habitat: no renderer in Isaac (physics only); the brain renders the head camera in Habitat from "
                         "the same HM3D scan at the robot's pose (leaves the GPU to full-precision Qwen)")
parser.add_argument("--physics-device", default="cuda:0", help="cpu frees GPU memory for the brain's VLM")
parser.add_argument("--realtime-render", action="store_true", help="RTX real-time for the head camera (default)")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_kit_args = os.environ.get("G1_KIT_ARGS", "")
_kit_args += (" --/rtx/post/histogram/enabled=true" if args.iso <= 0 else f" --/rtx/post/tonemap/filmIso={args.iso}")
if args.pt:  # choose the renderer at launch (setting it after startup hung the app)
    _kit_args += (f" --/rtx/rendermode=PathTracing --/rtx/pathtracing/spp={args.spp}"
                  f" --/rtx/pathtracing/totalSpp={args.spp} --/rtx/pathtracing/optixDenoiser/enabled=true")
_render_cfg = dict(renderer="PathTracing" if args.pt else os.environ.get("G1_RT_RENDERER", "RealTimePathTracing"), samples_per_pixel_per_frame=args.spp,
                   denoiser=True, max_bounces=6, max_specular_transmission_bounces=8, sync_loads=True)
if args.perception == "habitat":
    app = AppLauncher(headless=True, enable_cameras=False).app  # Isaac Lab's headless app: physics, no renderer
else:
    app = AppLauncher(headless=True, enable_cameras=True, kit_args=_kit_args.strip(), rendering_mode=args.rendering_mode,
                      **_render_cfg,
                      experience=os.environ.get("G1_EXPERIENCE", "/ml/isaac-sim/apps/isaacsim.exp.base.python.kit")).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import wire  # noqa: E402

A = os.path.abspath(args.assets)
SCENES = {  # usd, (x, y, z) offset applied to the scene
    "office": (f"{A}/Isaac/Environments/Office/office.usd", (0, 0, 0)),
    "hospital": (f"{A}/Isaac/Environments/Hospital/hospital.usd", (0, 0, 0)),
    "warehouse": (f"{A}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd", (0, 0, 0)),
    "dtw": (f"{A}/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd", (0, 0, 0)),
    "hm3d": (os.path.join(ROOT, "results/isaac/00808_collision.usd"), (0, 0, 0)),
    "cube": (os.path.join(ROOT, "results/isaac/cube_test.usda"), (0, 0, 0)),
    "hm3d_nocoll": (os.path.join(ROOT, "results/isaac/00808_nocoll.usd"), (0, 0, 0)),
    "hm3d_nomat": (os.path.join(ROOT, "results/isaac/00808_nomat.usd"), (0, 0, 0)),
}
_PN = args.policy.capitalize()
TASK = f"Isaac-Velocity-{_PN}-G1-Play-v0"
CKPT = f"{A}/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-{_PN}-G1-v0/checkpoint.pt"
os.makedirs(args.out, exist_ok=True)


def yaw_quat_xyzw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


# ---------------------------------------------------------------- env config: Isaac Lab's own G1 flat (play) env
cfg = parse_env_cfg(TASK, device=args.physics_device, num_envs=1)
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 0.0
if args.policy == "rough":  # the rough policy on flat ground: plane terrain (its height scan then reads a flat floor)
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    if getattr(cfg.curriculum, "terrain_levels", None) is not None:
        cfg.curriculum.terrain_levels = None
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
# PhysX's GPU buffers default to sizes for thousands of parallel training robots (~2.5 GB); one robot needs a sliver,
# and the freed memory lets the brain keep Qwen3-VL on the same 8 GB GPU.
_px = cfg.sim.physics
for _k, _v in dict(gpu_max_rigid_contact_count=2**19, gpu_max_rigid_patch_count=2**16, gpu_found_lost_pairs_capacity=2**18,
                   gpu_found_lost_aggregate_pairs_capacity=2**20, gpu_total_aggregate_pairs_capacity=2**18,
                   gpu_collision_stack_size=2**24, gpu_heap_capacity=2**24, gpu_temp_buffer_capacity=2**22,
                   gpu_max_soft_body_contacts=2**10, gpu_max_particle_contacts=2**10).items():
    if hasattr(_px, _k):
        setattr(_px, _k, _v)
# the scene: NVIDIA's USD (with its own colliders); the terrain plane stays as the physics floor
if args.scene != "none" and args.set_mode == "cfg":
    usd, off = SCENES[args.scene]
    cfg.scene.stage_set = AssetBaseCfg(prim_path="/World/Set", spawn=sim_utils.UsdFileCfg(usd_path=usd),
                                       init_state=AssetBaseCfg.InitialStateCfg(pos=off))
env = ManagerBasedRLEnv(cfg)
if args.scene != "none" and args.set_mode == "ref":
    from pxr import Gf, UsdGeom
    usd, off = SCENES[args.scene]
    _p = env.sim.stage.DefinePrim("/World/Set", "Xform")
    _p.GetReferences().AddReference(usd)
    if any(off):
        UsdGeom.XformCommonAPI(_p).SetTranslate(Gf.Vec3d(*off))
if args.scene != "none":
    # NVIDIA's scenes keep their ceiling lights inside USD instances, which RTX doesn't light from: un-instance just
    # the instances that contain a light, so the scene is lit as designed.
    from pxr import Usd, UsdLux
    _roots = set()
    for _p in Usd.PrimRange(env.sim.stage.GetPrimAtPath("/World"), Usd.TraverseInstanceProxies()):
        if _p.IsInstanceProxy() and _p.HasAPI(UsdLux.LightAPI):
            _a = _p.GetParent()
            while _a and not _a.IsInstance():
                _a = _a.GetParent()
            if _a:
                _roots.add(str(_a.GetPath()))
    for _r in _roots:
        env.sim.stage.GetPrimAtPath(_r).SetInstanceable(False)
    print(f"un-instanced {len(_roots)} prims holding lights", flush=True)
# Extra soft ceiling panels (lighting only, no geometry): NVIDIA's office leaves its glass meeting rooms dark, so the
# glass reads as black instead of glass. (x0, y0, x1, y1) areas get a grid of downward rect lights at z = 2.7 m.
EXTRA_LIGHTS = {"office": [(-10.6, 35.5, -5.4, 48.5), (-16.5, 34.5, -11.4, 49.0)]}
if args.extra_lights == "auto" and args.scene in EXTRA_LIGHTS:
    from pxr import Gf, UsdGeom, UsdLux
    _n = 0
    for (x0, y0, x1, y1) in EXTRA_LIGHTS[args.scene]:
        for xx in np.arange(x0 + 1.2, x1, 2.4):
            for yy in np.arange(y0 + 1.2, y1, 2.4):
                L = UsdLux.RectLight.Define(env.sim.stage, f"/World/ExtraLights/L{_n}")
                L.CreateWidthAttr(1.2)
                L.CreateHeightAttr(1.2)
                L.CreateIntensityAttr(9000.0 * args.light_scale)
                L.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
                UsdGeom.Xformable(L.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(xx), float(yy), 2.7))
                _n += 1
    print(f"extra ceiling lights: {_n}", flush=True)
if args.scene != "none" and os.environ.get("G1_HIDE_GROUND", "1") == "1":  # hide the terrain plane's visual where a scene provides its own floor
    from pxr import UsdGeom
    stage = env.sim.stage
    for p in stage.Traverse():
        if str(p.GetPath()).startswith("/World/ground") and p.IsA(UsdGeom.Gprim):
            UsdGeom.Imageable(p).MakeInvisible()
# NVIDIA's pretrained actor (rsl_rl 5 format): ELU MLP (flat 123 -> 256 -> 128 -> 128 -> 37; rough 310 -> 512 -> 256 ->
# 128 -> 37), no observation normalization; deterministic action = the MLP's mean. Layer sizes come from the checkpoint.
_sd = torch.load(CKPT, map_location=args.physics_device, weights_only=False)["actor_state_dict"]
_W = [_sd[f"mlp.{i}.weight"] for i in (0, 2, 4, 6)]
_layers = []
for _i, _w in enumerate(_W):
    _layers += [torch.nn.Linear(_w.shape[1], _w.shape[0])] + ([torch.nn.ELU()] if _i < 3 else [])
policy = torch.nn.Sequential(*_layers).to(args.physics_device).eval()
policy.load_state_dict({k[len("mlp."):]: v for k, v in _sd.items() if k.startswith("mlp.")})


class _Env:
    """Minimal stepping wrapper: policy observations in, joint-position actions out (as Isaac Lab's rsl_rl wrapper)."""

    def get_observations(self):
        return env.observation_manager.compute()["policy"]

    def step(self, act):
        obs, rew, term_, trunc, extras = env.step(act)
        return obs["policy"], rew, term_, extras


venv = _Env()

robot = env.scene["robot"]


class RepCam:
    """A USD camera with Replicator annotators (the Kit RTX path that renders textured/MDL materials; Isaac Lab 3's
    camera sensor showed only the dome light once a textured scene was in the stage)."""

    def __init__(self, path, w, h, focal, depth=False):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom
        self.Gf = Gf
        cam = UsdGeom.Camera.Define(env.sim.stage, path)
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955 * h / w)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
        x = UsdGeom.Xformable(cam.GetPrim())
        x.ClearXformOpOrder()
        self.op = x.AddTransformOp()
        rp = rep.create.render_product(path, resolution=(w, h))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach(rp)
        self.depth = None
        if depth:
            self.depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            self.depth.attach(rp)
        self.w, self.h = w, h

    def look_at(self, eye, target):
        z = np.asarray(eye, float) - np.asarray(target, float)
        z /= np.linalg.norm(z) + 1e-9
        xx = np.cross([0.0, 0.0, 1.0], z)
        xx /= np.linalg.norm(xx) + 1e-9
        y = np.cross(z, xx)
        self.op.Set(self.Gf.Matrix4d(*xx, 0.0, *y, 0.0, *z, 0.0, *np.asarray(eye, float), 1.0))

    def grab(self, updates=2):
        for _ in range(updates):
            app.update()
        img = np.asarray(self.rgb.get_data())
        for _ in range(20):
            if img.ndim == 3 and img.shape[0] == self.h:
                break
            app.update()
            img = np.asarray(self.rgb.get_data())
        rgb = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
        d = np.asarray(self.depth.get_data(), np.float32).reshape(self.h, self.w) if self.depth is not None else None
        return rgb, d


RENDER = args.perception == "isaac"
head = RepCam("/World/HeadCam", 640, 640, 10.5, depth=True) if RENDER else None
chase = RepCam("/World/ChaseCam", 1920, 1080, 18.0) if (args.chase is not None and RENDER) else None
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


_cmd = [0.0, 0.0]


def control_step(v, w, obs):
    if args.cmd_tau > 0:  # optional low-pass on the command (actuator-level; the brain's decision is unchanged)
        a = min(1.0, dt_ctrl / args.cmd_tau)
        _cmd[0] += a * (v - _cmd[0])
        _cmd[1] += a * (w - _cmd[1])
        v, w = _cmd
    term.vel_command_b[:, 0] = v
    term.vel_command_b[:, 1] = 0.0
    term.vel_command_b[:, 2] = w
    with torch.inference_mode():
        act = policy(obs)
        obs, _, _, _ = venv.step(act)
    t_sim[0] += dt_ctrl
    pos, yaw, q = root_pose()
    log["steps"].append({"t": round(t_sim[0], 4), "pos": pos.round(5).tolist(), "quat_xyzw": q.round(6).tolist(),
                         "joints": robot.data.joint_pos.torch[0].cpu().numpy().round(5).tolist(),
                         "cmd": [round(float(v), 4), round(float(w), 4)]})
    return obs


def look():
    """Place the stabilized head camera, render, return (rgb uint8, GT depth, cam pos, yaw)."""
    pos, yaw, _ = root_pose()
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    cpos = np.array([pos[0], pos[1], args.cam_height]) + f * 0.10
    if head is None:  # Habitat perception: the brain renders the view
        return np.zeros((1, 1, 3), np.uint8), None, cpos, yaw
    head.look_at(cpos, cpos + f)
    rgb, depth = head.grab(updates=args.pt_updates if args.pt else 2)
    return rgb, depth, cpos, yaw


def to_habitat(p):
    return [float(p[0]), float(p[2] if len(p) > 2 else 0.0), float(-p[1])]


_w0 = time.time()
import omni.usd  # noqa: E402
_ctx = omni.usd.get_context()
for k in range(600 if RENDER else 0):  # pump the Kit app until the stage reports everything (payloads, textures) loaded
    app.update()
    _msg, _done, _total = _ctx.get_stage_loading_status()
    if k % 50 == 0:
        print(f"  loading {k}: {_done}/{_total} {_msg}", flush=True)
    if _total == 0 and k > 30:
        break
for k in range(args.warmup if RENDER else 0):  # let RTX load the scene's geometry and materials before any frame is used
    app.update()
    if k % 100 == 99:
        _r, _d, _, _ = look()
        print(f"  warm-up {k + 1}: depth ahead {np.median(_d[260:380, 220:420]):.2f} m, {time.time() - _w0:.0f} s", flush=True)
obs = venv.get_observations()
for _ in range(50):  # settle: stand for 1 s
    obs = control_step(0.0, 0.0, obs)

if args.profile is not None:  # scripted command profile, no brain: walking-quality tests and clean demo walks
    segs = json.load(open(args.profile)) if os.path.exists(args.profile) else json.loads(args.profile)
    log["profile"] = segs
    for secs, v, w in segs:
        for _ in range(int(round(secs / dt_ctrl))):
            obs = control_step(v, w, obs)
    pos, yaw, _ = root_pose()
    log["result"] = f"profile done at {t_sim[0]:.1f} s"
    print(f"profile done: {t_sim[0]:.1f} s, z {pos[2]:.2f} m, at {pos.round(2)} yaw {math.degrees(yaw):.1f}", flush=True)
    json.dump(log, open(os.path.join(args.out, "motion.json"), "w"))
    env.close()
    app.close()
    sys.exit(0)

if args.probe is not None:
    v, w, secs = (list(args.probe) + [0.5, 0.0, 6.0][len(args.probe):])[:3]
    n = int(secs / dt_ctrl)
    for k in range(n):
        obs = control_step(v, w, obs)
        if k % 50 == 0 or k == n - 1:
            rgb, depth, cpos, yaw = look()
            Image.fromarray(rgb).save(os.path.join(args.out, f"probe_{k:04d}.jpg"), quality=92)
            if chase is not None and (k == n - 1 or k == 0):
                chase.look_at(args.chase[:3], args.chase[3:])
                crgb, _ = chase.grab(updates=args.pt_updates if args.pt else 3)
                Image.fromarray(crgb).save(os.path.join(args.out, f"chase_{k:04d}.jpg"), quality=94)
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
                 "log_path": os.path.join(os.path.abspath(args.out), "brain.pkl"),
                 **({"habitat_scene": "00808-y9hTuugGdiq", "habitat_floor_y": 0.0614} if not RENDER else {})})
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
                     "forward_xz": fwd_h, "bearing": bearing, "h": rgb.shape[0], "w": rgb.shape[1],
                     "cam_height_m": args.cam_height}, rgb.tobytes() if RENDER else b"")
    r, _ = wire.recv(sock)
    if depth is not None:
        h, w = depth.shape
        gt_ahead = float(np.median(depth[int(.4 * h):int(.9 * h), int(.25 * w):int(.75 * w)]))
    else:
        gt_ahead = float(r.get("gt_ahead_m", 99.0))
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
        # FORESIGHT: angular > 0 = turn right (foresight.sim.velocity_step); Isaac yaw rate > 0 = turn left
        obs = control_step(r["linear"], -r["angular"], obs)
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
