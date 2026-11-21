"""Pass 2 of the "Through the Glass" demo: re-render recorded live G1 runs (tools/isaac/g1_live.py motion.json),
joint for joint, path traced, with scripted cinematic cameras. Nothing about the motion is changed: each frame sets
the robot's recorded root pose and 37 joint positions (nearest 20 ms control step) and only renders.

Two robots can share a shot: env_0 = "hero", env_1 = "ghost" (drawn translucent, the depth-only run).

  tools/isaac/isaaclab_run.sh tools/isaac/replay_render.py --scene hm3d --shots shots.json --out results/isaac/film/home

shots.json: [{"name": "s1", "hero": "results/.../motion.json", "ghost": null, "t0": 0, "t1": 20, "speed": 2.0,
             "cam": {"mode": "fixed", "eye": [x, y, z], "target": [x, y, z], "track": 0.5, "focal": 18}}, ...]
cam modes: fixed (eye; target blended toward the robot by `track`), follow (behind the robot: back, side, height,
smoothing), rig (offset [fwd, left, up] in the robot's frame, looking at the body), orbit (center, radius, height,
deg0, deg1). Cameras track the robot's path smoothed over +-0.6 s ("raw": true for the unsmoothed root pose);
the robot itself is posed exactly as recorded, interpolated between control steps.
Per shot: <out>/<name>/NNNNN.jpg and frames.json (sim time, camera eye/target/focal, robot poses) for the overlays.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--assets", default=os.path.join(ROOT, "isaac_assets"))
ap.add_argument("--scene", required=True, choices=["office", "hm3d", "warehouse", "hospital", "dtw"])
ap.add_argument("--robot", default="g1", choices=["g1", "sonic"],
                help="g1 = Isaac Lab's G1 (37 joints, Isaac Lab policy runs); sonic = Isaac Sim's G1 29-DoF + Dex3 hands "
                     "(the robot of tools/isaac/g1_sonic.py runs; joints matched by name, hand joints at 0)")
ap.add_argument("--shots", required=True)
ap.add_argument("--only", nargs="*", default=None, help="render only these shot names")
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=1440, help="square frame size")
ap.add_argument("--fps", type=float, default=15.0, help="rendered frames per video second (blended up later)")
ap.add_argument("--spp", type=int, default=24)
ap.add_argument("--updates", type=int, default=2, help="app updates per frame")
ap.add_argument("--still", action="store_true", help="render only each shot's first frame")
ap.add_argument("--iso", type=float, default=0.0,
                help="fixed exposure (film ISO), set at launch; 0 = auto exposure (histogram). Auto exposure drifts and "
                     "steps between frames and between runs, and turning it off at runtime does not stick")
ap.add_argument("--tex-streaming", type=int, default=1, help="0 = load every texture fully at launch (no streaming)")
ap.add_argument("--deterministic", type=int, default=1,
                help="1 = synchronous geometry streaming and no path-tracing/light caches (the synthetic-data settings). "
                     "Without it the first 35-130 frames after launch come out washed out with a random colour cast")
ap.add_argument("--debug-settings", action="store_true", help="print /rtx settings that change between frames, and frame luma")
args, _ = ap.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_kit = (f"--/rtx/rendermode=PathTracing --/rtx/pathtracing/spp={args.spp} --/rtx/pathtracing/totalSpp={args.spp}"
        " --/rtx/pathtracing/optixDenoiser/enabled=true"
        + (f" --/rtx/post/histogram/enabled=false --/rtx/post/tonemap/filmIso={args.iso}" if args.iso > 0
           else " --/rtx/post/histogram/enabled=true")
        + (" --/rtx-transient/hydra/geometrystreaming/syncLoad=true --/rtx/pathtracing/cached/enabled=false"
           " --/rtx/pathtracing/lightcache/cached/enabled=false --/rtx/pathtracing/lightcache/spatialCache/enabled=false"
           " --/rtx/raytracing/cached/enabled=false --/rtx/raytracing/lightcache/spatialCache/enabled=false"
           if args.deterministic else "")
        + ("" if args.tex_streaming else " --/rtx-transient/resourcemanager/enableTextureStreaming=false"))
app = AppLauncher(headless=True, enable_cameras=True, kit_args=_kit, renderer="PathTracing",
                  samples_per_pixel_per_frame=args.spp, denoiser=True, max_bounces=6,
                  max_specular_transmission_bounces=8, sync_loads=True,
                  experience="/ml/isaac-sim/apps/isaacsim.exp.base.python.kit").app

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

A = os.path.abspath(args.assets)
SCENES = {"office": f"{A}/Isaac/Environments/Office/office.usd",
          "hospital": f"{A}/Isaac/Environments/Hospital/hospital.usd",
          "warehouse": f"{A}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
          "dtw": f"{A}/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
          "hm3d": os.path.join(ROOT, "results/isaac/00808_collision.usd")}
EXTRA_LIGHTS = {"office": [(-10.6, 35.5, -5.4, 48.5), (-16.5, 34.5, -11.4, 49.0), (-16.5, 30.0, -5.4, 34.5)]}
shots = json.load(open(args.shots))
if args.only:
    shots = [s for s in shots if s["name"] in args.only]
need_ghost = any(s.get("ghost") for s in shots)

cfg = parse_env_cfg("Isaac-Velocity-Flat-G1-Play-v0", device="cuda:0", num_envs=2 if need_ghost else 1)
cfg.scene.num_envs = 2 if need_ghost else 1
cfg.scene.env_spacing = 0.0
cfg.scene.robot.spawn.usd_path = f"{A}/Isaac/IsaacLab/Robots/Unitree/G1/g1.usd"
if args.robot == "sonic":  # posed only (gravity off), so one stiff implicit actuator for every joint is enough
    from isaaclab.actuators import ImplicitActuatorCfg
    cfg.scene.robot.spawn.usd_path = f"{A}/Isaac/Robots/Unitree/G1/g1.usd"
    cfg.scene.robot.actuators = {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=200.0, damping=10.0)}
    cfg.scene.robot.init_state.joint_pos = {".*": 0.0}
    for _f in list(vars(cfg.rewards)):  # Isaac Lab G1 reward terms name joints this robot doesn't have
        if not _f.startswith("_"):
            setattr(cfg.rewards, _f, None)
    for _f in ("add_base_mass", "base_com"):
        if hasattr(cfg.events, _f):
            setattr(cfg.events, _f, None)
cfg.scene.robot.init_state.pos = (0.0, 0.0, -20.0)  # parked below the floor until posed
cfg.terminations.time_out = None
cfg.commands.base_velocity.debug_vis = False
cfg.terminations.base_contact = None
for name in ("push_robot", "base_external_force_torque"):
    if hasattr(cfg.events, name):
        setattr(cfg.events, name, None)
cfg.scene.stage_set = AssetBaseCfg(prim_path="/World/Set", spawn=sim_utils.UsdFileCfg(usd_path=SCENES[args.scene]))
_px = cfg.sim.physics
for _k, _v in dict(gpu_max_rigid_contact_count=2**19, gpu_max_rigid_patch_count=2**16, gpu_found_lost_pairs_capacity=2**18,
                   gpu_found_lost_aggregate_pairs_capacity=2**20, gpu_total_aggregate_pairs_capacity=2**18,
                   gpu_collision_stack_size=2**24, gpu_heap_capacity=2**24, gpu_temp_buffer_capacity=2**22).items():
    if hasattr(_px, _k):
        setattr(_px, _k, _v)
cfg.sim.gravity = (0.0, 0.0, 0.0)  # the rendered pose is the recorded one: a physics step can't sag it
env = ManagerBasedRLEnv(cfg)
stage = env.sim.stage
robot = env.scene["robot"]

# lights (as in the live runs) and the hidden terrain plane
if args.scene in EXTRA_LIGHTS:
    n = 0
    for (x0, y0, x1, y1) in EXTRA_LIGHTS[args.scene]:
        for xx in np.arange(x0 + 1.2, x1, 2.4):
            for yy in np.arange(y0 + 1.2, y1, 2.4):
                L = UsdLux.RectLight.Define(stage, f"/World/ExtraLights/L{n}")
                L.CreateWidthAttr(1.2)
                L.CreateHeightAttr(1.2)
                L.CreateIntensityAttr(9000.0)
                L.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
                UsdGeom.Xformable(L.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(xx), float(yy), 2.7))
                n += 1
if args.scene == "hm3d":  # warm interior bulbs over the bedroom, hallway and living room
    for i, (x, y) in enumerate([(-3.0, -0.3), (-1.2, -0.3), (-1.5, -2.6), (1.0, -2.8), (1.6, -1.0), (-3.3, 1.2)]):
        s_ = UsdLux.SphereLight.Define(stage, f"/World/ExtraLights/B{i}")
        s_.CreateRadiusAttr(0.12)
        s_.CreateIntensityAttr(45000.0)
        s_.CreateColorAttr(Gf.Vec3f(1.0, 0.9, 0.78))
        UsdGeom.Xformable(s_.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(x, y, 2.25))
for p in stage.Traverse():
    if str(p.GetPath()).startswith("/World/ground") and p.IsA(UsdGeom.Gprim):
        UsdGeom.Imageable(p).MakeInvisible()

# ghost material: translucent cyan hologram on every mesh of env_1's robot
if need_ghost:
    gm = UsdShade.Material.Define(stage, "/World/Looks/Ghost")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/Ghost/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.35, 0.8, 1.0))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.12, 0.45, 0.7))
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.38)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    gm.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    # bound on the robot root, stronger than descendants: reaches the instanced meshes without un-instancing them
    # (changing instancing after the env starts invalidates the physics views)
    groot = stage.GetPrimAtPath("/World/envs/env_1/Robot")
    UsdShade.MaterialBindingAPI.Apply(groot).Bind(gm, UsdShade.Tokens.strongerThanDescendants)

# camera (Replicator render product: the Kit RTX path)
cam = UsdGeom.Camera.Define(stage, "/World/FilmCam")
cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(20.955)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.03, 200.0))
cam_x = UsdGeom.Xformable(cam.GetPrim())
cam_x.ClearXformOpOrder()
cam_op = cam_x.AddTransformOp()
rp = rep.create.render_product("/World/FilmCam", resolution=(args.res, args.res))
an = rep.AnnotatorRegistry.get_annotator("rgb")
an.attach(rp)
an_d = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
an_d.attach(rp)


def look_at(eye, target):
    z = np.asarray(eye, float) - np.asarray(target, float)
    z /= np.linalg.norm(z) + 1e-9
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    return Gf.Matrix4d(*x, 0.0, *y, 0.0, *z, 0.0, *np.asarray(eye, float), 1.0)


class Motion:
    """A recorded run. at(t) interpolates between the 20 ms control steps (position and joints linearly, rotation by
    slerp), so frames at any rate get the in-between pose instead of snapping to the nearest step (judder).
    cam_at(t) is the root path smoothed over +-cam_sigma s (Gaussian), for cameras only: the robot is posed exactly
    as recorded, but the camera doesn't copy every step's sway."""

    def __init__(self, path, cam_sigma=0.6):
        m = json.load(open(path))
        S = m["steps"]
        self.names = m.get("joint_names")
        self.t = np.array([s["t"] for s in S])
        self.pos = np.array([s["pos"] for s in S])
        self.q = np.array([s["quat_xyzw"] for s in S])
        self.j = np.array([s["joints"] for s in S])
        self.end = float(self.t[-1])
        x, y, z, w = self.q.T
        yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        dt = float(np.median(np.diff(self.t)))
        r = max(1, int(3 * cam_sigma / dt))
        ker = np.exp(-0.5 * (np.arange(-r, r + 1) * dt / cam_sigma) ** 2)
        ker /= ker.sum()

        def smooth(a):
            pad = np.concatenate([np.full(r, a[0]), a, np.full(r, a[-1])])
            return np.convolve(pad, ker, "valid")
        self.cam_xy = np.stack([smooth(self.pos[:, 0]), smooth(self.pos[:, 1])], 1)
        self.cam_yaw = smooth(yaw)

    def _k(self, t):
        k = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        u = float(np.clip((t - self.t[k]) / max(self.t[k + 1] - self.t[k], 1e-9), 0.0, 1.0))
        return k, u

    def at(self, t):
        k, u = self._k(t)
        q0, q1 = self.q[k], self.q[k + 1].copy()
        d = float(np.dot(q0, q1))
        if d < 0:
            q1, d = -q1, -d
        if d > 0.9995:
            q = q0 + u * (q1 - q0)
        else:
            th = math.acos(d)
            q = (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
        q = q / np.linalg.norm(q)
        return (1 - u) * self.pos[k] + u * self.pos[k + 1], q, (1 - u) * self.j[k] + u * self.j[k + 1]

    def cam_at(self, t):
        k, u = self._k(t)
        return (1 - u) * self.cam_xy[k] + u * self.cam_xy[k + 1], (1 - u) * self.cam_yaw[k] + u * self.cam_yaw[k + 1]


def pose_robots(hero, ghost, t, t_ghost):
    poses, joints = [], []
    for m, tt in ((hero, t), (ghost, t_ghost)):
        if m is None:
            if need_ghost:
                poses.append([0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 1.0])
                joints.append(robot.data.default_joint_pos.torch[0].cpu().numpy())
            continue
        p, q, j = m.at(tt)
        poses.append(list(p) + list(q))
        if m.names is not None and list(m.names) != list(robot.joint_names):  # match by name; others at default
            full = robot.data.default_joint_pos.torch[0].cpu().numpy().copy()
            for i, nm in enumerate(robot.joint_names):
                if nm in m.names:
                    full[i] = j[m.names.index(nm)]
            j = full
        joints.append(j)
    root = torch.tensor(np.array(poses), dtype=torch.float32, device="cuda:0")
    jp = torch.tensor(np.array(joints), dtype=torch.float32, device="cuda:0")
    robot.write_root_pose_to_sim(root)
    robot.write_root_velocity_to_sim(torch.zeros(len(poses), 6, device="cuda:0"))
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    robot.set_joint_position_target(jp)
    robot.write_data_to_sim()
    env.sim.forward()


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


env.reset()
for _ in range(3):  # a few real steps so the physics views are live before any pose is written
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda:0"))
for _ in range(60):  # let the scene stream in
    app.update()
os.makedirs(args.out, exist_ok=True)
_s0 = carb.settings.get_settings()
RTX_DEFAULT = {k: _s0.get(k) for k in ("/rtx/post/histogram/enabled", "/rtx/post/tonemap/filmIso",
                                       "/rtx/pathtracing/spp", "/rtx/pathtracing/totalSpp")}
print("rtx defaults:", RTX_DEFAULT, flush=True)
motions = {}
for shot in shots:
    hero = motions.setdefault(shot["hero"], Motion(shot["hero"])) if shot.get("hero") else None
    ghost = motions.setdefault(shot["ghost"], Motion(shot["ghost"])) if shot.get("ghost") else None
    od = os.path.join(args.out, shot["name"])
    os.makedirs(od, exist_ok=True)
    c = shot["cam"]
    t0, t1, speed = float(shot["t0"]), float(shot["t1"]), float(shot.get("speed", 1.0))
    n = max(1, int(round((t1 - t0) / speed * args.fps)))
    if args.still:
        n = 1
    cam.GetFocalLengthAttr().Set(float(c.get("focal", 18.0)))
    _st = carb.settings.get_settings()
    for key, val in {**RTX_DEFAULT, **shot.get("rtx", {})}.items():  # per-shot renderer settings (post/exposure)
        _st.set(key, val)
    upd = int(shot.get("updates", args.updates))
    sm = {"eye": None, "tgt": None}
    log, w0 = [], time.time()
    for k in range(n):
        t = t0 + (t1 - t0) * (k / max(n - 1, 1)) if n > 1 else t0
        tg = t + float(shot.get("ghost_offset", 0.0))
        pose_robots(hero, ghost, t, tg)
        lead = hero if hero is not None else ghost
        tl = t if hero is not None else tg
        rp_, rq, _ = lead.at(tl)
        ryaw = yaw_of(rq)
        if c.get("raw", False):  # old behaviour: the camera follows the raw root pose
            cxy, cyaw = rp_[:2], ryaw
        else:  # smoothed path: no step-by-step sway in the camera
            cxy, cyaw = lead.cam_at(tl)
        f = np.array([math.cos(cyaw), math.sin(cyaw), 0.0])
        rgt = np.array([math.sin(cyaw), -math.cos(cyaw), 0.0])
        body = np.array([cxy[0], cxy[1], 0.95])
        u = k / max(n - 1, 1)
        if c["mode"] == "fixed":
            eye = np.array(c["eye"], float) + u * np.array(c.get("eye_move", [0, 0, 0]), float)
            tgt = (1 - c.get("track", 0.0)) * np.array(c["target"], float) + c.get("track", 0.0) * body
        elif c["mode"] == "follow":
            eye = body - f * c.get("back", 2.2) + rgt * c.get("side", 0.6) + np.array([0, 0, c.get("height", 0.55)])
            tgt = body + f * c.get("ahead", 1.0) + np.array([0, 0, -0.15])
            # keep the camera out of walls: raycast from just outside the robot toward the desired eye position and
            # pull in to 85% of the free distance (the scan's colliders)
            try:
                from omni.physx import get_physx_scene_query_interface
                v = eye - body
                dist = float(np.linalg.norm(v))
                dirv = v / max(dist, 1e-6)
                o = body + dirv * 0.45
                hit = get_physx_scene_query_interface().raycast_closest(carb.Float3(*o.tolist()),
                                                                        carb.Float3(*dirv.tolist()), dist)
                if hit["hit"]:
                    eye = o + dirv * max(0.25, 0.85 * hit["distance"])
            except Exception as e:  # noqa: BLE001
                if k == 0:
                    print("raycast unavailable, plain follow camera:", e, flush=True)
        elif c["mode"] == "rig":  # a camera carried along in the robot's (smoothed) frame, looking at the body
            off = np.array(c.get("offset", [2.2, 1.2, 0.35]), float)  # forward, left, up from the body centre (m)
            eye = body + f * off[0] - rgt * off[1] + np.array([0, 0, off[2]])
            look = np.array(c.get("look", [0.3, 0.0, -0.1]), float)
            tgt = body + f * look[0] - rgt * look[1] + np.array([0, 0, look[2]])
        elif c["mode"] == "pov":  # the robot's own head camera: level, 1.22 m, 0.1 m ahead of the base
            eye = np.array([rp_[0], rp_[1], 1.22]) + f * 0.10
            tgt = eye + f
        else:  # orbit
            ang = math.radians(c["deg0"] + (c["deg1"] - c["deg0"]) * (u * u * (3 - 2 * u)))
            ctr = np.array(c.get("center", body), float)
            eye = ctr + np.array([c["radius"] * math.cos(ang), c["radius"] * math.sin(ang), c["height"]])
            tgt = ctr
        a = c.get("smooth", 1.0 if not c.get("raw") else 0.15) if sm["eye"] is not None else 1.0
        sm["eye"] = eye if sm["eye"] is None else (1 - a) * sm["eye"] + a * eye
        sm["tgt"] = tgt if sm["tgt"] is None else (1 - a) * sm["tgt"] + a * tgt
        cam_op.Set(look_at(sm["eye"], sm["tgt"]))
        # frame 0: extra updates; "preroll" lets the renderer settle first (exposure/streaming converged over ~120
        # updates after launch in the 2026-11-20 walk test; frames before that came out overexposed)
        for _ in range(upd if k else upd + 6 + int(shot.get("preroll", 0))):
            app.update()
        img = np.asarray(an.get_data())
        if img.ndim != 3:
            for _ in range(10):
                app.update()
                img = np.asarray(an.get_data())
                if img.ndim == 3:
                    break
        if args.debug_settings:
            _flat = {}
            def _walk(pfx, d):
                for kk, vv in (d or {}).items():
                    (_walk if isinstance(vv, dict) else lambda a, b: _flat.__setitem__(a, b))(f"{pfx}/{kk}", vv)
            for root in ("/rtx", "/rtx-transient"):
                _walk(root, carb.settings.get_settings().get(root))
            _prev = globals().get("_dbg_prev")
            ch = {kk: (_prev.get(kk), vv) for kk, vv in _flat.items() if _prev is not None and _prev.get(kk) != vv} if _prev else {}
            globals()["_dbg_prev"] = _flat
            print(f"  dbg k {k} luma {img[..., :3].mean():.1f} changed {len(ch)} " + "; ".join(f"{kk}: {a!r} -> {b!r}" for kk, (a, b) in list(ch.items())[:12]), flush=True)
        Image.fromarray(np.ascontiguousarray(img[..., :3])).save(os.path.join(od, f"{k:05d}.jpg"), quality=94)
        dep = np.asarray(an_d.get_data(), np.float32).reshape(args.res, args.res)
        np.save(os.path.join(od, f"{k:05d}_d.npy"), dep[::4, ::4].astype(np.float16))
        log.append({"k": k, "t": round(t, 4), "t_ghost": round(tg, 4), "eye": sm["eye"].round(4).tolist(),
                    "target": sm["tgt"].round(4).tolist(), "focal": float(c.get("focal", 18.0)),
                    "robot": [float(rp_[0]), float(rp_[1]), float(ryaw)]})
        if k % 20 == 0:
            print(f"  {shot['name']} {k + 1}/{n} t {t:.1f} {(time.time() - w0) / (k + 1):.2f} s/frame", flush=True)
    json.dump({"shot": shot, "res": args.res, "fps": args.fps, "aperture": 20.955, "frames": log},
              open(os.path.join(od, "frames.json"), "w"))
    print(f"{shot['name']}: {n} frames -> {od}", flush=True)
env.close()
app.close()
