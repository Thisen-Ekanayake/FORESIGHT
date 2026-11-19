"""The Mirror Test: a Go2 walks up to the full-height hallway mirror in HM3D 00808 and moves in front of it, in Isaac
Sim with ray-traced reflections, so its head camera sees its own body reflected (Habitat's camera has no body and its
mirrors are baked photographs, so there it can't).

A scripted choreography, not a closed loop: approach, stop, turn left/right, side-step, walk in until the mirror fills
the view (the case where Qwen3-VL-2B says "no mirror", docs/results/foresight_B.md). The perception (Depth Anything V2,
Qwen3-VL) runs afterwards on the saved head-camera frames (tools/isaac/mirror_test_perception.py), because Isaac and
the models don't fit in 8 GB together.

  /ml/isaac-sim/python.sh tools/isaac/mirror_test.py --out results/isaac/mirror_test

Per frame: chase/NNNNN.jpg, eye/NNNNN.jpg (640x640, 90 deg, level), eye_depth/NNNNN.npy (float16 distance to image
plane, Isaac ground truth; a mirror pixel is the mirror's own distance), eye_mirror/NNNNN.png (mirror pixels from
Isaac's instance segmentation); frames.json: time, base pose, head camera pose, phase.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument("--house", default="results/isaac/00808")
parser.add_argument("--out", required=True)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--chase-res", default="1920x1080")
parser.add_argument("--eye-res", default="640x640")
parser.add_argument("--renderer", default="RaytracedLighting", choices=["RaytracedLighting", "PathTracing"])
parser.add_argument("--spp", type=int, default=64)
parser.add_argument("--still", type=float, default=None, help="render only the frame at this time to <out>/still_*.jpg")
parser.add_argument("--t-to", type=float, default=None)
parser.add_argument("--chase", default="side", choices=["side", "behind", "corner"])
parser.add_argument("--robot", default="h1", choices=["h1", "go2"])
parser.add_argument("--emission", type=float, default=0.25, help="scan-texture emission (1.0 = the 2026-11-01 replay)")
parser.add_argument("--bulbs", type=float, default=12000.0)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "renderer": args.renderer})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.deprecation_manager import import_module  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy, H1FlatTerrainPolicy  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdGeom  # noqa: E402

from house_stage import add_floor_collider, add_lights, build_house, look_at, make_cam  # noqa: E402

torch = import_module("torch")
settings = carb.settings.get_settings()
if args.renderer == "PathTracing":
    settings.set("/rtx/pathtracing/spp", args.spp)
    settings.set("/rtx/pathtracing/totalSpp", args.spp)
settings.set("/rtx/post/aa/op", 3)
settings.set("/rtx/reflections/maxRoughness", 0.3)
settings.set("/rtx/reflections/maxReflectionBounces", 6)  # two facing mirrors in this hallway
settings.set("/rtx/translucency/maxRefractionBounces", 4)

for d in ("chase", "eye", "eye_depth", "eye_mirror"):
    os.makedirs(os.path.join(args.out, d), exist_ok=True)

FLOOR_Z = 0.0614  # the run's start position height (Habitat y) = this storey's floor in the scan
MIRROR_X = -0.10  # hallway mirror plane (mirror115/116), facing -x; the start pose faces it from 2.5 m

# --- choreography: keyframes (t, x, y, yaw_deg, phase) ------------------------------------------------------------
K = [
    (0.0, -2.59, -0.29, 0.0, "approach"),
    (1.5, -2.59, -0.29, 0.0, "approach"),
    (5.5, -1.10, -0.29, 0.0, "approach"),
    (7.0, -1.10, -0.29, 0.0, "look"),
    (8.5, -1.10, -0.29, 28.0, "turn"),
    (10.5, -1.10, -0.29, -28.0, "turn"),
    (11.8, -1.10, -0.29, 0.0, "turn"),
    (13.3, -1.10, 0.06, 0.0, "step"),
    (15.3, -1.10, -0.64, 0.0, "step"),
    (16.8, -1.10, -0.29, 0.0, "step"),
    (17.8, -1.10, -0.29, 0.0, "look"),
    (20.3, -0.48, -0.29, 0.0, "close"),
    (22.5, -0.48, -0.29, 0.0, "close"),
]
KT = np.array([k[0] for k in K])
KP = np.array([[k[1], k[2]] for k in K])
KY = np.deg2rad([k[3] for k in K])


def smooth_interp(t, xs, ys):
    """Piecewise interpolation with smoothstep easing inside each segment (stops and starts are gentle)."""
    t = float(np.clip(t, xs[0], xs[-1]))
    i = int(np.clip(np.searchsorted(xs, t, side="right") - 1, 0, len(xs) - 2))
    u = (t - xs[i]) / max(xs[i + 1] - xs[i], 1e-9)
    u = u * u * (3 - 2 * u)
    return ys[i] + (ys[i + 1] - ys[i]) * u


def ref(t):
    return np.array([smooth_interp(t, KT, KP[:, 0]), smooth_interp(t, KT, KP[:, 1])]), float(smooth_interp(t, KT, KY))


def phase_at(t):
    i = int(np.clip(np.searchsorted(KT, t, side="right") - 1, 0, len(K) - 1))
    return K[min(i + 1, len(K) - 1)][4] if t > KT[i] else K[i][4]


# --- stage --------------------------------------------------------------------------------------------------------
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
info = build_house(stage, args.house, emission=args.emission)
print(f"house: {info['n_meshes']} meshes, {len(info['mirrors'])} mirrors")
add_lights(stage, np.array([-5.0, -3.0]), np.array([1.0, 2.0]), FLOOR_Z, bulbs=args.bulbs)
add_floor_collider(stage, (-2.0, -0.5), FLOOR_Z)

RenderingManager.set_dt(1.0 / args.fps)
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(1.0 / 200.0)
p0, y0 = ref(0.0)
# Robot: Unitree H1 humanoid (default; head camera ~1.6 m, close to the 1.5 m camera FORESIGHT's runs were decided
# with) or the Go2 of the 2026-11-01 replay. BASE_Z = spawn height of the base link; HEAD = camera offset from it.
ROBOT = {"h1": (H1FlatTerrainPolicy, 1.05, 0.18, 0.62), "go2": (Go2FlatTerrainPolicy, 0.42, 0.30, 0.10)}[args.robot]
go2 = ROBOT[0](prim_path="/World/Robot", position=[float(p0[0]), float(p0[1]), FLOOR_Z + ROBOT[1]],
               orientation=[math.cos(y0 / 2), 0.0, 0.0, math.sin(y0 / 2)])
state = {"first": True, "t": 0.0}


def base_pose():
    pos, quat = go2.robot.get_world_poses()
    p = np.asarray(pos.numpy()[0], float)
    w, x, y, z = np.asarray(quat.numpy()[0], float)
    return p, math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def track(t):
    p, yaw = base_pose()
    tgt, yaw_t = ref(t + 0.2)
    nxt, _ = ref(t + 0.4)
    v_ff = (nxt - tgt) / 0.2
    _, yaw_n = ref(t + 0.4)
    w_ff = (yaw_n - yaw_t) / 0.2
    e = tgt - p[:2]
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = c * e[0] + s * e[1], -s * e[0] + c * e[1]
    fx, fy = c * v_ff[0] + s * v_ff[1], -s * v_ff[0] + c * v_ff[1]
    dyaw = (yaw_t - yaw + math.pi) % (2 * math.pi) - math.pi
    return [float(np.clip(fx + 2.0 * ex, -0.8, 0.8)), float(np.clip(fy + 2.0 * ey, -0.5, 0.5)),
            float(np.clip(w_ff + 2.5 * dyaw, -1.2, 1.2))]


def on_physics_step(step_size, context):
    if state["first"]:
        go2.initialize()
        state["first"] = False
        return
    if state.get("running"):
        state["t"] += step_size
    go2.forward(step_size, torch.tensor(track(state["t"])))


# --- cameras --------------------------------------------------------------------------------------------------------
chase_op = make_cam(stage, "/World/ChaseCam", 14.0)
eye_op = make_cam(stage, "/World/EyeCam", 10.5)  # 90 deg horizontal, like the Habitat camera
cw, ch = (int(v) for v in args.chase_res.split("x"))
ew, eh = (int(v) for v in args.eye_res.split("x"))
rp_chase = rep.create.render_product("/World/ChaseCam", resolution=(cw, ch))
rp_eye = rep.create.render_product("/World/EyeCam", resolution=(ew, eh))
an_chase = rep.AnnotatorRegistry.get_annotator("rgb")
an_chase.attach(rp_chase)
an_eye = rep.AnnotatorRegistry.get_annotator("rgb")
an_eye.attach(rp_eye)
an_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
an_depth.attach(rp_eye)
an_inst = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast")
an_inst.attach(rp_eye)
mirror_paths = set(info["mirrors"])
cam_state = {"eye": None, "look": None}


def place_cameras(t):
    p, yaw = base_pose()
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    # Head camera: level, at the front of the head.
    head = p + f * ROBOT[2] + np.array([0, 0, ROBOT[3]])
    eye_op.Set(look_at(head, head + f))
    # Chase camera: behind and to the side of the robot, framing robot + mirror + reflection.
    tall = args.robot == "h1"
    if args.chase == "corner":
        # Fixed in the room's back-left corner (its back wall is at x = -3.3, the dresser side at y = +1.0), panning
        # to keep the robot and the mirror in frame.
        eye = np.array([-3.12, 0.72, FLOOR_Z + 1.65])
        look = 0.55 * np.array([p[0], p[1], FLOOR_Z + 1.0]) + 0.45 * np.array([MIRROR_X, p[1] - 0.2, FLOOR_Z + 1.0])
    elif args.chase == "side":
        # clamped inside the room: its back wall (the second mirror, mirror128) is at x = -4.16
        eye = np.array([max(p[0] - (1.9 if tall else 1.25), -3.9), p[1] - (1.0 if tall else 0.85), FLOOR_Z + (1.45 if tall else 0.95)])
        look = np.array([MIRROR_X - 0.2, p[1] - 0.05, FLOOR_Z + (1.0 if tall else 0.45)])
    else:
        eye = np.array([p[0] - 1.9, p[1], FLOOR_Z + (1.5 if tall else 0.9)])
        look = np.array([MIRROR_X, p[1], FLOOR_Z + (1.0 if tall else 0.4)])
    a = 0.12 if cam_state["eye"] is not None else 1.0
    cam_state["eye"] = eye if cam_state["eye"] is None else (1 - a) * cam_state["eye"] + a * eye
    cam_state["look"] = look if cam_state["look"] is None else (1 - a) * cam_state["look"] + a * look
    chase_op.Set(look_at(cam_state["eye"], cam_state["look"]))
    return head, f


_cb = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0.0)
timeline.set_end_time(1.0e7)
timeline.set_looping(False)
timeline.set_play_every_frame(True)
timeline.play()
simulation_app.update()
for _ in range(40):  # settle onto its feet before the clock starts
    place_cameras(0.0)
    simulation_app.update()
state["running"] = True


def mirror_mask(seg):
    data, idmap = seg["data"], seg["info"].get("idToLabels", {})
    ids = [int(i) for i, path in idmap.items() if any(str(path).startswith(m) for m in mirror_paths)]
    return np.isin(np.asarray(data).reshape(eh, ew), ids)


t_end = args.t_to or KT[-1]
frames, n = [], 0
w0 = time.time()
while simulation_app.is_running() and state["t"] < t_end:
    head, f = place_cameras(state["t"])
    state["loops"] = state.get("loops", 0) + 1
    if state["loops"] % 15 == 0:
        print(f"  loop {state['loops']}, t {state['t']:.2f} s, {(time.time() - w0) / state['loops']:.2f} s/loop", flush=True)
    simulation_app.update()  # one update per frame: each advances physics by 1/fps (spp samples per update)
    c, e = np.asarray(an_chase.get_data()), np.asarray(an_eye.get_data())
    if c.ndim != 3 or c.shape[0] != ch or e.ndim != 3:
        continue
    if args.still is not None and state["t"] < args.still:
        continue
    tag = "still_" if args.still is not None else ""
    if args.still is not None:
        Image.fromarray(np.ascontiguousarray(c[..., :3])).save(os.path.join(args.out, "still_chase.jpg"), quality=95)
        Image.fromarray(np.ascontiguousarray(e[..., :3])).save(os.path.join(args.out, "still_eye.jpg"), quality=95)
    else:
        Image.fromarray(np.ascontiguousarray(c[..., :3])).save(os.path.join(args.out, "chase", f"{n:05d}.jpg"), quality=93)
        Image.fromarray(np.ascontiguousarray(e[..., :3])).save(os.path.join(args.out, "eye", f"{n:05d}.jpg"), quality=95)
    d = np.asarray(an_depth.get_data(), np.float32).reshape(eh, ew)
    mm = mirror_mask(an_inst.get_data())
    np.save(os.path.join(args.out, "eye_depth", f"{tag}{n:05d}.npy"), d.astype(np.float16))
    Image.fromarray((mm * 255).astype(np.uint8)).save(os.path.join(args.out, "eye_mirror", f"{tag}{n:05d}.png"))
    p, yaw = base_pose()
    r, ry = ref(state["t"])
    frames.append({"t": round(state["t"], 3), "pos": p.round(4).tolist(), "yaw": round(yaw, 4),
                   "cam_pos": head.round(4).tolist(), "cam_fwd": f.round(4).tolist(),
                   "cam_height": round(float(head[2] - FLOOR_Z), 3), "phase": phase_at(state["t"]),
                   "mirror_frac": round(float(mm.mean()), 4),
                   "track_err_m": round(float(np.linalg.norm(p[:2] - r)), 3)})
    n += 1
    if args.still is not None:
        print("still:", frames[-1])
        break
    if n % 60 == 0:
        print(f"  frame {n}, t {state['t']:.1f}/{t_end:.1f} s, err {frames[-1]['track_err_m']:.2f} m, "
              f"mirror {frames[-1]['mirror_frac']:.2f}, {(time.time() - w0) / n:.2f} s/frame", flush=True)
if args.still is None:
    json.dump({"fps": args.fps, "floor_z": FLOOR_Z, "hfov_deg": 90.0, "frames": frames},
              open(os.path.join(args.out, "frames.json"), "w"))
    errs = [x["track_err_m"] for x in frames]
    print(f"done: {n} frames -> {args.out}; tracking error median {np.median(errs):.3f} max {np.max(errs):.3f} m")
simulation_app.close()
