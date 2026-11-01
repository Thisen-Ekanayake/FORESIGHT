"""Replay a recorded FORESIGHT run in Isaac Sim: the same HM3D house, RTX-rendered, with a Unitree Go2 walking the
recorded path.

What this is and isn't: the navigation decisions were made in Habitat-Sim (tools/run_object_search.py); here a
Go2 (Isaac Sim's own flat-terrain walking policy) is steered by a tracking controller along that recorded path,
so its legs, contacts and body motion are simulated by PhysX while the route and the stops are the run's own.
The house is the HM3D scan converted by tools/export_hm3d_for_isaac.py; its 18 mirror surfaces get a real mirror
material, so reflections are ray traced rather than baked into a texture.

Run with Isaac Sim's python (GPU needed; nothing else may hold the GPU):
  /ml/isaac-sim/python.sh tools/isaac/render_replay.py --run <run>_isaac.json --house results/isaac/00808 \
      --out results/isaac/replay/<name>

Outputs <out>/chase/NNNNN.jpg, <out>/eye/NNNNN.jpg and <out>/frames.json (sim time + robot pose per frame) for
the compositor.
"""
import argparse
import json
import math
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="JSON from tools/isaac/export_run.py")
parser.add_argument("--house", default="results/isaac/00808", help="dir from tools/export_hm3d_for_isaac.py")
parser.add_argument("--out", required=True)
parser.add_argument("--fps", type=float, default=15.0, help="frames per second of *sim* time")
parser.add_argument("--chase-res", default="1920x1280")
parser.add_argument("--eye-res", default="640x640")
parser.add_argument("--t-from", type=float, default=0.0)
parser.add_argument("--t-to", type=float, default=None)
parser.add_argument("--max-frames", type=int, default=None)
parser.add_argument("--renderer", default="RaytracedLighting", choices=["RaytracedLighting", "PathTracing"])
parser.add_argument("--spp", type=int, default=64, help="path tracing samples per pixel")
parser.add_argument("--save-usd", default=None, help="also save the built stage here")
parser.add_argument("--pose-check", action="store_true",
                    help="render the head camera at the run's start pose, as Habitat's 1.5 m camera saw it, to <out>/pose_check.jpg")
parser.add_argument("--still", default=None, help="render one chase frame at this sim time to <out>/still.jpg and exit")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "renderer": args.renderer})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.deprecation_manager import import_module  # noqa: E402
from isaacsim.core.experimental.utils.stage import define_prim  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

torch = import_module("torch")
settings = carb.settings.get_settings()
if args.renderer == "PathTracing":
    settings.set("/rtx/pathtracing/spp", args.spp)
    settings.set("/rtx/pathtracing/totalSpp", args.spp)
settings.set("/rtx/post/aa/op", 3)  # DLAA-style antialiasing in real-time mode
settings.set("/rtx/reflections/maxRoughness", 0.3)
settings.set("/rtx/translucency/maxRefractionBounces", 4)

run = json.load(open(args.run))
os.makedirs(os.path.join(args.out, "chase"), exist_ok=True)
os.makedirs(os.path.join(args.out, "eye"), exist_ok=True)


# --- frames: Habitat (x, y up, z; forward -z) -> Isaac (x, -z, y; z up) ------------------------------------
def h2i(p):
    return np.array([p[0], -p[2], p[1]], float)


def yaw_of(fwd_xz):
    return math.atan2(-fwd_xz[1], fwd_xz[0])


T = np.array([0.0] + [k["t"] for k in run["ticks"]])
P = np.stack([h2i(run["start"]["pos"])] + [h2i(k["pos"]) for k in run["ticks"]])
Y = np.unwrap(np.array([yaw_of(run["ticks"][0]["fwd"])] + [yaw_of(k["fwd"]) for k in run["ticks"]]))
floor_z = float(P[0, 2])


def ref(t):
    t = float(np.clip(t, T[0], T[-1]))
    return np.array([np.interp(t, T, P[:, i]) for i in range(3)]), float(np.interp(t, T, Y))


# --- stage: the house ---------------------------------------------------------------------------------------------
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
# HM3D GLBs are stored z-up (the dataset config's stage "up" is [0, 0, 1]; Habitat rotates them to y-up on load),
# so their raw coordinates already are this stage's frame: Habitat (x, y, z) = GLB (x, z, -y).
house = UsdGeom.Xform.Define(stage, "/World/House")
mesh_npz = np.load(os.path.join(args.house, "meshes.npz"))
keys = sorted({k.split("_")[0] for k in mesh_npz.files})
materials = {}


def textured_material(tex_idx):
    if tex_idx in materials:
        return materials[tex_idx]
    path = f"/World/Looks/tex_{tex_idx:03d}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    st = UsdShade.Shader.Define(stage, path + "/st")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(stage, path + "/Tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.abspath(os.path.join(args.house, f"tex_{tex_idx:03d}.jpg")))
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
    rgb = tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    # The scan's textures already carry the home's real lighting: a little emission keeps them close to the
    # photographs while the RTX lights add shading, shadows of the robot and reflections.
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    materials[tex_idx] = mat
    return mat


mirror_mat = UsdShade.Material.Define(stage, "/World/Looks/Mirror")
_ms = UsdShade.Shader.Define(stage, "/World/Looks/Mirror/Surface")
_ms.CreateIdAttr("UsdPreviewSurface")
_ms.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.95, 0.95))
_ms.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
_ms.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
mirror_mat.CreateSurfaceOutput().ConnectToSource(_ms.ConnectableAPI(), "surface")
n_mirror = 0
for k in keys:
    name = str(mesh_npz[f"{k}_name"])
    pos, uv, idx = mesh_npz[f"{k}_pos"], mesh_npz[f"{k}_uv"], mesh_npz[f"{k}_idx"]
    m = UsdGeom.Mesh.Define(stage, f"/World/House/m{k}")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pos))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(idx) // 3, 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(uv))
    is_mirror = name.startswith("mirror")
    n_mirror += is_mirror
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mirror_mat if is_mirror else textured_material(int(mesh_npz[f"{k}_tex"])))
print(f"house: {len(keys)} meshes, {len(materials)} textures, {n_mirror} mirrors")

# --- lights -------------------------------------------------------------------------------------------------------
dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
dome.CreateIntensityAttr(250.0)
lo, hi = P.min(0), P.max(0)
for i, (x, y) in enumerate([(lo[0], lo[1]), (hi[0], hi[1]), ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2), (lo[0], hi[1]), (hi[0], lo[1])]):
    s = UsdLux.SphereLight.Define(stage, f"/World/Lights/Ceiling{i}")
    s.CreateRadiusAttr(0.15)
    s.CreateIntensityAttr(6000.0)
    s.CreateColorAttr(Gf.Vec3f(1.0, 0.92, 0.82))  # warm interior bulbs
    s.CreateEnableColorTemperatureAttr(False)
    UsdGeom.Xformable(s.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), floor_z + 2.3))

# --- physics: a flat floor at the scan's floor height (the scan mesh itself is visual only) ---------------------
define_prim("/World/PhysicsScene", "PhysicsScene")
ground = UsdGeom.Cube.Define(stage, "/World/Ground")
ground.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(ground.GetPrim())
gx.AddTranslateOp().Set(Gf.Vec3d(float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), floor_z - 0.05))
gx.AddScaleOp().Set(Gf.Vec3f(60.0, 60.0, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
ground.CreateVisibilityAttr("invisible")
if args.save_usd:
    stage.GetRootLayer().Export(args.save_usd)

RenderingManager.set_dt(1.0 / args.fps)
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(1.0 / 200.0)

t0 = args.t_from
p0, y0 = ref(t0)
go2 = Go2FlatTerrainPolicy(prim_path="/World/Go2", position=[float(p0[0]), float(p0[1]), floor_z + 0.42],
                           orientation=[math.cos(y0 / 2), 0.0, 0.0, math.sin(y0 / 2)])
command = torch.zeros(3)
state = {"first": True, "t": t0}


def base_pose():
    pos, quat = go2.robot.get_world_poses()
    p = np.asarray(pos.numpy()[0], float)
    w, x, y, z = np.asarray(quat.numpy()[0], float)
    return p, math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def track(t):
    """Body-frame (vx, vy, wz) steering the Go2 onto the recorded pose a little ahead in time."""
    p, yaw = base_pose()
    tgt, yaw_t = ref(t + 0.25)
    nxt, _ = ref(t + 0.5)
    v_ff = (nxt - tgt) / 0.25
    e = tgt - p
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = c * e[0] + s * e[1], -s * e[0] + c * e[1]
    fx, fy = c * v_ff[0] + s * v_ff[1], -s * v_ff[0] + c * v_ff[1]
    dyaw = (yaw_t - yaw + math.pi) % (2 * math.pi) - math.pi
    return [float(np.clip(fx + 2.0 * ex, -0.8, 0.8)), float(np.clip(fy + 2.0 * ey, -0.5, 0.5)),
            float(np.clip(2.5 * dyaw, -1.2, 1.2))]


def on_physics_step(step_size, context):
    if state["first"]:
        go2.initialize()
        state["first"] = False
        return
    state["t"] += step_size
    cmd = track(state["t"])
    go2.forward(step_size, torch.tensor(cmd))


# --- cameras -----------------------------------------------------------------------------------------------------------
def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    z = np.asarray(eye, float) - np.asarray(target, float)
    z /= np.linalg.norm(z) + 1e-9
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    return Gf.Matrix4d(*x, 0.0, *y, 0.0, *z, 0.0, *np.asarray(eye, float), 1.0)


def make_cam(path, focal):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.03, 100.0))
    x = UsdGeom.Xformable(cam.GetPrim())
    x.ClearXformOpOrder()
    return x.AddTransformOp()


chase_op, eye_op = make_cam("/World/ChaseCam", 12.0), make_cam("/World/EyeCam", 10.5)
cw, ch = (int(v) for v in args.chase_res.split("x"))
ew, eh = (int(v) for v in args.eye_res.split("x"))
rp_chase = rep.create.render_product("/World/ChaseCam", resolution=(cw, ch))
rp_eye = rep.create.render_product("/World/EyeCam", resolution=(ew, eh))
an_chase = rep.AnnotatorRegistry.get_annotator("rgb")
an_chase.attach(rp_chase)
an_eye = rep.AnnotatorRegistry.get_annotator("rgb")
an_eye.attach(rp_eye)
cam_state = {"eye": None, "look": None}


def place_cameras(t, back_m: float = 1.9, height_m: float = 1.25, straight_m: float = 0.12):
    """Chase camera on the route the robot itself walked, at the furthest earlier point (up to `back_m` away) from
    which the robot walked a straight line to where it is now: that line of sight is free space by construction
    (a point merely on the route can look through a wall after a turn). 1.25 m up, looking just ahead of the
    robot; smoothed. Eye camera on the Go2's head."""
    p, yaw = base_pose()
    k = min(int(np.searchsorted(T, t)), len(T) - 1)
    best = None
    for j in range(k, max(0, k - 600) - 1, -1):
        d = np.linalg.norm(P[j, :2] - p[:2])
        if d < 0.3:
            continue
        u = (p[:2] - P[j, :2]) / d
        seg = P[j:k + 1, :2] - P[j, :2]
        off = np.abs(seg[:, 0] * u[1] - seg[:, 1] * u[0])
        if off.max() > straight_m:
            break
        best = P[j]
        if d >= back_m:
            break
    if best is None or np.linalg.norm(best[:2] - p[:2]) < 0.8:
        # Nothing usable behind (e.g. the opening spin): a point it walks to *later* in a straight line, looking back.
        for j in range(k, min(len(T), k + 600)):
            d = np.linalg.norm(P[j, :2] - p[:2])
            if d < 0.3:
                continue
            u = (P[j, :2] - p[:2]) / d
            seg = P[k:j + 1, :2] - p[:2]
            if np.abs(seg[:, 0] * u[1] - seg[:, 1] * u[0]).max() > straight_m:
                break
            if d >= 0.8:
                best = P[j]
            if d >= back_m:
                break
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    if best is not None and np.linalg.norm(best[:2] - p[:2]) >= 0.8:
        eye = np.array([best[0], best[1], floor_z + height_m])
    elif cam_state["eye"] is not None:
        eye = cam_state["eye"]  # turning on the spot / just turned a corner: hold the camera
    else:
        eye = np.array([*(p[:2] - f[:2] * 1.2), floor_z + height_m])
    look = np.array([p[0], p[1], floor_z + 0.3]) + f * 0.6
    a = 0.08 if cam_state["eye"] is not None else 1.0
    cam_state["eye"] = eye if cam_state["eye"] is None else (1 - a) * cam_state["eye"] + a * eye
    cam_state["look"] = look if cam_state["look"] is None else (1 - a) * cam_state["look"] + a * look
    chase_op.Set(look_at(cam_state["eye"], cam_state["look"]))
    head = p + f * 0.42 + np.array([0, 0, 0.12])
    eye_op.Set(look_at(head, head + f - np.array([0, 0, 0.12])))


_cb = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0.0)
timeline.set_end_time(1.0e7)
timeline.set_looping(False)
timeline.set_play_every_frame(True)
timeline.play()
simulation_app.update()
for _ in range(20):  # let the Go2 settle onto its feet before the replay clock starts
    place_cameras(t0)
    simulation_app.update()
state["t"] = t0

if args.pose_check:  # same pose, height and 90 deg view as the Habitat camera at the start: compare the two frames
    f = np.array([math.cos(Y[0]), math.sin(Y[0]), 0.0])
    eye = P[0] + np.array([0, 0, 1.5])
    for _ in range(30):
        eye_op.Set(look_at(eye, eye + f))
        chase_op.Set(look_at(eye - f * 0.01, eye + f))
        simulation_app.update()
    Image.fromarray(np.ascontiguousarray(np.asarray(an_eye.get_data())[..., :3])).save(os.path.join(args.out, "pose_check.jpg"))
    Image.fromarray(np.ascontiguousarray(np.asarray(an_chase.get_data())[..., :3])).save(os.path.join(args.out, "pose_check_wide.jpg"))
    simulation_app.close()
    sys.exit(0)

t_end = min(args.t_to or T[-1], T[-1])
frames, n = [], 0
w0 = time.time()
while simulation_app.is_running() and state["t"] < t_end and (args.max_frames is None or n < args.max_frames):
    place_cameras(state["t"])
    simulation_app.update()
    if args.renderer == "PathTracing":
        for _ in range(3):
            simulation_app.update()
    c, e = np.asarray(an_chase.get_data()), np.asarray(an_eye.get_data())
    if c.ndim != 3 or c.shape[0] != ch or e.ndim != 3:
        continue
    if args.still is not None and state["t"] < float(args.still):
        continue
    Image.fromarray(np.ascontiguousarray(c[..., :3])).save(os.path.join(args.out, "chase", f"{n:05d}.jpg"), quality=93)
    Image.fromarray(np.ascontiguousarray(e[..., :3])).save(os.path.join(args.out, "eye", f"{n:05d}.jpg"), quality=93)
    p, yaw = base_pose()
    r, _ = ref(state["t"])
    frames.append({"t": round(state["t"], 3), "pos": p.round(3).tolist(), "yaw": round(yaw, 3),
                   "track_err_m": round(float(np.linalg.norm(p[:2] - r[:2])), 3)})
    n += 1
    if args.still is not None:
        break
    if n % 50 == 0:
        print(f"  frame {n}, sim t {state['t']:.1f}/{t_end:.1f} s, track err {frames[-1]['track_err_m']:.2f} m, "
              f"{(time.time() - w0) / n:.2f} s/frame", flush=True)
json.dump({"fps": args.fps, "frames": frames, "floor_z": floor_z}, open(os.path.join(args.out, "frames.json"), "w"))
errs = [f["track_err_m"] for f in frames]
print(f"done: {n} frames -> {args.out}; tracking error median {np.median(errs):.3f} m, max {np.max(errs):.3f} m")
simulation_app.close()
