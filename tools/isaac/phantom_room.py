"""The Phantom Room: what Depth Anything V2 believes about the 00808 hallway mirror, shown as a point cloud inside a
path-traced cut-away of the house, with the humanoid where it stood.

The points are the head-camera depth of the Mirror Test (tools/isaac/mirror_test.py, tools/isaac/mirror_test_perception.py)
back-projected into the world: DA V2 reads the reflection as a room behind the wall (the phantom room, red). Then each
point slides to where the mirror fill (`foresight.perception.depth.sources.fill_from_border`, on Isaac's mirror mask:
an oracle segmenter) puts it, back onto the mirror plane (cyan).

  /ml/isaac-sim/python.sh tools/isaac/phantom_room.py --run results/isaac/mirror_test --out results/isaac/phantom
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--house", default="results/isaac/00808")
parser.add_argument("--out", required=True)
parser.add_argument("--res", default="1920x1440")
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--renderer", default="PathTracing", choices=["RaytracedLighting", "PathTracing"])
parser.add_argument("--spp", type=int, default=48)
parser.add_argument("--emission", type=float, default=0.25)
parser.add_argument("--max-points", type=int, default=70000)
parser.add_argument("--still", type=float, default=None, help="render one frame at this video time")
parser.add_argument("--robot", default="none", choices=["h1", "none"],
                    help="h1: the robot's USD in its zero joint pose (arms straight out); none: the head camera's path instead")
parser.add_argument("--max-frame", type=int, default=100,
                    help="use the depth of frames up to this one (the walk-up: beyond ~1.5 m from the mirror its base "
                         "is still in view, so the fill has floor below it to borrow from)")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "renderer": args.renderer})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

from house_stage import add_lights, build_house, look_at, make_cam  # noqa: E402

settings = carb.settings.get_settings()
if args.renderer == "PathTracing":
    settings.set("/rtx/pathtracing/spp", args.spp)
    settings.set("/rtx/pathtracing/totalSpp", args.spp)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
settings.set("/rtx/reflections/maxReflectionBounces", 6)
os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

FLOOR_Z = 0.0614
MIRROR_X = -0.10
meta = json.load(open(os.path.join(args.run, "frames.json")))
cl = np.load(os.path.join(args.run, "phantom.npz"))
P_DA, P_FILL, RGB, MIR, FR = cl["da"], cl["fill"], cl["rgb"], cl["mirror"], cl["frame"]
CUT_Z = FLOOR_Z + 1.15  # dollhouse wall height
keep = (FR <= args.max_frame) & (MIR | (P_DA[:, 2] < CUT_Z))  # real-room points only below the cut, like the walls
P_DA, P_FILL, RGB, MIR, FR = P_DA[keep], P_FILL[keep], RGB[keep], MIR[keep], FR[keep]
rng = np.random.default_rng(0)
if len(P_DA) > args.max_points:  # keep every mirror point, subsample the rest
    m_idx = np.nonzero(MIR)[0]
    o_idx = np.nonzero(~MIR)[0]
    o_idx = rng.choice(o_idx, max(0, args.max_points - len(m_idx)), replace=False)
    sel = np.sort(np.concatenate([m_idx, o_idx]))
    P_DA, P_FILL, RGB, MIR, FR = P_DA[sel], P_FILL[sel], RGB[sel], MIR[sel], FR[sel]
GHOST = MIR & (P_DA[:, 0] > MIRROR_X + 0.05)  # mirror pixels DA put behind the mirror plane
print(f"points {len(P_DA)}, mirror {MIR.sum()}, behind the mirror {GHOST.sum()}")

# --- timeline (video seconds) -----------------------------------------------------------------------------------
T_REVEAL0, T_REVEAL1 = 2.5, 8.0  # points appear in capture order
T_FILL0, T_FILL1 = 12.0, 15.5    # phantom points slide onto the mirror plane
T_END = 19.0


def ease(u):
    u = float(np.clip(u, 0, 1))
    return u * u * (3 - 2 * u)


# --- stage ------------------------------------------------------------------------------------------------------
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
# Dollhouse cut: walls end at 1.15 m (the mirror, one tall slab, keeps its full height); only this room's footprint.
info = build_house(stage, args.house, cut_z=CUT_Z, z_range=(FLOOR_Z - 0.5, FLOOR_Z + 3.0),
                   emission=args.emission, xy_box=(-3.5, -2.4, -0.02, 1.3))
print(f"house diorama: {len(info['mirrors'])} mirrors")
add_lights(stage, np.array([-3.0, -1.8]), np.array([-0.6, 0.9]), FLOOR_Z, dome=0.0, bulbs=9000.0)
_dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")  # dim navy surround: the points read against it
_dome.CreateIntensityAttr(60.0)
_dome.CreateColorAttr(Gf.Vec3f(0.35, 0.42, 0.65))

# Point clouds: the real room (non-mirror pixels, image colour via displayColor) and the mirror pixels (one emissive
# colour, animated from red = phantom room to cyan = filled), as two Points prims.
def points_prim(path, n, width):
    p = UsdGeom.Points.Define(stage, path)
    p.CreateWidthsAttr(Vt.FloatArray.FromNumpy(np.full(n, width, np.float32)))
    p.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    return p


def flat_material(path, color, emissive):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    em = sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
    em.Set(Gf.Vec3f(*(np.asarray(color) * emissive)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat, sh, em


ROOM, MIRP = ~MIR, MIR
room = points_prim("/World/CloudRoom", int(ROOM.sum()), 0.014)
room_mat = UsdShade.Material.Define(stage, "/World/Looks/RoomCloud")
_sh = UsdShade.Shader.Define(stage, "/World/Looks/RoomCloud/Surface")
_sh.CreateIdAttr("UsdPreviewSurface")
_rd = UsdShade.Shader.Define(stage, "/World/Looks/RoomCloud/Color")
_rd.CreateIdAttr("UsdPrimvarReader_float3")
_rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
_out = _rd.CreateOutput("result", Sdf.ValueTypeNames.Float3)
_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(_out)
_sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(_out)
_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
room_mat.CreateSurfaceOutput().ConnectToSource(_sh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(room.GetPrim()).Bind(room_mat)
room.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy((RGB[ROOM].astype(np.float32) / 255.0) * 0.45))
ghost = points_prim("/World/CloudMirror", int(MIRP.sum()), 0.017)
ghost_mat, ghost_sh, ghost_em = flat_material("/World/Looks/MirrorCloud", (0.95, 0.06, 0.12), 0.6)
UsdShade.MaterialBindingAPI.Apply(ghost.GetPrim()).Bind(ghost_mat)
col_ghost = np.array([0.95, 0.06, 0.12], np.float32)
col_fixed = np.array([0.05, 0.7, 1.0], np.float32)

# The head camera's path over the frames shown: a trail of white points ending in a brighter one.
if args.robot == "none":
    path = np.array([f["cam_pos"] for f in meta["frames"][: int(FR.max()) + 1]])
    trail = points_prim("/World/CamPath", len(path), 0.035)
    trail.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(path.astype(np.float32)))
    tm, _, _ = flat_material("/World/Looks/CamPath", (1.0, 1.0, 1.0), 1.2)
    UsdShade.MaterialBindingAPI.Apply(trail.GetPrim()).Bind(tm)
    head = points_prim("/World/CamNow", 1, 0.12)
    head.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(path[-1:].astype(np.float32)))
    UsdShade.MaterialBindingAPI.Apply(head.GetPrim()).Bind(tm)

# The humanoid where it stood for the look/turn/step phases (a static pose, no physics).
if args.robot == "h1":
    from isaacsim.storage.native import get_assets_root_path
    look = meta["frames"][int(FR.max())]  # where it stood at the last frame whose depth is shown
    h1 = stage.DefinePrim("/World/H1", "Xform")
    h1.GetReferences().AddReference(get_assets_root_path() + "/Isaac/Robots/Unitree/H1/h1.usd")
    xf = UsdGeom.Xformable(h1)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(look["pos"][0], look["pos"][1], look["pos"][2]))
    xf.AddRotateZOp().Set(math.degrees(look["yaw"]))

cam_op = make_cam(stage, "/World/Cam", 18.0)
W, H = (int(v) for v in args.res.split("x"))
rp = rep.create.render_product("/World/Cam", resolution=(W, H))
an = rep.AnnotatorRegistry.get_annotator("rgb")
an.attach(rp)

frame_order = np.argsort(np.argsort(FR, kind="stable"), kind="stable") / max(len(FR) - 1, 1)  # 0..1 capture order
CENTER = np.array([-0.4, -0.4, FLOOR_Z + 0.6])  # between the robot (x ~ -2.1) and the phantom room (x 0.2-3)


def camera(t):
    """Slow orbit around the hallway, from high behind-left to the side where the phantom room shows best."""
    a = math.radians(-122 + 50 * ease(t / T_END))
    r, z = 6.6 - 0.7 * ease(t / T_END), 6.2 - 0.8 * ease(t / T_END)
    eye = CENTER + np.array([r * math.cos(a), r * math.sin(a), z])
    return look_at(eye, CENTER + np.array([0.2, 0, -0.4]))


def set_cloud(t):
    reveal = -1.0 if t < T_REVEAL0 else ease((t - T_REVEAL0) / (T_REVEAL1 - T_REVEAL0))
    shown = frame_order <= reveal
    u = ease((t - T_FILL0) / (T_FILL1 - T_FILL0))
    p = P_DA.copy()
    p[MIR] = (1 - u) * P_DA[MIR] + u * P_FILL[MIR]
    p[~shown] = np.array([0, 0, -50.0])  # hidden: parked far below the floor
    room.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(p[ROOM].astype(np.float32)))
    ghost.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(p[MIRP].astype(np.float32)))
    c = (1 - u) * col_ghost + u * col_fixed
    ghost_sh.GetInput("diffuseColor").Set(Gf.Vec3f(*c.tolist()))
    ghost_em.Set(Gf.Vec3f(*(c * 0.6).tolist()))
    return {"shown": int(shown.sum()), "fill": round(u, 3)}


times = [args.still] if args.still is not None else list(np.arange(0, T_END, 1.0 / args.fps))
# No timeline here (the H1 is a static pose), so frames are captured with Replicator's explicit step: a plain
# app update leaves the annotator buffer empty.
cam_op.Set(camera(times[0]))
set_cloud(times[0])
for _ in range(20):
    simulation_app.update()
rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
log, w0 = [], time.time()
for n, t in enumerate(times):
    cam_op.Set(camera(t))
    st = set_cloud(t)
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    img = np.asarray(an.get_data())
    if img.ndim != 3:
        print("  no image at", t, getattr(img, "shape", None), img.dtype, flush=True)
        continue
    name = "still.jpg" if args.still is not None else os.path.join("frames", f"{n:05d}.jpg")
    Image.fromarray(np.ascontiguousarray(img[..., :3])).save(os.path.join(args.out, name), quality=94)
    log.append({"n": n, "t": round(float(t), 3), **st})
    if n % 30 == 0:
        print(f"  frame {n}/{len(times)} t {t:.1f} {st} {(time.time() - w0) / (n + 1):.2f} s/frame", flush=True)
json.dump({"fps": args.fps, "frames": log, "max_frame": int(FR.max()), "cam_x_range": [meta["frames"][int(FR.min())]["cam_pos"][0], meta["frames"][int(FR.max())]["cam_pos"][0]], "n_points": int(len(P_DA)), "n_mirror": int(MIR.sum()),
           "n_ghost": int(GHOST.sum()), "t_fill": [T_FILL0, T_FILL1], "t_reveal": [T_REVEAL0, T_REVEAL1]},
          open(os.path.join(args.out, "phantom_frames.json" if args.still is None else "still.json"), "w"))
print("done", len(log), "frames")
simulation_app.close()
