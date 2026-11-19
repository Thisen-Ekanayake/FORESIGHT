"""Shared stage building for the Isaac Sim demos: the HM3D house from tools/export_hm3d_for_isaac.py as USD meshes,
its mirror meshes with a perfect mirror material, interior lights and a flat physics floor.

Same construction as tools/isaac/render_replay.py (which keeps its own inline copy so the 2026-11-01 replay stays
reproducible), plus an optional height cut for a dollhouse view. Import only from inside Isaac Sim's python, after
SimulationApp has started.
"""
import os

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt


def _textured(stage, house_dir, tex_idx, cache, emission=1.0):
    if tex_idx in cache:
        return cache[tex_idx]
    path = f"/World/Looks/tex_{tex_idx:03d}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    st = UsdShade.Shader.Define(stage, path + "/st")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(stage, path + "/Tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.abspath(os.path.join(house_dir, f"tex_{tex_idx:03d}.jpg")))
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
    rgb = tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    # The scan's textures already carry the home's real lighting: emission keeps them close to the photographs
    # while the RTX lights add shading, the robot's shadow and reflections.
    if emission > 0:  # UsdPreviewSurface has no emission scale: route the texture through a multiply node
        mul = UsdShade.Shader.Define(stage, path + "/EmScale")
        mul.CreateIdAttr("UsdUVTexture")
        mul.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.abspath(os.path.join(house_dir, f"tex_{tex_idx:03d}.jpg")))
        mul.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        mul.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        mul.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(emission, emission, emission, 1.0))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(mul.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    cache[tex_idx] = mat
    return mat


def mirror_material(stage):
    mat = UsdShade.Material.Define(stage, "/World/Looks/Mirror")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/Mirror/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.95, 0.95))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def emissive_material(stage, path, color, intensity=1.0, opacity=1.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    c = Gf.Vec3f(*[float(v) for v in color])
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(c)
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(c * float(intensity))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    if opacity < 1.0:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def build_house(stage, house_dir, cut_z=None, z_range=None, emission=1.0, xy_box=None):
    """Add the house under /World/House. cut_z: drop every triangle whose lowest vertex is above it (a dollhouse
    cut-away). z_range: (lo, hi) keep only triangles whose centroid lies in it (one storey).
    Returns {"mirrors": [prim paths], "n_meshes": int}."""
    UsdGeom.Xform.Define(stage, "/World/House")
    npz = np.load(os.path.join(house_dir, "meshes.npz"))
    keys = sorted({k.split("_")[0] for k in npz.files})
    cache, mirrors = {}, []
    mmat = mirror_material(stage)
    for k in keys:
        name = str(npz[f"{k}_name"])
        pos, uv, idx = npz[f"{k}_pos"], npz[f"{k}_uv"], npz[f"{k}_idx"]
        tri = idx.reshape(-1, 3)
        keep = np.ones(len(tri), bool)
        if cut_z is not None:
            keep &= pos[tri, 2].min(1) < cut_z
        if z_range is not None:
            cz = pos[tri, 2].mean(1)
            keep &= (cz > z_range[0]) & (cz < z_range[1])
        if xy_box is not None:  # (x0, y0, x1, y1): keep triangles whose centroid is inside (a diorama)
            c = pos[tri, :2].mean(1)
            keep &= (c[:, 0] > xy_box[0]) & (c[:, 0] < xy_box[2]) & (c[:, 1] > xy_box[1]) & (c[:, 1] < xy_box[3])
        if not keep.any():
            continue
        tri = tri[keep]
        m = UsdGeom.Mesh.Define(stage, f"/World/House/m{k}")
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pos))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tri), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tri.reshape(-1).astype(np.int32)))
        m.CreateSubdivisionSchemeAttr("none")
        m.CreateDoubleSidedAttr(True)
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(uv))
        if name.startswith("mirror"):
            mirrors.append(str(m.GetPath()))
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mmat)
        else:
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(_textured(stage, house_dir, int(npz[f"{k}_tex"]), cache, emission))
    return {"mirrors": mirrors, "n_meshes": len(keys)}


def add_lights(stage, lo, hi, floor_z, dome=250.0, bulbs=6000.0):
    d = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    d.CreateIntensityAttr(dome)
    for i, (x, y) in enumerate([(lo[0], lo[1]), (hi[0], hi[1]), ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2),
                                (lo[0], hi[1]), (hi[0], lo[1])]):
        s = UsdLux.SphereLight.Define(stage, f"/World/Lights/Ceiling{i}")
        s.CreateRadiusAttr(0.15)
        s.CreateIntensityAttr(bulbs)
        s.CreateColorAttr(Gf.Vec3f(1.0, 0.92, 0.82))
        UsdGeom.Xformable(s.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), floor_z + 2.3))


def add_floor_collider(stage, center_xy, floor_z):
    from isaacsim.core.experimental.utils.stage import define_prim
    define_prim("/World/PhysicsScene", "PhysicsScene")
    g = UsdGeom.Cube.Define(stage, "/World/Ground")
    g.CreateSizeAttr(1.0)
    gx = UsdGeom.Xformable(g.GetPrim())
    gx.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), floor_z - 0.05))
    gx.AddScaleOp().Set(Gf.Vec3f(60.0, 60.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    g.CreateVisibilityAttr("invisible")
    # Go2 policy trained on friction 1.0 / restitution 0 (see render_replay.py).
    gm = UsdShade.Material.Define(stage, "/World/Looks/GroundPhysics")
    pm = UsdPhysics.MaterialAPI.Apply(gm.GetPrim())
    pm.CreateStaticFrictionAttr().Set(1.0)
    pm.CreateDynamicFrictionAttr().Set(1.0)
    pm.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(gm, UsdShade.Tokens.weakerThanDescendants, "physics")


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    z = np.asarray(eye, float) - np.asarray(target, float)
    z /= np.linalg.norm(z) + 1e-9
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    return Gf.Matrix4d(*x, 0.0, *y, 0.0, *z, 0.0, *np.asarray(eye, float), 1.0)


def make_cam(stage, path, focal, aperture=20.955):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(aperture)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.03, 100.0))
    x = UsdGeom.Xformable(cam.GetPrim())
    x.ClearXformOpOrder()
    return x.AddTransformOp()
