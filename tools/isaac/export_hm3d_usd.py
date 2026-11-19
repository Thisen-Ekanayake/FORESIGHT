"""Write an HM3D home (from tools/export_hm3d_for_isaac.py) as a standalone USD with triangle-mesh colliders, floor
moved to z = 0, for the live G1 runs (tools/isaac/g1_live.py --scene hm3d). Mirrors get the perfect-mirror material.

  /ml/IsaacLab/env_isaaclab/bin/python tools/isaac/export_hm3d_usd.py results/isaac/00808 results/isaac/00808_collision.usd \
      --floor-z 0.0614 --z-range -0.5 3.0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from house_stage import build_house  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("house")
ap.add_argument("out")
ap.add_argument("--floor-z", type=float, default=0.0614)
ap.add_argument("--z-range", nargs=2, type=float, default=[-0.5, 3.0], help="keep this storey (relative to floor)")
ap.add_argument("--emission", type=float, default=0.25)
a = ap.parse_args()

stage = Usd.Stage.CreateNew(a.out)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
info = build_house(stage, a.house, z_range=(a.floor_z + a.z_range[0], a.floor_z + a.z_range[1]), emission=a.emission)
UsdGeom.Xformable(stage.GetPrimAtPath("/World/House")).AddTranslateOp().Set(Gf.Vec3d(0, 0, -a.floor_z))
n = 0
for p in stage.Traverse():
    if p.IsA(UsdGeom.Mesh):
        UsdPhysics.CollisionAPI.Apply(p)
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("none")
        n += 1
stage.GetRootLayer().Save()
print(f"wrote {a.out}: {n} collider meshes, {len(info['mirrors'])} mirrors")
