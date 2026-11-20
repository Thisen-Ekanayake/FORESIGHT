"""Inspect an NVIDIA scene USD for the live runs: a top-down plan (all geometry between 0.2 and 1.8 m height) and the
glass/mirror panels (prims whose name or bound material mentions glass/mirror/window), with their bounds and
material settings, so start/goal poses can be chosen around a real glass trap.

  tools/isaac/isaaclab_run.sh tools/isaac/inspect_scene.py isaac_assets/Isaac/Environments/Office/office.usd \
      results/isaac/scenes/office
Writes plan.png (5 cm/px, glass in orange), panels.json.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom, UsdShade

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("usd")
ap.add_argument("out")
ap.add_argument("--floor-z", type=float, default=None, help="storey floor height (default: 5th percentile of prim bottoms)")
ap.add_argument("--region", nargs=4, type=float, default=None, metavar=("X0", "Y0", "X1", "Y1"), help="plan only this area")
a = ap.parse_args()
usd, out = a.usd, a.out
os.makedirs(out, exist_ok=True)
stage = Usd.Stage.Open(usd)
up = UsdGeom.GetStageUpAxis(stage)
mpu = UsdGeom.GetStageMetersPerUnit(stage)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
KEYS = ("glass", "mirror", "window", "transparent")


def material_info(prim):
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return None, {}
    params = {}
    for sh in Usd.PrimRange(mat.GetPrim()):
        if sh.IsA(UsdShade.Shader):
            for inp in UsdShade.Shader(sh).GetInputs():
                n = inp.GetBaseName()
                if any(k in n.lower() for k in ("opacity", "transmi", "ior", "glass", "roughness", "thin_walled", "enable_opacity")):
                    v = inp.Get()
                    params[n] = v if isinstance(v, (int, float, bool, str)) or v is None else str(v)
    return str(mat.GetPath()), params


boxes, panels = [], []
for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Gprim):
        continue
    if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
        continue
    bb = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bb.IsEmpty():
        continue
    lo, hi = np.array(bb.GetMin()) * mpu, np.array(bb.GetMax()) * mpu
    if up == "Y":  # convert to z-up for the plan
        lo, hi = lo[[0, 2, 1]] * [1, -1, 1], hi[[0, 2, 1]] * [1, -1, 1]
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    name = str(prim.GetPath())
    mpath, params = material_info(prim)
    is_panel = any(k in (name + " " + (mpath or "")).lower() for k in KEYS)
    boxes.append((lo, hi, is_panel))
    if is_panel:
        panels.append({"prim": name, "material": mpath, "params": params, "min": lo.round(3).tolist(),
                       "max": hi.round(3).tolist(), "size": (hi - lo).round(3).tolist()})

all_lo = np.min([b[0] for b in boxes], 0)
all_hi = np.max([b[1] for b in boxes], 0)
floor_z = a.floor_z if a.floor_z is not None else float(np.percentile([b[0][2] for b in boxes], 5))
if a.region:
    all_lo = np.array([a.region[0], a.region[1], all_lo[2]])
    all_hi = np.array([a.region[2], a.region[3], all_hi[2]])
res = 0.05
W, H = int((all_hi[0] - all_lo[0]) / res) + 1, int((all_hi[1] - all_lo[1]) / res) + 1
img = Image.new("RGB", (W, H), (16, 20, 30))
d = ImageDraw.Draw(img)


def px(x, y):
    return ((x - all_lo[0]) / res, (all_hi[1] - y) / res)


for lo, hi, is_panel in sorted(boxes, key=lambda b: b[2]):
    if hi[2] < floor_z + 0.2 or lo[2] > floor_z + 1.8:
        continue
    if (hi[0] - lo[0]) * (hi[1] - lo[1]) > 40:  # skip floors/ceilings/whole-room shells
        continue
    a, b = px(lo[0], hi[1]), px(hi[0], lo[1])
    d.rectangle([a, b], outline=(255, 160, 40) if is_panel else (150, 160, 180), width=2 if is_panel else 1)
for x in range(int(all_lo[0]), int(all_hi[0]) + 1):  # 1 m grid with labels every 5 m
    d.line([px(x, all_lo[1]), px(x, all_hi[1])], fill=(40, 48, 70) if x % 5 else (70, 80, 120))
    if x % 5 == 0:
        d.text(px(x, all_hi[1]), str(x), fill=(255, 255, 0))
for y in range(int(all_lo[1]), int(all_hi[1]) + 1):
    d.line([px(all_lo[0], y), px(all_hi[0], y)], fill=(40, 48, 70) if y % 5 else (70, 80, 120))
    if y % 5 == 0:
        d.text(px(all_lo[0], y), str(y), fill=(255, 255, 0))
img.save(os.path.join(out, "plan.png"))
json.dump({"usd": usd, "up": up, "meters_per_unit": mpu, "bounds_min": all_lo.round(2).tolist(),
           "bounds_max": all_hi.round(2).tolist(), "floor_z": round(floor_z, 3), "panels": panels},
          open(os.path.join(out, "panels.json"), "w"), indent=1)
print(f"{usd}: up {up}, {mpu} m/unit, bounds {all_lo.round(1)} .. {all_hi.round(1)}, floor z {floor_z:.2f}, "
      f"{len(boxes)} prims, {len(panels)} glass/mirror/window panels -> {out}")
