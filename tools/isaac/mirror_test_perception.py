#!/usr/bin/env python
"""Perception on the Mirror Test's Isaac Sim head-camera frames (tools/isaac/mirror_test.py), run afterwards in the
habitat env because Isaac and the models don't fit in 8 GB together.

Per frame: Depth Anything V2 (the indoor-metric checkpoint FORESIGHT navigates with), scaled by FloorScale at the
head camera's height, against Isaac's ground-truth depth. Every `--vlm-every` frames: Qwen3-VL-2B's mirror presence
check (the yes/no prompt FORESIGHT v2/v3 use), "do you see a robot?", grounding boxes for "mirror" and "robot", and a
free description at a few key frames.

The self-motion test: while the robot side-steps parallel to the mirror, a real second robot standing where the
reflection is would slide across the image like the rest of the room; a reflection keeps the same image position,
because it moves with the camera. Both predictions are computed from the camera poses (the reflection's position is
the mirror image of the robot's own); the observed track is Qwen's "robot" box centre.

  python tools/isaac/mirror_test_perception.py results/isaac/mirror_test
Writes perception.json, da/NNNNN.npy (float16, scaled), and phantom.npz (world points for the Phantom Room render).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from foresight.perception.depth.sources import fill_from_border  # noqa: E402
from foresight.planning.local_map import FloorScale  # noqa: E402

MIRROR_X = -0.10
PRESENT = "Is there a mirror in this image? Answer yes or no."
ROBOT = "Is there a robot in this image? Answer yes or no."
SELF = ("You are a humanoid robot and this image is from the camera in your head. Is the robot in this image you, "
        "seen in a mirror? Answer yes or no.")
DESCRIBE = "You are the camera on a humanoid robot's head. In one or two sentences: what is directly in front of you?"


def intrinsics(w, hfov_deg=90.0):
    return (w / 2) / np.tan(np.deg2rad(hfov_deg) / 2)


def backproject(depth, cam_pos, fwd, f, stride):
    """Optical-axis depth (H, W) -> world points (N, 3), z up; camera level, looking along fwd."""
    h, w = depth.shape
    v, u = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
    z = depth[v, u]
    x_r = (u - w / 2 + 0.5) / f * z  # right
    y_d = (v - h / 2 + 0.5) / f * z  # down
    fwd = np.asarray(fwd, float)
    right = np.array([fwd[1], -fwd[0], 0.0])
    p = cam_pos[None, None, :] + z[..., None] * fwd + x_r[..., None] * right - y_d[..., None] * np.array([0, 0, 1.0])
    return p.reshape(-1, 3), v.reshape(-1), u.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--vlm-every", type=int, default=3)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--cloud-every", type=int, default=6, help="frames between point-cloud samples")
    ap.add_argument("--stride", type=int, default=6)
    a = ap.parse_args()
    root = Path(a.run_dir)
    meta = json.load(open(root / "frames.json"))
    frames = meta["frames"]
    (root / "da").mkdir(exist_ok=True)

    from foresight.perception.depth.depth_anything import ZeroShotDepthEstimator
    est = ZeroShotDepthEstimator()
    fs = FloorScale(hfov_deg=90.0, cam_height_m=frames[0]["cam_height"])
    out = []
    cloud = {"da": [], "fill": [], "gt": [], "rgb": [], "mirror": [], "frame": []}
    for i, fr in enumerate(frames):
        rgb = np.asarray(Image.open(root / "eye" / f"{i:05d}.jpg").convert("RGB"))
        gt = np.load(root / "eye_depth" / f"{i:05d}.npy").astype(np.float32)
        mm = np.asarray(Image.open(root / "eye_mirror" / f"{i:05d}.png")) > 127
        raw = est.estimate(rgb)
        fs.cam_h = fr["cam_height"]
        s = fs.update(raw)
        da = raw * s
        np.save(root / "da" / f"{i:05d}.npy", da.astype(np.float16))
        ok = np.isfinite(gt) & (gt > 0.05) & (gt < 20)
        rec = {"i": i, "t": fr["t"], "phase": fr["phase"], "scale": round(float(s), 4),
               "mirror_frac": fr["mirror_frac"],
               "absrel": round(float(np.mean(np.abs(da[ok] - gt[ok]) / gt[ok])), 4)}
        if mm.sum() > 50:
            mok = mm & ok
            rec["mirror_da_m"] = round(float(np.median(da[mok])), 3)
            rec["mirror_gt_m"] = round(float(np.median(gt[mok])), 3)
            rec["mirror_excess_m"] = round(float(np.median(da[mok] - gt[mok])), 3)
        out.append(rec)
        if i % a.cloud_every == 0 and fr["phase"] in ("approach", "look", "turn", "step"):
            from scipy.ndimage import find_objects, label
            lab, _ = label(mm)
            regions = [(sl, lab[sl] == j) for j, sl in enumerate(find_objects(lab), start=1) if (lab[sl] == j).sum() >= 20]
            filled = fill_from_border(da, regions, mm) if regions else da
            f = intrinsics(rgb.shape[1])
            cp = np.array(fr["cam_pos"])
            p_da, v, u = backproject(da, cp, fr["cam_fwd"], f, a.stride)
            p_fill, _, _ = backproject(filled, cp, fr["cam_fwd"], f, a.stride)
            p_gt, _, _ = backproject(np.where(ok, gt, 0), cp, fr["cam_fwd"], f, a.stride)
            keep = ok[v, u] & (da[v, u] < 8)
            for k, arr in (("da", p_da), ("fill", p_fill), ("gt", p_gt)):
                cloud[k].append(arr[keep])
            cloud["rgb"].append(rgb[v, u][keep])
            cloud["mirror"].append(mm[v, u][keep])
            cloud["frame"].append(np.full(keep.sum(), i))
        if i % 60 == 0:
            print(f"  depth {i}/{len(frames)} scale {s:.2f} absrel {rec['absrel']:.3f} "
                  f"mirror excess {rec.get('mirror_excess_m', '-')}", flush=True)
    del est
    import torch
    torch.cuda.empty_cache()
    np.savez_compressed(root / "phantom.npz", **{k: np.concatenate(v) for k, v in cloud.items()})
    print("cloud points:", sum(len(x) for x in cloud["da"]))

    if not a.no_vlm:
        from foresight.perception.vlm.qwen_vl import Qwen3VLSceneReasoner
        vlm = Qwen3VLSceneReasoner()
        key = {}
        for ph in ("approach", "look", "step", "close"):  # one description per phase: its middle frame
            idx = [r["i"] for r in out if r["phase"] == ph]
            if idx:
                key[idx[len(idx) // 2]] = ph
        for rec in out:
            i = rec["i"]
            if i % a.vlm_every and i not in key:
                continue
            rgb = np.asarray(Image.open(root / "eye" / f"{i:05d}.jpg").convert("RGB"))
            rec["p_mirror"] = round(float(vlm.option_probs(rgb, PRESENT, ["yes", "no"])[0]), 4)
            rec["p_robot"] = round(float(vlm.option_probs(rgb, ROBOT, ["yes", "no"])[0]), 4)
            rec["p_self"] = round(float(vlm.option_probs(rgb, SELF, ["yes", "no"])[0]), 4)
            rec["box_robot"] = vlm.locate(rgb, "robot")
            rec["box_mirror"] = vlm.locate(rgb, "mirror")
            if i in key:
                rec["describe"] = vlm.describe(rgb, prompt=DESCRIBE, max_new_tokens=80)
                print(f"  [{key[i]}] {rec['describe']}", flush=True)
            if i % 30 == 0:
                print(f"  vlm {i}: P(mirror) {rec['p_mirror']:.2f} P(robot) {rec['p_robot']:.2f} "
                      f"P(self) {rec['p_self']:.2f} robot boxes {rec['box_robot']}", flush=True)

    # Self-motion test predictions (image x of the reflection's head, 0-1000 like Qwen's boxes).
    for rec in out:
        fr = frames[rec["i"]]
        cp, fwd = np.array(fr["cam_pos"]), np.array(fr["cam_fwd"])
        refl = cp.copy()
        refl[0] = 2 * MIRROR_X - cp[0]  # mirror image of the robot's own head
        step0 = next(f for f in frames if f["phase"] == "step")
        other = np.array(step0["cam_pos"])
        other[0] = 2 * MIRROR_X - other[0]  # a second robot standing still where the reflection was at the step start
        right = np.array([fwd[1], -fwd[0], 0.0])
        for name, q in (("x_reflection", refl), ("x_other_robot", other)):
            d = q - cp
            z, xr = d @ fwd, d @ right
            rec[name] = round(float(500 + 500 * xr / max(z, 1e-3)), 1)  # f = w/2 at 90 deg
    json.dump({"frames": out, "prompts": {"mirror": PRESENT, "robot": ROBOT, "self": SELF, "describe": DESCRIBE}},
              open(root / "perception.json", "w"))
    print("wrote", root / "perception.json")


if __name__ == "__main__":
    main()
