#!/usr/bin/env python
"""Qwen3-VL-2B (bf16, the evaluated model) on the office head-camera shot (results/isaac/film/office/o_pov): P("Is there a
glass wall in this image?") and its boxes on a few frames, for the video's GLASS label. Writes
results/isaac/film/office_vlm.json. Run after the renders (needs the GPU to itself)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from foresight.perception.vlm.qwen_vl import PRESENT_PROMPT, Qwen3VLSceneReasoner  # noqa: E402

d = ROOT / "results/isaac/film/office/o_pov"
frames = sorted(d.glob("[0-9]*.jpg"))
vlm = Qwen3VLSceneReasoner()
res = []
for f in frames[:: max(1, len(frames) // 4)][:4]:
    img = np.asarray(Image.open(f).convert("RGB").resize((640, 640)))
    p = float(vlm.option_probs(img, PRESENT_PROMPT.format(name="glass wall"), ["yes", "no"])[0])
    res.append({"frame": f.name, "p_glass": round(p, 3), "boxes": vlm.locate(img, "glass wall")})
    print(res[-1], flush=True)
first = res[0]
json.dump({"p_glass": first["p_glass"], "frames": res}, open(ROOT / "results/isaac/film/office_vlm.json", "w"), indent=1)
