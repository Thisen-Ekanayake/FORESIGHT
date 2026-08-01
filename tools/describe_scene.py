#!/usr/bin/env python
"""Run zero-shot VLM scene reasoning (Qwen3-VL-2B) on a captured RGB frame."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foresight.perception.vlm.qwen_vl import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, Qwen3VLSceneReasoner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-dir",
        required=True,
        help="A results/runs/sensor_capture/<scene>/ dir containing rgb.png",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        default="Describe this scene for a robot deciding where it is safe to move next.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--out", default=None, help="Output dir, default: <capture-dir>/vlm_description")
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    rgb = np.array(Image.open(capture_dir / "rgb.png").convert("RGB"))

    out_dir = Path(args.out) if args.out else capture_dir / "vlm_description"
    out_dir.mkdir(parents=True, exist_ok=True)

    reasoner = Qwen3VLSceneReasoner(model_name=args.model)
    description = reasoner.describe(
        rgb,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    result = {
        "model": args.model,
        "capture_dir": str(capture_dir),
        "prompt": args.prompt,
        "description": description,
    }
    (out_dir / "description.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
