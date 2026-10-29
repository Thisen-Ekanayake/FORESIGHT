#!/usr/bin/env python
"""Laya decision worker: a long-lived process answering typed questions over stdin/stdout JSON lines.

Laya needs Python >=3.10, but habitat-sim 0.3.3 is pinned to 3.9, so the navigation stack can't
import it directly. Run this with the `.venv-laya` interpreter (Python 3.11); the 3.9 side talks to
it through `foresight.planning.laya_client.LayaClient`.

Protocol, one JSON object per line:
  in:  {"state": <str|dict>, "questions": {...}}
  out: {"answers": {...}, "infer_ms": float}       or {"error": str}
The first line written is {"ready": true, "load_s": float, "device": str} once the model is loaded.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/laya", help="Local checkpoint dir (English root, or a subfolder)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32-weights", action="store_true",
                        help="Keep fp32 weights on CUDA. By default they're cast to bf16 (the checkpoints already "
                             "compute under bf16 autocast): halves VRAM, so Laya fits beside Qwen3-VL on an 8 GB GPU")
    args = parser.parse_args()

    import laya
    import torch

    t0 = time.perf_counter()
    agent = laya.load(args.model, device=args.device)
    if args.device.startswith("cuda") and not args.fp32_weights:
        agent.model.to(torch.bfloat16)
        torch.cuda.empty_cache()
    print(json.dumps({"ready": True, "load_s": time.perf_counter() - t0, "device": args.device}), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.perf_counter()
            result = agent.predict(req["state"], req["questions"])
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t) * 1000
            print(json.dumps({"answers": result["answers"], "infer_ms": infer_ms}, default=float), flush=True)
        except Exception as e:  # keep serving; the client decides what a failed decision means
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)


if __name__ == "__main__":
    main()
