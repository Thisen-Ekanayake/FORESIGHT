#!/usr/bin/env python
"""FORESIGHT's brain for the live Isaac Sim runs: the exact decision stack of the point-goal evaluation
(tools/run_pointnav_eval.py `build`: Depth Anything V2 → FloorScale → map navigator, plus for FORESIGHT the
Qwen3-VL-2B box-fill guard), served over a local socket to the Isaac Lab world (tools/isaac/g1_live.py).

Timing is the evaluation's: each tick's compute (depth + decision, minus background VLM work) is measured and
returned, and the world advances by max(0.1 s, that) on the previous command; VLM looks are delivered their measured
latency later in sim time by the PerceptionWorker. The world is paused while the models run (one 8 GB GPU).

  python tools/isaac/foresight_brain.py            # habitat env; serves until killed
Each episode's log (per-tick decisions, plan, boxes, small depth maps) is written to the path the world gives.
"""
import argparse
import os
import pickle
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import wire  # noqa: E402

KEEP = ("depth_scale", "depth_ahead_m", "safety_level", "min_clearance_m", "looking_around", "escaping",
        "background_ms", "event", "hazard", "vlm_progress", "plan_path", "plan_cost")


def jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    return x


class Brain:
    def __init__(self):
        from foresight.perception.depth.depth_anything import ZeroShotDepthEstimator
        self.est = ZeroShotDepthEstimator()
        self.decide = None

    def reset(self, h):
        import run_pointnav_eval as rpe
        args = SimpleNamespace(max_linear_mps=h.get("max_linear_mps", 0.35), no_floor_scale=False,
                               robust_scan=h.get("robust_scan", True), guard=h.get("guard", "fill"),
                               fill_hold_s=h.get("fill_hold_s", 3.0), threshold=0.6, vlm_tokens=60,
                               hazard_labels=h.get("hazard_labels", ["mirror"]))
        start, goal = np.asarray(h["start_xz"], float), np.asarray(h["goal_xz"], float)
        self.decide, self.ctl = rpe.build(h["cond"], {}, args, start, goal,
                                          nav_extra={"cam_height_m": float(h["cam_height_m"])})
        self.log = {"header": h, "ticks": []}
        self.log_path = h.get("log_path")
        print(f"reset: {h['cond']} start {start.round(2)} goal {goal.round(2)} cam {h['cam_height_m']:.2f} m "
              f"labels {args.hazard_labels}", flush=True)

    def tick(self, h, payload):
        rgb = np.frombuffer(payload, np.uint8).reshape(h["h"], h["w"], 3)
        pose = {"position": np.asarray(h["position"], float), "forward_xz": list(h["forward_xz"])}
        t0 = time.perf_counter()
        depth = self.est.estimate(rgb)
        depth_ms = (time.perf_counter() - t0) * 1000
        lin, ang, d = self.decide(rgb, depth, float(h["bearing"]), float(h["t"]), pose)
        wall_ms = (time.perf_counter() - t0) * 1000
        tick_ms = wall_ms - float(d.get("background_ms", 0.0) or 0.0)
        rec = {k: jsonable(d.get(k)) for k in KEEP if k in d}
        nav = getattr(self.ctl, "nav", self.ctl)
        worker = getattr(nav, "worker", None)
        looks = {}
        if worker is not None:
            for k, r in worker.results.items():
                if abs(k - float(h["t"])) < 1e-6 or k not in self._seen_looks:
                    looks[k] = {"boxes": jsonable(r.get("boxes", {})), "present": jsonable(r.get("present", {}))}
            self._seen_looks.update(looks)
        small = depth[::4, ::4].astype(np.float16)
        self.log["ticks"].append({"t": h["t"], "position": h["position"], "forward_xz": h["forward_xz"],
                                  "linear": float(lin), "angular": float(ang), "tick_ms": tick_ms,
                                  "depth_ms": depth_ms, "diag": rec, "looks": looks, "depth_small": small,
                                  "walls": jsonable(getattr(nav.map, "walls", []))})
        return {"linear": float(lin), "angular": float(ang), "tick_ms": tick_ms, "depth_ms": depth_ms,
                "safety": rec.get("safety_level"), "event": rec.get("event"), "looks": jsonable(looks)}

    def end(self, h):
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            self.log["end"] = h
            with open(self.log_path, "wb") as f:
                pickle.dump(self.log, f)
            print("wrote", self.log_path, len(self.log["ticks"]), "ticks", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=wire.PORT)
    a = ap.parse_args()
    brain = Brain()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", a.port))
    srv.listen(1)
    print(f"brain ready on 127.0.0.1:{a.port}", flush=True)
    while True:
        conn, _ = srv.accept()
        brain._seen_looks = {}
        try:
            while True:
                h, payload = wire.recv(conn)
                kind = h.get("type")
                if kind == "reset":
                    brain._seen_looks = {}
                    brain.reset(h)
                    wire.send(conn, {"ok": True})
                elif kind == "tick":
                    wire.send(conn, brain.tick(h, payload))
                elif kind == "end":
                    brain.end(h)
                    wire.send(conn, {"ok": True})
                elif kind == "quit":
                    wire.send(conn, {"ok": True})
                    return
        except ConnectionError:
            print("world disconnected", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
