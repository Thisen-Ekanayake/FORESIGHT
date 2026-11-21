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
        "background_ms", "event", "hazard", "vlm_progress", "plan_path", "plan_cost", "p_hazard", "verifying",
        "wall_added", "description", "description_seen_at", "laya_called",
        # object search (task "search")
        "mode", "target_label", "goal_xz", "frontier_xz", "memory", "found", "search_events", "episode_done")


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
        args = SimpleNamespace(max_linear_mps=h.get("max_linear_mps", 0.35), no_floor_scale=bool(h.get("no_floor_scale")),
                               robust_scan=h.get("robust_scan", True), guard=h.get("guard", "fill"),
                               fill_hold_s=h.get("fill_hold_s", 3.0), threshold=0.6, vlm_tokens=60,
                               hazard_labels=h.get("hazard_labels", ["mirror"]))
        start, goal = np.asarray(h["start_xz"], float), np.asarray(h["goal_xz"], float)
        # the depth the navigator is given: "da" (Depth Anything V2, the estimator) or a ground-truth source from the
        # point-goal evaluation ("gt", "gt:noise=0.1", ...: foresight.perception.depth.sources), on Habitat's GT
        # depth at the robot's pose; noise is seeded per run, as the evaluation seeds it per episode
        from foresight.perception.depth.sources import make_source
        self.depth_spec = h.get("depth", "da")
        self.source = make_source(self.depth_spec, self.est)
        self.source.reset(None, seed=int(h.get("seed", 0)))
        if self.depth_spec != "da" and not h.get("habitat_scene"):
            raise ValueError("ground-truth depth sources need habitat perception (habitat_scene)")
        if h.get("task") == "search":
            self._reset_search(h, start)
        elif h["cond"] == "foresight_v1":
            # FORESIGHT v1, exactly as the 2026-10-29 mirror-trap demo (tools/run_mirror_trap_demo.py): Qwen3-VL describes
            # in the background with the reflection-aware prompt, Laya reads each note, a flagged view is checked again
            # and walled off on the map navigator's own map.
            from foresight.perception.vlm.qwen_vl import REFLECTION_AWARE_PROMPT
            from foresight.planning.laya_client import LayaClient
            from foresight.planning.laya_fusion import BackgroundDescriber, ForesightNavigator
            from foresight.planning.map_navigator import MapNavigator
            vlm = rpe.model("vlm")
            if getattr(self, "_laya", None) is None:
                self._laya = LayaClient(model="models/laya").__enter__()
                self._laya.predict("warmup", {"q": {"type": "noul", "instructions": "warmup"}})
            describer = BackgroundDescriber(
                lambda rgb: vlm.describe(rgb, prompt=REFLECTION_AWARE_PROMPT, max_new_tokens=96))
            # Confirmed mirror walls are impassable here (plan_kw virtual_cost): with the default obstacle cost the
            # planner crossed its own wall into depth's phantom free space beyond the mirror (2026-11-20 G1 run).
            nav = MapNavigator(goal, (start + goal) / 2, max_linear_mps=h.get("max_linear_mps", 0.35),
                               cam_height_m=float(h["cam_height_m"]),
                               plan_kw={"virtual_cost": 1e6} if h.get("hard_walls", True) else None)
            ctl = ForesightNavigator(self._laya, describer, nav, threshold=0.6)
            self.decide, self.ctl = ctl.step, ctl
        else:
            self.decide, self.ctl = rpe.build(h["cond"], {}, args, start, goal,
                                              nav_extra={"cam_height_m": float(h["cam_height_m"])})
        self.log = {"header": h, "ticks": []}
        self._open_habitat(h)
        self.log_path = h.get("log_path")
        print(f"reset: {h.get('task', 'pointnav')} {h['cond']} depth {self.depth_spec}"
              f"{' no-floor-scale' if args.no_floor_scale else ''} start {start.round(2)} goal {goal.round(2)} "
              f"cam {h['cam_height_m']:.2f} m labels {args.hazard_labels}", flush=True)

    def _reset_search(self, h, start):
        """Object search, as tools/run_object_search.py runs it: frontier exploration + Qwen3-VL grounding of the
        instruction's objects; cond foresight_v1 (or foresight) adds the v1 reflection check (scene notes read by
        Laya, a reflection flagged twice becomes a wall, objects seen through it are ignored); cond map = baseline."""
        import run_pointnav_eval as rpe
        from foresight.perception.vlm.qwen_vl import (DIRECTION_PROMPT, GROUNDING_NAMES, PRESENT_PROMPT,
                                                      REFLECTION_AWARE_PROMPT)
        from foresight.planning.object_search import ObjectSearchNavigator, PerceptionWorker
        vlm = rpe.model("vlm")
        holder = []
        labels = lambda: holder[0].look_labels()  # noqa: E731
        present = lambda rgb, lab: vlm.option_probs(rgb, PRESENT_PROMPT.format(name=GROUNDING_NAMES.get(lab, lab)), ["yes", "no"])[0]  # noqa: E731
        direction = lambda rgb, lab: vlm.option_probs(  # noqa: E731
            rgb, DIRECTION_PROMPT.format(name=GROUNDING_NAMES.get(lab, lab)), ["left", "center", "right"])
        locate = lambda rgb, lab: vlm.locate(rgb, lab)  # noqa: E731
        nav_kw = dict(instructions=list(h.get("instructions") or ["sofa", "bed"]), start_xz=start,
                      max_linear_mps=h.get("max_linear_mps", 0.35), cam_height_m=float(h["cam_height_m"]))
        if h["cond"] == "map":
            worker = PerceptionWorker(locate, labels, direction=direction, present=present)
            nav = ObjectSearchNavigator(worker=worker, **nav_kw)
            holder.append(nav)
            self.decide = lambda rgb, d, bearing, t, pose: nav.step(d, pose, t, rgb=rgb)
            self.ctl = nav
            return
        from foresight.planning.laya_client import LayaClient
        from foresight.planning.laya_fusion import ForesightNavigator
        if getattr(self, "_laya", None) is None:
            self._laya = LayaClient(model="models/laya").__enter__()
            self._laya.predict("warmup", {"q": {"type": "noul", "instructions": "warmup"}})
        worker = PerceptionWorker(locate, labels, describe=lambda rgb: vlm.describe(
            rgb, prompt=REFLECTION_AWARE_PROMPT, max_new_tokens=96), direction=direction, present=present,
            extra_labels=["mirror"])
        nav = ObjectSearchNavigator(worker=worker, poll_worker=False, **nav_kw)
        holder.append(nav)
        ctl = ForesightNavigator(self._laya, worker, nav, threshold=0.6, verify_timeout_s=12.0, in_path_only=True)
        self.decide, self.ctl = ctl.step, ctl

    def _open_habitat(self, h):
        self.hsim = None
        if h.get("habitat_scene"):  # perception from Habitat's render of the same scan at the robot's pose
            import habitat_sim
            from make_pointnav_episodes import ANNOTATED, scene_ref
            from foresight.sim.sensors import make_sim_config
            if getattr(self, "_hsim_scene", None) != h["habitat_scene"]:
                if getattr(self, "_hsim", None) is not None:
                    self._hsim.close()
                self._hsim = habitat_sim.Simulator(make_sim_config(str(ANNOTATED), scene_ref(ANNOTATED, h["habitat_scene"]),
                                                                   640, 640, sensor_uuids=("rgb", "depth")))
                self._hsim_scene = h["habitat_scene"]
            self.hsim = self._hsim
            self.floor_y = float(h.get("habitat_floor_y", 0.0))

    def habitat_view(self, h):
        """Render the head camera in Habitat: camera at the given xz, h["cam_height_m"] over the scan's floor, level,
        facing forward_xz (Habitat frame). Habitat's sensor sits 1.5 m above the agent."""
        import habitat_sim
        import quaternion  # noqa: F401
        from habitat_sim.utils.common import quat_from_angle_axis
        fx, fz = h["forward_xz"]
        st = habitat_sim.AgentState()
        st.position = np.array([h["position"][0], self.floor_y + float(h["cam_height_m"]) - 1.5, h["position"][2]], np.float32)
        st.rotation = quat_from_angle_axis(float(np.arctan2(-fx, -fz)), np.array([0.0, 1.0, 0.0]))
        self.hsim.get_agent(0).set_state(st, reset_sensors=True, infer_sensor_states=False)
        obs = self.hsim.get_sensor_observations()
        return np.ascontiguousarray(obs["rgb"][..., :3]), obs["depth"]

    def tick(self, h, payload):
        gt = None
        if self.hsim is not None:
            rgb, gt = self.habitat_view(h)
            h = dict(h, position=[h["position"][0], self.floor_y + float(h["cam_height_m"]), h["position"][2]])
        else:
            rgb = np.frombuffer(payload, np.uint8).reshape(h["h"], h["w"], 3)
        pose = {"position": np.asarray(h["position"], float), "forward_xz": list(h["forward_xz"])}
        t0 = time.perf_counter()
        depth = self.est.estimate(rgb) if self.depth_spec == "da" else self.source(rgb, {"depth": gt})
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
        out = {"linear": float(lin), "angular": float(ang), "tick_ms": tick_ms, "depth_ms": depth_ms,
               "safety": rec.get("safety_level"), "event": rec.get("event"), "looks": jsonable(looks)}
        if "mode" in rec:  # object search: where it is in the task; done = found everything (or gave up)
            out.update(mode=rec["mode"], target=rec.get("target_label"), found=rec.get("found"),
                       done=bool(rec.get("episode_done")))
        if gt is not None:
            gh, gw = gt.shape
            out["gt_ahead_m"] = float(np.median(gt[int(.4 * gh):int(.9 * gh), int(.25 * gw):int(.75 * gw)]))
            self.log["ticks"][-1]["gt_ahead_m"] = out["gt_ahead_m"]
            if len(self.log["ticks"]) % 2 == 1:  # keep every other head frame (small JPEG) for the video's POV inserts
                import io
                from PIL import Image
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, "JPEG", quality=85)
                self.log["ticks"][-1]["rgb_jpg"] = buf.getvalue()
        return out

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
