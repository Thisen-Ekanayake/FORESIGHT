#!/usr/bin/env python
"""Object search ("FORESIGHT, find the sofa. Then go back to the bed.") — baseline vs. FORESIGHT.

No map and no goal coordinates are given: each robot explores frontiers of its own occupancy map (zero-shot
monocular depth) while a background Qwen3-VL worker grounds the instruction's object names in the camera
frame; sightings are fused on the map and an object becomes a goal after two looks agree
(`foresight.planning.object_search`). Conditions share everything except reflection handling:
  baseline   explorer + grounding + object memory
  foresight  the same, plus Qwen3-VL scene notes read by Laya every tick; a reflection flagged twice becomes
             a wall on the map (`laya_fusion.ForesightNavigator`), and objects seen through it are ignored.
Every look runs on one worker (FORESIGHT's looks = description + grounding, so they are slower), and sim
time advances by measured compute (`foresight.sim.timed_episode`).

  --phase run     episodes x conditions -> <out>/<episode>/<condition>.pkl
  --phase eval    ground-truth scoring (`foresight.sim.objectnav_eval`) -> <out>/eval.json + a table
  --oracle        test harness: grounding from the simulator's semantic labels instead of Qwen3-VL (no VLM,
                  fast) — for developing the explorer only, never for reported numbers
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # Qwen + depth + Laya + renderer fill the 8 GB GPU
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HM3D = "data/processed/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json"
EXAMPLE = "data/processed/scene_datasets/hm3d/example/hm3d_example_glb.scene_dataset_config.json"

# The demo episode starts at the mirror-trap pose (facing 00808's hallway mirror, the living room with the
# sofa behind it); the rest are the evaluation set, fixed before any were run — see EVAL_SET.
EPISODES = {
    "mirror-sofa-bed": dict(dataset=HM3D, scene="00808-y9hTuugGdiq", start=[-2.592, 0.061, 0.290], yaw_deg=-89.1,
                            instructions=["sofa", "bed"], max_sim_s=300.0,
                            # the hallway mirror (closet mirror wall 527: obb centre x -0.07, z -1.13..1.33), for drawing
                            mirror_xz=[[-0.07, -1.13], [-0.07, 1.33]],
                            place="HM3D 00808: a real home (Matterport scan) in Habitat-Sim. No map, no goal coordinates."),
}
EVAL_SET = Path("experiments/configs/object_search/episodes.json")
if EVAL_SET.exists():
    for _e in json.loads(EVAL_SET.read_text())["episodes"]:
        EPISODES[_e["name"]] = dict({k: v for k, v in _e.items() if k != "name"}, dataset=HM3D)


def episode(name):
    e = dict(EPISODES[name], name=name)
    e["yaw"] = math.radians(e["yaw_deg"])
    e["start"] = np.asarray(e["start"], float)
    return e


class OracleLocate:
    """Boxes from the simulator's instance labels (harness only): one box per visible instance >= 0.4% of the frame."""

    def __init__(self):
        self.semantic, self.id2cat = None, None

    def update(self, pose):
        from foresight.sim.objectnav_eval import CATEGORY_NAMES
        if self.id2cat is None:
            sim = pose["oracle_sim"]
            self.id2cat = {}
            for o in sim.semantic_scene.objects:
                if o is not None and o.category is not None:
                    n = o.category.name().lower().strip()
                    for lab, names in CATEGORY_NAMES.items():
                        if n in names:
                            self.id2cat[o.semantic_id] = lab
        self.semantic = pose.pop("oracle_semantic")
        pose.pop("oracle_sim", None)

    def describe(self, rgb):
        """Scripted scene note (harness only): mentions a mirror iff a labelled mirror fills the view's centre."""
        sem = self.semantic
        h, w = sem.shape
        centre = sem[int(.3 * h):int(.7 * h), int(.3 * w):int(.7 * w)]
        mirror = np.isin(centre, [i for i, c in self.id2cat.items() if c == "mirror"]).mean()
        if mirror > 0.25:
            return "The open hallway ahead is actually a reflection in a large mirror on the wall."
        return "An ordinary room; walls, floor and furniture, no mirror or glass in view."

    def __call__(self, rgb, label):
        sem = self.semantic
        h, w = sem.shape
        boxes = []
        for i in np.unique(sem):
            if self.id2cat.get(int(i)) != label:
                continue
            ys, xs = np.nonzero(sem == i)
            if ys.size < 0.004 * h * w:
                continue
            boxes.append([int(xs.min() / w * 1000), int(ys.min() / h * 1000), int(xs.max() / w * 1000), int(ys.max() / h * 1000)])
        return boxes


_MODELS = {}  # loaded once per process: two Qwen3-VL copies don't fit the 8 GB GPU


def _model(name):
    if name not in _MODELS:
        if name == "depth":
            from foresight.perception.depth.depth_anything import ZeroShotDepthEstimator
            _MODELS[name] = ZeroShotDepthEstimator()
        else:
            from foresight.perception.vlm.qwen_vl import Qwen3VLSceneReasoner
            vlm = Qwen3VLSceneReasoner(model_name="models/Qwen3-VL-2B-Instruct")
            vlm.describe(np.zeros((64, 64, 3), np.uint8), max_new_tokens=4)  # warmup
            _MODELS[name] = vlm
    return _MODELS[name]


def run(args, e, cond, out: Path):
    from foresight.planning.object_search import ObjectSearchNavigator, PerceptionWorker
    from foresight.sim.timed_episode import run_timed_episode

    depth = _model("depth")
    holder = []
    labels = lambda: holder[0].look_labels()  # every named object, current target first (later legs use memory)
    oracle = OracleLocate() if args.oracle else None
    vlm = None
    if not args.oracle:
        vlm = _model("vlm")
    locate = oracle if oracle is not None else (lambda rgb, lab: vlm.locate(rgb, lab))
    direction = present = None
    if vlm is not None:
        from foresight.perception.vlm.qwen_vl import PRESENT_PROMPT, GROUNDING_NAMES
        present = lambda rgb, lab: vlm.option_probs(rgb, PRESENT_PROMPT.format(name=GROUNDING_NAMES.get(lab, lab)), ["yes", "no"])[0]
    if vlm is not None and not args.no_prior:
        from foresight.perception.vlm.qwen_vl import DIRECTION_PROMPT, GROUNDING_NAMES
        direction = lambda rgb, lab: vlm.option_probs(
            rgb, DIRECTION_PROMPT.format(name=GROUNDING_NAMES.get(lab, lab)), ["left", "center", "right"])
    common = dict(dataset_config=e["dataset"], scene_id=e["scene"], depth_estimator=depth,
                  start_position=e["start"], start_yaw_rad=e["yaw"], goal_position=e["start"],
                  max_sim_s=args.max_sim_s or e["max_sim_s"], width=args.render_size, height=args.render_size,
                  model_size=(args.model_size, args.model_size), stop_on_contact_m=args.contact_m,
                  keep_frames=False, oracle_semantic=oracle is not None)
    nav_kw = dict(instructions=e["instructions"], start_xz=e["start"][[0, 2]], max_linear_mps=args.max_linear_mps)
    t0 = time.time()
    laya = None
    if cond == "baseline":
        worker = PerceptionWorker(locate, labels, min_latency_s=args.oracle_latency_s if oracle else 0.0,
                                  direction=direction, present=present)
        nav = ObjectSearchNavigator(worker=worker, **nav_kw)
        holder.append(nav)

        def decide(rgb, d, bearing, t, pose):
            if oracle is not None:
                oracle.update(pose)
            return nav.step(d, pose, t, rgb=rgb)
    else:
        from foresight.perception.vlm.qwen_vl import REFLECTION_AWARE_PROMPT
        from foresight.planning.laya_client import LayaClient
        from foresight.planning.laya_fusion import ForesightNavigator
        laya = LayaClient(model=args.laya_model)
        laya.predict("warmup", {"q": {"type": "noul", "instructions": "warmup"}})
        if oracle is not None:
            worker = PerceptionWorker(locate, labels, describe=oracle.describe, min_latency_s=2.1 + args.oracle_latency_s)
        else:
            worker = PerceptionWorker(locate, labels, describe=lambda rgb: vlm.describe(
                rgb, prompt=REFLECTION_AWARE_PROMPT, max_new_tokens=args.vlm_tokens), direction=direction, present=present,
                extra_labels=["mirror"])
        nav = ObjectSearchNavigator(worker=worker, poll_worker=False, **nav_kw)
        holder.append(nav)
        ctl = ForesightNavigator(laya, worker, nav, threshold=args.threshold, verify_timeout_s=12.0, in_path_only=True)

        def decide(rgb, d, bearing, t, pose):
            if oracle is not None:
                oracle.update(pose)
            return ctl.step(rgb, d, bearing, t, pose)
    try:
        ep = run_timed_episode(decide=decide, **common)
    finally:
        if laya is not None:
            laya.close()
    wall = time.time() - t0
    _thin(ep)
    with open(out / f"{cond}.pkl", "wb") as f:
        pickle.dump(ep, f)
    meta = {"condition": cond, "episode": e["name"], "instructions": e["instructions"], "oracle": bool(oracle),
            "wall_s": wall, "look_latency_s": worker.latencies_s, "look_jobs": worker.jobs,
            "tick_compute_ms_median": float(np.median([x["tick_compute_ms"] for x in ep.steps])),
            "laya_infer_ms_median": (float(np.median([x["laya_infer_ms"] for x in ep.steps])) if cond == "foresight" else None)}
    (out / f"{cond}_meta.json").write_text(json.dumps(meta, indent=1))
    _summary(cond, ep, meta)


def assets(args, e, out: Path):
    """Photographic top-down cutaway framing both runs (orthographic camera 2.1 m up, ceiling culled; view width =
    1/ortho_scale m, image right = +x, down = +z, as calibrated for the mirror-trap demo) and the start frame's
    RGB / predicted depth / sim GT depth for the explainer."""
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
    from PIL import Image
    from foresight.sim.sensors import make_sim_config

    pts = [e["start"]]
    for cond in ("baseline", "foresight"):
        if (out / f"{cond}.pkl").exists():
            with open(out / f"{cond}.pkl", "rb") as f:
                ep = pickle.load(f)
            pts += list(ep.positions)
            pts += [np.array([o["xz"][0], 0, o["xz"][1]]) for o in ep.steps[-1].get("memory", [])
                    if o["hits"] >= 3 and not o["rejected"]]
    pts = np.array(pts, float)
    lo, hi = pts[:, [0, 2]].min(0), pts[:, [0, 2]].max(0)
    from foresight.viz.object_search_video import MAP_H, MAP_W
    # the video shows the square cutaway's middle MAP_H of MAP_W rows: frame the z extent for that crop
    width_m = float(max(6.0, hi[0] - lo[0] + 2.0, (hi[1] - lo[1] + 2.0) * MAP_W / MAP_H))
    center = np.array([(lo[0] + hi[0]) / 2, e["start"][1], (lo[1] + hi[1]) / 2])
    R = args.map_res
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file, cfg.scene_id = e["dataset"], e["scene"]
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid, spec.sensor_type, spec.resolution = "rgb", habitat_sim.SensorType.COLOR, [R, R]
    spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    spec.position, spec.orientation, spec.ortho_scale = [0.0, 2.1, 0.0], [-math.pi / 2, 0.0, 0.0], 1.0 / width_m
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    sim.get_agent(0).set_state(habitat_sim.AgentState(position=center))
    Image.fromarray(sim.get_sensor_observations()["rgb"][..., :3]).save(out / "topdown.png")
    sim.close()
    sim = habitat_sim.Simulator(make_sim_config(e["dataset"], e["scene"], 1024, 1024))
    sim.get_agent(0).set_state(habitat_sim.AgentState(
        position=e["start"], rotation=quat_from_angle_axis(e["yaw"], np.array([0, 1.0, 0]))))
    obs = sim.get_sensor_observations()
    sim.close()
    rgb = obs["rgb"][..., :3]
    small = np.asarray(Image.fromarray(rgb).resize((args.model_size, args.model_size), Image.BILINEAR))
    pred = np.asarray(Image.fromarray(_model("depth").estimate(small)).resize(rgb.shape[1::-1], Image.BILINEAR))
    np.savez_compressed(out / "explainer.npz", rgb=rgb, gt=obs["depth"], pred=pred)
    (out / "assets.json").write_text(json.dumps({"center": center.tolist(), "width_m": width_m, "res": R}, indent=1))
    print(f"  assets: top-down {width_m:.1f} m wide -> {out / 'topdown.png'}")


def _thin(ep, every_s: float = 0.5):
    """Keep the per-tick map only every `every_s` of sim time (a 280x280 map per tick is ~150 MB a run)."""
    last = -np.inf
    for x in ep.steps:
        if x["t"] - last >= every_s:
            last = x["t"]
        else:
            x.pop("map_score", None)
        for k in ("memory", "found", "search_events"):  # cumulative: keep only on map ticks and the last tick
            if "map_score" not in x and x is not ep.steps[-1]:
                x.pop(k, None)


def _summary(cond, ep, meta):
    s = ep.steps
    last = s[-1]
    print(f"{cond}: {len(s)} ticks, {ep.times[-1] + last['dt']:.1f} s sim, wall {meta['wall_s']:.0f} s | end mode {last['mode']} | "
          f"crash at {ep.contact_at} | navmesh collisions {int(ep.collided.sum())} | "
          f"looks median {np.median(meta['look_latency_s']):.2f} s | tick median {meta['tick_compute_ms_median']:.0f} ms")
    for ev in last["search_events"]:
        print(f"   t={ev[0]:6.1f} {ev[1]} {ev[2]}")
    for o in last["memory"]:
        print(f"   memory {o['label']:12s} at ({o['xz'][0]:6.2f},{o['xz'][1]:6.2f}) hits {o['hits']} (raw {o['raw_hits']}) misses {o['misses']} rejected {o['rejected']}")
    for x in s:
        if x.get("event"):
            print(f"   mirror check t={x['t']:.1f} {x['event']} P={x['p_hazard']:.2f}")


def evaluate(args, names):
    from foresight.sim.objectnav_eval import score_episode
    rows = []
    for name in names:
        e = episode(name)
        for cond in args.conditions:
            pkl = Path(args.out) / name / f"{cond}.pkl"
            if not pkl.exists():
                continue
            with open(pkl, "rb") as f:
                ep = pickle.load(f)
            r = score_episode(e["dataset"], e["scene"], ep, e["instructions"])
            r.update(episode=name, condition=cond)
            rows.append(r)
            legs = " | ".join(f"{l['label']}: {'OK' if l['success'] else ('declared, wrong' if l['declared'] else 'not found')}"
                              f" spl {l['spl']:.2f}" + (f" d={l['dist_to_nearest_m']:.2f}m" if l['declared'] else "") for l in r["legs"])
            print(f"{name:28s} {cond:9s} crash={r['crashed']!s:5s} {legs}")
    (Path(args.out) / "eval.json").write_text(json.dumps(rows, indent=1, default=float))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["run", "eval", "assets", "render"], default="run")
    p.add_argument("--episodes", nargs="+", default=["mirror-sofa-bed"])
    p.add_argument("--conditions", nargs="+", default=["baseline", "foresight"], choices=["baseline", "foresight"])
    p.add_argument("--out", default="results/runs/object_search/demo")
    p.add_argument("--oracle", action="store_true", help="harness: semantic-label grounding, no VLM for the baseline")
    p.add_argument("--oracle-latency-s", type=float, default=0.6, help="harness: simulated look latency")
    p.add_argument("--max-linear-mps", type=float, default=0.35)
    p.add_argument("--render-size", type=int, default=640, help="square sensor resolution (video re-renders views)")
    p.add_argument("--model-size", type=int, default=640, help="square resolution the models see")
    p.add_argument("--laya-model", default="models/laya")
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--contact-m", type=float, default=0.3)
    p.add_argument("--vlm-tokens", type=int, default=96)
    p.add_argument("--eval", default="results/runs/object_search/eval/eval.json", help="render: scoreboard source")
    p.add_argument("--video", default=None)
    p.add_argument("--tts-python", default="/ml/HelaBERT_Analysis/.venv/bin/python")
    p.add_argument("--kokoro-home", default="/ml/Models/kokoro/hf")
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--map-res", type=int, default=1000)
    p.add_argument("--max-sim-s", type=float, default=None, help="override the episodes' time budget (quick tests)")
    p.add_argument("--no-prior", action="store_true", help="explore nearest/biggest frontier only (no VLM direction hint)")
    args = p.parse_args()
    names = args.episodes
    if names == ["eval"]:
        names = [e["name"] for e in json.loads(EVAL_SET.read_text())["episodes"]] + ["mirror-sofa-bed"]
    if args.phase == "assets":
        for name in names:
            assets(args, episode(name), Path(args.out) / name)
    elif args.phase == "render":
        from foresight.viz.object_search_video import render_video
        e = episode(names[0])
        rows = json.loads(Path(args.eval).read_text()) if args.eval and Path(args.eval).exists() else []
        n_scenes = len({EPISODES[r["episode"]]["scene"] for r in rows if r["episode"] in EPISODES})
        render_video(Path(args.out), e, rows, n_scenes, Path(args.video or Path(args.out) / "foresight_object_search.mp4"),
                     args.tts_python, args.kokoro_home, args.voice)
    elif args.phase == "run":
        for name in names:
            out = Path(args.out) / name
            out.mkdir(parents=True, exist_ok=True)
            for cond in args.conditions:
                print(f"== {name} / {cond}", flush=True)
                run(args, episode(name), cond, out)
    else:
        evaluate(args, names)


if __name__ == "__main__":
    main()
