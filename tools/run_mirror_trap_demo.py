#!/usr/bin/env python
"""Mirror-trap demo: depth-only baseline vs. FORESIGHT (VLM + Laya), from the same staged start.

A staged encounter puts the agent in front of a mirror with its goal behind it (8 m by the real route).
Both conditions map and plan with the same `MapNavigator` (self-built occupancy map from zero-shot
monocular depth). Depth reads the reflection as open floor, so the depth-only planner routes into it.
The FORESIGHT condition runs Qwen3-VL in the background (~0.5 Hz, reflection-aware prompt) and has Laya
re-decide every control tick from the latest description; P(mirror) > 0.6 on a note makes the robot turn
back for a second look, and a second flag puts a wall across that view on its map, so it plans the real
way round. A control stage (no reflective surface in view) checks that FORESIGHT doesn't just stop everywhere.

Both runs use `foresight.sim.timed_episode` (sim time advances by measured compute), so decision
latency has real consequences.
  --phase run     per-tick episodes for every stage in --stages, plus render assets (photographic
                  top-down cutaway, start-frame depth for the explainer)
  --phase render  composes the narrated square 1920x1920 video (foresight.viz.mirror_trap_video; Kokoro voice-over)
Stage poses were found by results/runs/glass_scout + results/runs/mirror_trap.
"""
import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HM3D = "data/processed/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json"
PLACE = "HM3D scene 00808, a real home scanned with a Matterport camera"
STAGES = {
    "hm3d-hallway-mirror": dict(
        dataset=HM3D, scene="00808-y9hTuugGdiq", start=[-2.592, 0.061, 0.290], yaw_deg=-89.1,
        # Goal: open floor behind the mirror (the start->goal line crosses the mirror 0.45 m off its centre);
        # 8.0 m by the walkable route: out of the bedroom, along the corridor, up the living room. The first
        # pick, 4.5 m straight ahead, sat at the foot of the spiral staircase, which a mapper rightly treats as
        # an obstacle.
        goal=[2.4, 0.061, 1.1], trap_ahead_m=2.52, trap_width_m=1.6, trap="mirror", max_sim_s=90.0, stop_radius_m=0.4,
        place=PLACE, hook="Open hallway… or a mirror?"),
    # No mirror/glass/window pixels anywhere along this 4.15 m drive (checked every 0.5 m with semantics).
    "hm3d-bedroom-control": dict(
        dataset=HM3D, scene="00808-y9hTuugGdiq", start=[-9.679, 0.061, 3.462], yaw_deg=-4.9,
        goal_ahead_m=3.7, trap_ahead_m=None, trap_width_m=0.0, trap="mirror", max_sim_s=14.0, stop_radius_m=0.35,
        place=PLACE, hook=None),
}


def stage(name):
    st = dict(STAGES[name], name=name)
    yaw = math.radians(st["yaw_deg"])
    fwd = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    st["yaw"], st["fwd"] = yaw, fwd
    st["start"] = np.asarray(st["start"], float)
    st["goal"] = np.asarray(st["goal"], float) if "goal" in st else st["start"] + fwd * st["goal_ahead_m"]
    st["trap_point"] = None if st["trap_ahead_m"] is None else st["start"] + fwd * st["trap_ahead_m"]
    return st


def run(args, out: Path, st):
    from foresight.perception.depth.depth_anything import ZeroShotDepthEstimator
    from foresight.perception.vlm.qwen_vl import REFLECTION_AWARE_PROMPT, Qwen3VLSceneReasoner
    from foresight.planning.laya_client import LayaClient
    from foresight.planning.laya_fusion import BackgroundDescriber, ForesightNavigator
    from foresight.planning.map_navigator import MapNavigator
    from foresight.sim.timed_episode import run_timed_episode

    depth = ZeroShotDepthEstimator()
    common = dict(dataset_config=st["dataset"], scene_id=st["scene"], depth_estimator=depth,
                  start_position=st["start"], start_yaw_rad=st["yaw"], goal_position=st["goal"],
                  max_sim_s=st["max_sim_s"], width=args.render_size, height=args.render_size,
                  model_size=(args.model_size, args.model_size), stop_radius_m=st["stop_radius_m"],
                  stop_on_contact_m=args.contact_m, keep_frames=False)
    goal_xz, map_center = st["goal"][[0, 2]], (st["start"][[0, 2]] + st["goal"][[0, 2]]) / 2

    def navigator():  # identical mapping/planning/control for both conditions
        return MapNavigator(goal_xz, map_center, max_linear_mps=args.max_linear_mps)

    if "baseline" in args.conditions:
        nav = navigator()
        ep = run_timed_episode(decide=lambda rgb, d, bearing, t, pose: nav.step(d, pose, t), **common)
        _save(out / "baseline.pkl", ep)
        _summary("depth-only", ep)

    if "laya" in args.conditions:
        vlm = Qwen3VLSceneReasoner(model_name="models/Qwen3-VL-2B-Instruct")
        vlm.describe(np.zeros((args.model_size, args.model_size, 3), np.uint8), max_new_tokens=4)  # warmup
        with LayaClient(model=args.laya_model) as laya:
            laya.predict("warmup", {"q": {"type": "noul", "instructions": "warmup"}})
            describer = BackgroundDescriber(
                lambda rgb: vlm.describe(rgb, prompt=REFLECTION_AWARE_PROMPT, max_new_tokens=args.vlm_tokens))
            ctl = ForesightNavigator(laya, describer, navigator(), threshold=args.threshold)
            ep = run_timed_episode(decide=ctl.step, **common)
        _save(out / "laya.pkl", ep)
        _summary("FORESIGHT (VLM + Laya)", ep)
        lat = describer.latencies_s
        print(f"  VLM description latency: median {np.median(lat):.2f} s over {len(lat)} calls")
        (out / "laya_metrics.json").write_text(json.dumps({
            "vlm_describe_s": lat,
            "vlm_jobs": describer.jobs,
            "laya_model": args.laya_model,
            "laya_infer_ms_median": float(np.median([x["laya_infer_ms"] for x in ep.steps])),
            "laya_roundtrip_ms_median": float(np.median([x["laya_roundtrip_ms"] for x in ep.steps])),
            "tick_compute_ms_median": float(np.median([x["tick_compute_ms"] for x in ep.steps])),
        }, indent=1))
        del vlm

    if "assets" in args.conditions:
        _assets(args, out, st, depth)


def _assets(args, out, st, depth_estimator):
    """Photographic top-down cutaway of the stage (orthographic camera just under the ceiling) and,
    for trap stages, the start frame's RGB / predicted depth / sim GT depth for the explainer."""
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
    from PIL import Image
    from foresight.sim.sensors import make_sim_config

    pts = [st["start"], st["goal"]] + ([st["trap_point"]] if st["trap_point"] is not None else [])
    for cond in ("baseline", "laya"):  # frame the routes both robots actually drove
        if (out / f"{cond}.pkl").exists():
            with open(out / f"{cond}.pkl", "rb") as f:
                pts += list(pickle.load(f).positions)
    pts = np.array(pts)
    lo, hi = pts[:, [0, 2]].min(0), pts[:, [0, 2]].max(0)
    width_m = float(max(5.0, (hi - lo).max() + 1.6))
    center = np.array([(lo[0] + hi[0]) / 2, st["start"][1], (lo[1] + hi[1]) / 2])
    R = args.map_res
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file, cfg.scene_id = st["dataset"], st["scene"]
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid, spec.sensor_type, spec.resolution = "rgb", habitat_sim.SensorType.COLOR, [R, R]
    spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    # Looking straight down from 2.1 m: geometry above the camera (ceiling) is culled -> a cutaway plan.
    # Calibrated (phase correlation of 0.5 m camera shifts): view width = 1 / ortho_scale metres,
    # image right = +x, image down = +z.
    spec.position, spec.orientation, spec.ortho_scale = [0.0, 2.1, 0.0], [-math.pi / 2, 0.0, 0.0], 1.0 / width_m
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    sim.get_agent(0).set_state(habitat_sim.AgentState(position=center))
    Image.fromarray(sim.get_sensor_observations()["rgb"][..., :3]).save(out / "topdown.png")
    sim.close()
    meta = {"center": center.tolist(), "width_m": width_m, "res": R}

    if st["trap_point"] is not None:
        sim = habitat_sim.Simulator(make_sim_config(st["dataset"], st["scene"], args.render_size, args.render_size))
        sim.get_agent(0).set_state(habitat_sim.AgentState(
            position=st["start"], rotation=quat_from_angle_axis(st["yaw"], np.array([0, 1.0, 0]))))
        obs = sim.get_sensor_observations()
        sim.close()
        rgb = obs["rgb"][..., :3]
        small = np.asarray(Image.fromarray(rgb).resize((args.model_size, args.model_size), Image.BILINEAR))
        pred = np.asarray(Image.fromarray(depth_estimator.estimate(small)).resize(rgb.shape[1::-1], Image.BILINEAR))
        np.savez_compressed(out / "explainer.npz", rgb=rgb, gt=obs["depth"], pred=pred)
    (out / "assets.json").write_text(json.dumps(meta, indent=1))
    print(f"  assets: top-down {width_m:.1f} m wide -> {out / 'topdown.png'}")


def _save(path: Path, ep):
    with open(path, "wb") as f:
        pickle.dump(ep, f)


def _summary(name, ep):
    s = ep.steps
    travelled = float(np.linalg.norm(ep.positions[-1] - ep.positions[0]))
    print(f"{name}: {len(s)} ticks over {ep.times[-1] + s[-1]['dt']:.1f} s sim | navmesh collisions {int(ep.collided.sum())} | "
          f"moved {travelled:.2f} m | min true distance ahead {min(x['gt_depth_ahead_m'] for x in s):.2f} m | "
          f"reached goal at {ep.reached_goal_at} | tick compute median {np.median([x['tick_compute_ms'] for x in s]):.0f} ms")
    print(f"  contact (crash) at {ep.contact_at} | depth scale median {np.median([x['depth_scale'] for x in s]):.3f}")
    if "p_hazard" in s[0]:
        walls = [x["t"] for x in s if x["wall_added"]]
        print(f"  reflection walls added at t={[round(w, 1) for w in walls]}")
        for x in s:
            if x["event"]:
                print(f"  t={x['t']:.1f} {x['event']} (note seen at {x['description_seen_at']:.1f} s, P={x['p_hazard']:.2f})")
        print(f"  laya infer median {np.median([x['laya_infer_ms'] for x in s]):.1f} ms, "
              f"roundtrip {np.median([x['laya_roundtrip_ms'] for x in s]):.1f} ms")
        seen = []
        for x in s:
            if x["description"] and x["description"] not in seen:
                seen.append(x["description"])
                print(f"  [t={x['t']:.1f}s, seen at {x['description_seen_at']:.1f}s] P(mirror)={x['p_hazard']:.2f} :: "
                      f"{x['description'][:160]!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["run", "render", "all"], default="all")
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    parser.add_argument("--conditions", nargs="+", default=["baseline", "laya", "assets"])
    parser.add_argument("--out", default="results/runs/mirror_trap_demo")
    parser.add_argument("--max-linear-mps", type=float, default=0.35)
    parser.add_argument("--render-size", type=int, default=1024, help="Square sensor resolution for the video")
    parser.add_argument("--model-size", type=int, default=640, help="Square resolution the models see")
    parser.add_argument("--map-res", type=int, default=1000, help="Top-down cutaway resolution")
    parser.add_argument("--laya-model", default="models/laya",
                        help="English root: on real Qwen descriptions the multilingual checkpoint flagged mirrors in mirror-free text")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--contact-m", type=float, default=0.3,
                        help="End an episode as a crash when sim-GT depth straight ahead drops below this (both agents)")
    parser.add_argument("--vlm-tokens", type=int, default=96)
    parser.add_argument("--vlm-direct-ms", type=float, default=None,
                        help="Qwen3-VL answering the decision questions itself; default: median from --bench")
    parser.add_argument("--bench", default="results/runs/mirror_trap/bench_speed_640sq.json",
                        help="Qwen3-VL direct-decision timings on 640x640 frames from this run (bench_speed.json was 640x480)")
    parser.add_argument("--video", default=None, help="default: <out>/foresight_mirror_trap.mp4")
    parser.add_argument("--tts-python", default="/ml/HelaBERT_Analysis/.venv/bin/python",
                        help="An interpreter with kokoro installed (the voice-over runs out of process)")
    parser.add_argument("--kokoro-home", default="/ml/Models/kokoro/hf", help="HF_HOME holding hexgrad/Kokoro-82M")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice")
    args = parser.parse_args()
    root = Path(args.out)
    if args.phase in ("run", "all"):
        for name in args.stages:
            out = root / name
            out.mkdir(parents=True, exist_ok=True)
            print(f"== {name}")
            run(args, out, stage(name))
    if args.phase in ("render", "all"):
        from foresight.viz.mirror_trap_video import render_video
        bench = args.vlm_direct_ms
        if bench is None:
            bench = float(np.median([r["vlm_direct_ms"] for r in json.loads(Path(args.bench).read_text())]))
        render_video(root, {n: stage(n) for n in STAGES}, bench, Path(args.video or root / "foresight_mirror_trap.mp4"),
                     args.tts_python, args.kokoro_home, args.voice)


if __name__ == "__main__":
    main()
