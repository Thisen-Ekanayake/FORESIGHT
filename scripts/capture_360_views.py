"""Interactively drive a Habitat-Sim agent in a live preview window, then capture a 360-degree ring of
overlapping RGB stills.

Usage:
    python scripts/capture_360_views.py <scene_id>   e.g. 00800-TEEsavR23oF

Renders off-screen (no GPU window server needed beyond EGL/DISPLAY) and shows the live first-person view
in an OpenCV window, so you can see where you're pointed before capturing.

Controls (focus must be on the preview window):
    w / s   move forward / backward
    a / d   turn left / right
    c       capture the 360-degree ring from the current position
    q / ESC quit

The ring uses the camera's 90-degree HFOV (see foresight.sim.sensors.make_sim_config) and a 30% overlap
between consecutive shots, so the turn step per shot is 90 * (1 - 0.3) = 63 degrees. ceil(360 / 63) = 6
shots fully cover the loop (6 * 63 = 378 >= 360). Images are saved directly under the repo root for now
(no output directory plumbing yet).
"""
import sys
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG = str(
    REPO_ROOT
    / "data/processed/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json"
)

sys.path.insert(0, str(REPO_ROOT / "src"))
from foresight.sim.sensors import make_sim_config  # noqa: E402

HFOV_DEG = 90.0
OVERLAP = 0.3
STEP_DEG = HFOV_DEG * (1.0 - OVERLAP)  # 63 degrees
N_SHOTS = int(np.ceil(360.0 / STEP_DEG))  # 6

WINDOW_NAME = "capture_360_views (w/s move, a/d turn, c capture, q quit)"


def _rotate_agent(agent, delta_deg: float) -> None:
    import quaternion

    state = agent.get_state()
    delta_q = quaternion.from_rotation_vector([0.0, np.deg2rad(delta_deg), 0.0])
    state.rotation = state.rotation * delta_q
    agent.set_state(state, reset_sensors=False)


def _render_bgr(sim) -> np.ndarray:
    rgb = sim.get_sensor_observations()["rgb"][..., :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def capture_ring(sim, agent, scene_id: str) -> None:
    print(f"capturing {N_SHOTS} shots at {STEP_DEG:.1f} deg steps ({OVERLAP:.0%} overlap, {HFOV_DEG:.0f} deg HFOV)...")
    for i in range(N_SHOTS):
        heading_deg = i * STEP_DEG
        rgb = sim.get_sensor_observations()["rgb"][..., :3]
        out_path = REPO_ROOT / f"capture_{scene_id}_{i:02d}_{heading_deg:.0f}deg.png"
        Image.fromarray(rgb).save(out_path)
        print(f"  saved {out_path.name}")
        cv2.imshow(WINDOW_NAME, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        _rotate_agent(agent, STEP_DEG)
    print("done.")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <scene_id>", file=sys.stderr)
        sys.exit(1)
    scene_id = sys.argv[1]

    cfg = make_sim_config(DATASET_CONFIG, scene_id, width=640, height=480, sensor_uuids=("rgb",))
    sim = habitat_sim.Simulator(cfg)
    try:
        agent = sim.get_agent(0)
        state = habitat_sim.AgentState()
        state.position = sim.pathfinder.get_random_navigable_point()
        agent.set_state(state)

        print(f"scene: {scene_id}")
        print("controls: w/s move, a/d turn, c capture 360 ring, q quit (click the preview window first)")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        while True:
            cv2.imshow(WINDOW_NAME, _render_bgr(sim))
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):  # 27 = ESC
                break
            elif key == ord("w"):
                agent.act("move_forward")
            elif key == ord("s"):
                agent.act("move_backward")
            elif key == ord("a"):
                agent.act("turn_left")
            elif key == ord("d"):
                agent.act("turn_right")
            elif key == ord("c"):
                capture_ring(sim, agent, scene_id)
            pos = agent.get_state().position
            print(f"  pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    finally:
        cv2.destroyAllWindows()
        sim.close()


if __name__ == "__main__":
    main()
