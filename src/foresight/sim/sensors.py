"""RGB + ground-truth depth + semantic sensor capture from a single Habitat-Sim agent pose."""
from dataclasses import dataclass

import habitat_sim
import numpy as np


@dataclass
class Observation:
    rgb: np.ndarray       # (H, W, 3) uint8
    depth: np.ndarray     # (H, W) float32, meters, simulator ground truth
    semantic: np.ndarray  # (H, W) uint32, per-pixel object instance ID


def make_sim_config(
    dataset_config: str, scene_id: str, width: int, height: int, sensor_uuids=None
) -> habitat_sim.Configuration:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.scene_id = scene_id

    all_sensors = [
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
        ("semantic", habitat_sim.SensorType.SEMANTIC),
    ]
    # Default: all three (sensor-capture use). Pass sensor_uuids=("rgb",) to skip the
    # depth/semantic passes when only RGB is needed (e.g. the demo renderer), which is faster.
    if sensor_uuids is not None:
        all_sensors = [s for s in all_sensors if s[0] in set(sensor_uuids)]

    sensor_specs = []
    for uuid, sensor_type in all_sensors:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [height, width]
        spec.position = [0.0, 1.5, 0.0]
        sensor_specs.append(spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def capture_scene_observation(
    dataset_config: str,
    scene_id: str,
    width: int = 640,
    height: int = 480,
    seed: int = 0,
) -> Observation:
    """Spawn an agent at a random navigable point and capture one RGB+depth+semantic frame."""
    cfg = make_sim_config(dataset_config, scene_id, width, height)
    sim = habitat_sim.Simulator(cfg)
    try:
        sim.pathfinder.seed(seed)
        agent = sim.get_agent(0)

        state = habitat_sim.AgentState()
        state.position = sim.pathfinder.get_random_navigable_point()
        agent.set_state(state)

        obs = sim.get_sensor_observations()
    finally:
        sim.close()

    return Observation(
        rgb=obs["rgb"][..., :3],
        depth=obs["depth"],
        semantic=obs["semantic"],
    )
