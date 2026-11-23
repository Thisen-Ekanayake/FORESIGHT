"""NVIDIA GEAR-SONIC walking for live G1 runs: SONIC's own G1 in a plain Isaac Lab simulation (no manager env),
driven by FORESIGHT's (forward m/s, yaw rate rad/s) command, one 20 ms control step at a time.

Used by tools/isaac/g1_live.py --walker sonic (FORESIGHT decides; SONIC turns each decision into a human-like walk)
and, for the robot config, by tools/isaac/g1_sonic.py (scripted profiles). The planner + tracking policy port is
tools/isaac/sonic.py.

Command mapping (as SONIC's own gamepad deploy: left stick = move, right stick = turn the facing angle):
  forward v >= v_min   slow_walk at clip(v * speed_gain, 0.1, 0.8) m/s, moving and facing the commanded heading
  forward v <  v_min   idle (stand) facing the commanded heading: turning on the spot
  yaw rate w           integrates the commanded heading, kept within +-lead rad of the robot's actual heading so a
                       held turn can't wind up past what the robot can follow
speed_gain: FORESIGHT speed x gain = the slow_walk speed setting. In the scanned home (probe, 2026-11-23) SONIC
walked at its setting (0.27 -> 0.27 m/s), so the default is 1.0. (It overshot on the office test: 0.4 -> ~0.5 m/s.)
spawn_z 0.80: just above SONIC's standing height (0.789 m). Spawned lower (0.76), the feet start inside the scan's
triangle-mesh floor and the depenetration knocks the robot over in 0.5 s (a plane forgives it).
"""
from __future__ import annotations

import importlib.util
import math
import os
import re

import numpy as np

import sonic as S

PHYSX_SMALL = dict(gpu_max_rigid_contact_count=2**19, gpu_max_rigid_patch_count=2**16, gpu_found_lost_pairs_capacity=2**18,
                   gpu_found_lost_aggregate_pairs_capacity=2**20, gpu_total_aggregate_pairs_capacity=2**18,
                   gpu_collision_stack_size=2**24, gpu_heap_capacity=2**24, gpu_temp_buffer_capacity=2**22,
                   gpu_max_soft_body_contacts=2**10, gpu_max_particle_contacts=2**10)


def robot_cfg(wbc: str, root: str, prim_path: str = "/World/Robot"):
    """SONIC's G1 (29 DoF, Dex3 hands as fixed links): NVIDIA's URDF with mesh paths made absolute, and the gains,
    armature and default pose of G1_CYLINDER_MODEL_12_DEX_CFG (gear_sonic/envs/manager_env/robots/g1.py)."""
    desc = os.path.join(wbc, "gear_sonic/data/assets/robot_description")
    urdf_dir = os.path.join(root, "results/isaac/sonic/robot")
    os.makedirs(urdf_dir, exist_ok=True)
    src = open(os.path.join(desc, "urdf/g1/main.urdf")).read()
    urdf = os.path.join(urdf_dir, "g1_sonic.urdf")
    open(urdf, "w").write(re.sub(r"package://robot_description/", desc + "/", src))
    spec = importlib.util.spec_from_file_location("sonic_g1", os.path.join(wbc, "gear_sonic/envs/manager_env/robots/g1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path=prim_path)
    cfg.spawn.asset_path = urdf
    return cfg


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SonicWorld:
    """The world for a live run: ground plane, an optional scene USD (its own colliders), SONIC's G1 at `start`
    (x, y, yaw rad), 200 Hz physics. step(v, w) advances one 20 ms control step."""

    dt = 0.02

    def __init__(self, *, wbc: str, root: str, scene_usd: str | None, start, device: str = "cuda:0", seed: int = 1234,
                 gpu_onnx: bool = True, speed_gain: float = 1.0, v_min: float = 0.06, lead: float = 0.8,
                 replan: int = 5, spawn_z: float = 0.80):
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

        self.torch, self.device = torch, device
        self.speed_gain, self.v_min, self.lead, self.replan_every = speed_gain, v_min, lead, replan
        simcfg = sim_utils.SimulationCfg(dt=0.005, render_interval=4, device=device)
        # PhysX buffers for one robot, not thousands (~2.5 GB otherwise): SimulationCfg.physics is None by default in
        # Isaac Lab 3 (the manager envs fill it in), so give it a PhysxCfg of our own
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
        px = PhysxCfg()
        for k, v in PHYSX_SMALL.items():
            if hasattr(px, k):
                setattr(px, k, v)
        simcfg.physics = px
        self.sim = sim_utils.SimulationContext(simcfg)
        TerrainImporter(TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1, num_envs=1,
                                           env_spacing=0.0,
                                           physics_material=sim_utils.RigidBodyMaterialCfg(
                                               static_friction=1.0, dynamic_friction=1.0, restitution=0.0)))
        sim_utils.DomeLightCfg(intensity=1500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.0))
        if scene_usd:
            sim_utils.UsdFileCfg(usd_path=scene_usd).func("/World/Set", sim_utils.UsdFileCfg(usd_path=scene_usd))
        rcfg = robot_cfg(wbc, root)
        x0, y0, yaw0 = start
        rcfg.init_state.pos = (x0, y0, spawn_z)
        rcfg.init_state.rot = (0.0, 0.0, math.sin(yaw0 / 2), math.cos(yaw0 / 2))  # Isaac Lab 3: x y z w
        self.robot = Articulation(rcfg)
        self.sim.reset()
        self.stage = self.sim.stage
        names = list(self.robot.joint_names)
        assert sorted(names) == sorted(S.ISAAC_NAMES), f"joint set mismatch: {names}"
        self.joint_names = names
        self.sim_of_isaac = np.array([names.index(n) for n in S.ISAAC_NAMES])
        self._idx = torch.as_tensor(self.sim_of_isaac, device=device)

        jp = torch.zeros(1, 29, device=device)
        jp[0, self._idx] = torch.as_tensor(S.DEFAULT, dtype=torch.float32, device=device)
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        self.robot.set_joint_position_target(jp)
        self.robot.write_data_to_sim()
        self.robot.reset()

        self.planner = S.Planner(seed=seed, gpu=gpu_onnx)
        self.tracker = S.Tracker(gpu=gpu_onnx)
        self.planner.initialize(S.DEFAULT_MJ)
        _, quat0, _, _, _ = self.state()
        self.tracker.start(quat0, self.planner.motion.quat[self.planner.frame])
        self.inv_delta = S.qconj(self.tracker.apply_delta)  # world -> planner frame (yaw only)
        self.psi = S.qyaw(quat0)  # commanded heading (world yaw)
        self.k = 0
        self.last = ("idle", -1.0)

    def state(self):
        d = self.robot.data
        p = d.root_link_pose_w.torch[0].cpu().numpy().astype(float)  # x y z qx qy qz qw
        quat = np.array([p[6], p[3], p[4], p[5]])  # -> w x y z
        w_b = d.root_ang_vel_b.torch[0].cpu().numpy().astype(float)
        q = d.joint_pos.torch[0].cpu().numpy().astype(float)[self.sim_of_isaac]
        dq = d.joint_vel.torch[0].cpu().numpy().astype(float)[self.sim_of_isaac]
        return p[:3], quat, w_b, q, dq

    def _dir(self, yaw):
        return S.qrot(self.inv_delta, [math.cos(yaw), math.sin(yaw), 0.0]).tolist()

    def step(self, v: float, w: float):
        """One 20 ms control step at forward speed v (m/s) and yaw rate w (rad/s, + = turn left, Isaac's sign)."""
        torch = self.torch
        pos, quat, w_b, q, dq = self.state()
        yaw = S.qyaw(quat)
        self.psi = yaw + float(np.clip(_wrap(self.psi + w * self.dt - yaw), -self.lead, self.lead))
        if self.k % self.replan_every == 0:
            face = self._dir(self.psi)
            if v < self.v_min:
                mode, vel, move = "idle", -1.0, [0.0, 0.0, 0.0]
            else:
                mode, vel, move = "slow_walk", float(np.clip(v * self.speed_gain, 0.1, 0.8)), face
            self.planner.replan(S.MODES[mode], vel, move, face)
            self.last = (mode, vel)
        tgt = self.tracker.step(self.planner.motion, self.planner.frame, quat, w_b, q, dq)
        jt = torch.zeros(1, 29, device=self.device)
        jt[0, self._idx] = torch.as_tensor(tgt, dtype=torch.float32, device=self.device)
        self.robot.set_joint_position_target(jt)
        for _ in range(4):
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(0.005)
        self.planner.advance()
        self.k += 1

    def reference(self):
        """The planner's reference frame for this step (joint positions in Isaac Lab order, root quat w x y z)."""
        f = self.planner.frame
        return self.planner.motion.q[f], self.planner.motion.quat[f]
