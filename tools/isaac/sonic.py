"""NVIDIA GEAR-SONIC for the Unitree G1 (29 DoF), ported from the C++ reference deploy to Python + onnxruntime.

SONIC = a kinematic motion planner (planner_sonic.onnx: speed / direction / mode -> 30 Hz reference motion) plus a
motion-tracking whole-body policy (sonic_v1_1 encoder -> 64-d motion token -> decoder -> 29 joint targets, 50 Hz).
Weights: NVIDIA Open Model License (https://huggingface.co/nvidia/GEAR-SONIC). Reference implementation followed
line by line: NVlabs/GR00T-WholeBodyControl gear_sonic_deploy/src/g1/g1_deploy_onnx_ref (localmotion_kplanner.hpp,
g1_deploy_onnx_ref.cpp, policy_parameters.hpp) and gear_sonic/envs/manager_env/robots/g1.py (gains).

Conventions (as in the reference):
  * joint vectors are in IsaacLab order (ISAAC_NAMES) unless named *_mj (MuJoCo / motor order, MJ_NAMES);
  * quaternions are (w, x, y, z);
  * the planner works in its own world frame that starts at the origin facing +x; only headings reach the policy,
    through apply_delta = heading(robot at start) * heading(reference at start)^-1.
"""
import math
import os

import numpy as np

MODEL_DIR = os.environ.get("SONIC_DIR", "/ml/Models/gear_sonic")

MJ_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
# policy_parameters.hpp: ISAAC_TO_MJ[mujoco index] = isaaclab index;  MJ_OF_ISAAC[isaaclab index] = mujoco index
ISAAC_TO_MJ = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22,
                        24, 26, 28])
MJ_OF_ISAAC = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20,
                        27, 21, 28])
ISAAC_NAMES = [MJ_NAMES[MJ_OF_ISAAC[i]] for i in range(29)]
assert all(ISAAC_TO_MJ[MJ_OF_ISAAC[i]] == i for i in range(29))

DEFAULT_MJ = np.array([-0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0, 0, 0, 0,
                       0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0])
DEFAULT = DEFAULT_MJ[MJ_OF_ISAAC]

# motor constants (gear_sonic robots/g1.py; BeyondMimic)
_ARM = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_W = 10 * 2.0 * 3.1415926535
_STIFF = {k: v * _W ** 2 for k, v in _ARM.items()}
_DAMP = {k: 2.0 * 2.0 * v * _W for k, v in _ARM.items()}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_MOTOR_MJ = (["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"] * 2 + ["7520_14", "5020", "5020"]
             + ["5020"] * 5 + ["4010", "4010"] + ["5020"] * 5 + ["4010", "4010"])
_X2 = {4, 5, 10, 11, 13, 14}  # ankles and waist roll/pitch: double stiffness/damping/armature
KP_MJ = np.array([_STIFF[m] * (2.0 if i in _X2 else 1.0) for i, m in enumerate(_MOTOR_MJ)])
KD_MJ = np.array([_DAMP[m] * (2.0 if i in _X2 else 1.0) for i, m in enumerate(_MOTOR_MJ)])
ACTION_SCALE_MJ = np.array([0.25 * _EFFORT[m] / _STIFF[m] for m in _MOTOR_MJ])  # policy_parameters.hpp g1_action_scale
ACTION_SCALE = ACTION_SCALE_MJ[MJ_OF_ISAAC]

LOWER_BODY_MJ_ORDER_ISAAC_IDX = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18]
DEFAULT_HEIGHT = 0.788740
MODES = {"idle": 0, "slow_walk": 1, "walk": 2, "run": 3}


# ----------------------------------------------------------------------------- quaternion helpers (w, x, y, z)
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v):
    return qmul(qmul(q, np.array([0.0, *v])), qconj(q))[1:]


def qyaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def qheading(q):  # calc_heading_quat: yaw-only part
    a = qyaw(q)
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def qslerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _session(path, gpu=True):
    import onnxruntime as ort
    try:
        ort.preload_dlls()
    except Exception:
        pass
    prov = (["CUDAExecutionProvider"] if gpu else []) + ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=prov)


class Motion:
    """A 50 Hz reference: root position (T, 3), root quaternion (T, 4), joints (T, 29), joint velocities (T, 29)."""

    def __init__(self, pos, quat, q, dq):
        self.pos, self.quat, self.q, self.dq = pos, quat, q, dq

    @property
    def T(self):
        return len(self.q)


class Planner:
    """planner_sonic.onnx with the reference's context sampling, 30 -> 50 Hz resampling and 8-frame blend-in."""

    LOOKAHEAD = 2  # motion_look_ahead_steps (50 Hz)
    BLEND = 8

    def __init__(self, path=None, seed=1234, gpu=True):
        self.s = _session(path or os.path.join(MODEL_DIR, "planner_sonic.onnx"), gpu)
        self.seed = seed
        self.motion = None
        self.frame = 0  # current_frame_ on the (rebased) planner motion

    def _run(self, ctx, mode, vel, move, face, height=-1.0):
        feed = {"context_mujoco_qpos": ctx[None].astype(np.float32), "target_vel": np.array([vel], np.float32),
                "mode": np.array([mode], np.int64), "movement_direction": np.array([move], np.float32),
                "facing_direction": np.array([face], np.float32), "random_seed": np.array([self.seed], np.int64),
                "has_specific_target": np.zeros((1, 1), np.int64),
                "specific_target_positions": np.zeros((1, 4, 3), np.float32),
                "specific_target_headings": np.zeros((1, 4), np.float32),
                "allowed_pred_num_tokens": np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64),
                "height": np.array([height], np.float32)}
        qpos, n = self.s.run(None, feed)
        n = int(np.asarray(n).reshape(-1)[0])
        if not (0 < n <= 64) or not np.all(np.isfinite(qpos[0, :n])):
            raise RuntimeError(f"planner returned {n} frames / non-finite values")
        return self._resample(qpos[0, :n].astype(np.float64))

    @staticmethod
    def _resample(qp):  # ResampleGeneratedSequence50Hz
        n30 = len(qp)
        T = int(math.floor(n30 / 30.0 * 50))
        pos, quat, q = np.zeros((T, 3)), np.zeros((T, 4)), np.zeros((T, 29))
        for f in range(T):
            f30 = f / 50.0 * 30
            f0 = int(math.floor(f30))
            f1 = min(f0 + 1, n30 - 1)
            w1 = f30 - f0
            pos[f] = (1 - w1) * qp[f0, :3] + w1 * qp[f1, :3]
            quat[f] = qslerp(qp[f0, 3:7], qp[f1, 3:7], w1)
            q[f] = (1 - w1) * qp[f0, 7 + MJ_OF_ISAAC] + w1 * qp[f1, 7 + MJ_OF_ISAAC]
        dq = np.zeros_like(q)
        dq[:-1] = (q[1:] - q[:-1]) * 50.0
        dq[-1] = dq[-2]
        return Motion(pos, quat, q, dq)

    def initialize(self, joints_mj):
        """InitializeContext: 4 identical standing frames at the origin facing +x; first plan is IDLE."""
        ctx = np.zeros((4, 36))
        ctx[:, 2] = DEFAULT_HEIGHT
        ctx[:, 3] = 1.0
        ctx[:, 7:] = joints_mj
        self.motion = self._run(ctx, MODES["idle"], -1.0, [0, 0, 0], [1, 0, 0])
        self.frame = 0

    def replan(self, mode, vel, move, face, height=-1.0):
        """UpdatePlanning + the control loop's rebase/blend (g1_deploy_onnx_ref.cpp ~3226-3330)."""
        m, cur = self.motion, self.frame
        fgen = cur + self.LOOKAHEAD
        ctx = np.zeros((4, 36))
        for n in range(4):  # UpdateContextFromMotion: 4 frames at 30 Hz from the generation frame
            f50 = (fgen / 50.0 + n / 30.0) * 50.0
            f0 = min(int(math.floor(f50)), m.T - 1)
            f1 = min(f0 + 1, m.T - 1)
            w1 = f50 - f0
            ctx[n, 3:7] = qslerp(m.quat[f0], m.quat[f1], f50 - f0)
            ctx[n, :3] = (1 - w1) * m.pos[f0] + w1 * m.pos[f1]
            ctx[n, 7 + MJ_OF_ISAAC] = (1 - w1) * m.q[f0] + w1 * m.q[f1]
        g = self._run(ctx, mode, vel, move, face, height)
        L = fgen - cur + g.T
        if L <= 0:
            return
        bs = max(0, fgen - cur)
        pos, quat, q, dq = np.zeros((L, 3)), np.zeros((L, 4)), np.zeros((L, 29)), np.zeros((L, 29))
        for f in range(L):
            fo = int(np.clip(f + cur, 0, m.T - 1))
            fn = int(np.clip(f + cur - fgen, 0, g.T - 1))
            wn = float(np.clip((f - bs) / self.BLEND, 0.0, 1.0))
            q[f] = (1 - wn) * m.q[fo] + wn * g.q[fn]
            dq[f] = (1 - wn) * m.dq[fo] + wn * g.dq[fn]
            pos[f] = (1 - wn) * m.pos[fo] + wn * g.pos[fn]
            quat[f] = qslerp(m.quat[fo], g.quat[fn], wn)
        self.motion = Motion(pos, quat, q, dq)
        self.frame = 0

    def advance(self):
        self.frame = min(self.frame + 1, self.motion.T - 1)


class Tracker:
    """sonic_v1_1 encoder (g1 mode) + decoder, with the 10-frame histories of the reference deploy."""

    ENC_LAYOUT = [("encoder_mode_4", 4), ("motion_joint_positions_10frame_step5", 290),
                  ("motion_joint_velocities_10frame_step5", 290), ("motion_anchor_orientation_heading_10frame_step5", 60),
                  ("motion_anchor_orientation_heading", 6), ("motion_joint_positions_lowerbody_10frame_step5", 120),
                  ("motion_joint_velocities_lowerbody_10frame_step5", 120), ("vr_3point_local_target", 9),
                  ("vr_3point_local_orn_target", 12), ("smpl_joints_10frame_step1", 720),
                  ("smpl_anchor_orientation_heading_10frame_step1", 60), ("motion_joint_positions_wrists_10frame_step1", 60)]

    def __init__(self, variant="sonic_v1_1", gpu=True):
        d = os.path.join(MODEL_DIR, variant)
        self.enc = _session(os.path.join(d, "model_encoder.onnx"), gpu)
        self.dec = _session(os.path.join(d, "model_decoder.onnx"), gpu)
        off, self.enc_off = 0, {}
        for name, n in self.ENC_LAYOUT:
            self.enc_off[name] = (off, n)
            off += n
        assert off == 1751
        self.hist = []  # oldest -> newest: (ang_vel_b, q_rel, dq, last_action, gravity_b)
        self.last_action = np.zeros(29)
        self.apply_delta = None

    def start(self, base_quat, ref_root_quat):
        """reinitialize_heading_: robot heading at start vs the reference's heading at the current frame."""
        self.apply_delta = qmul(qheading(base_quat), qconj(qheading(ref_root_quat)))

    def _encoder_obs(self, motion, frame, base_quat):
        o = np.zeros(1751)
        T = motion.T
        idx = [min(frame + k * 5, T - 1) for k in range(10)]
        a, _ = self.enc_off["motion_joint_positions_10frame_step5"]
        o[a:a + 290] = motion.q[idx].reshape(-1)
        a, _ = self.enc_off["motion_joint_velocities_10frame_step5"]
        o[a:a + 290] = motion.dq[idx].reshape(-1)
        a, _ = self.enc_off["motion_anchor_orientation_heading_10frame_step5"]
        left = qconj(qheading(base_quat))
        for k, f in enumerate(idx):  # orientation_mode 1: robot heading only; 6-d = rot-matrix cols 0-1, row-wise
            r = qmat(qmul(left, qmul(self.apply_delta, motion.quat[f])))
            o[a + 6 * k:a + 6 * k + 6] = r[:, :2].reshape(-1)
        return o  # encoder_mode_4 = [0 (g1), 0, 0, 0]

    def step(self, motion, frame, base_quat, ang_vel_b, q, dq):
        """One 50 Hz control step. q/dq: IsaacLab order, absolute. Returns joint position targets (IsaacLab order)."""
        g = qrot(qconj(base_quat), [0.0, 0.0, -1.0])
        self.hist.append((np.asarray(ang_vel_b, float), q - DEFAULT, np.asarray(dq, float), self.last_action.copy(), g))
        self.hist = self.hist[-10:]
        H = [self.hist[0]] * (10 - len(self.hist)) + self.hist
        tok = self.enc.run(None, {"obs_dict": self._encoder_obs(motion, frame, base_quat)[None].astype(np.float32)})[0][0]
        obs = np.concatenate([tok] + [np.concatenate([h[i] for h in H]) for i in (0, 1, 2, 3, 4)])
        assert obs.shape == (994,)
        act = self.dec.run(None, {"obs_dict": obs[None].astype(np.float32)})[0][0].astype(np.float64)
        self.last_action = act
        return DEFAULT + act * ACTION_SCALE
