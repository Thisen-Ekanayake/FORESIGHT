"""Goal-seeking on a self-built occupancy map (see `foresight.planning.local_map`).

Shared by both mirror-trap conditions, so the only difference between them is whether Laya's reflection
flag gets to add a virtual wall: depth -> floor-plane scale -> scan -> map -> Dijkstra -> carrot ->
the shared `heading_to_velocity` control law -> the shared `ReactiveSafetyLayer` (on scale-corrected depth).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from foresight.planning.fusion import heading_to_velocity
from foresight.planning.local_map import FloorScale, OccupancyMap, _rotate, depth_scan, plan
from foresight.safety.reactive import ReactiveSafetyLayer


class MapNavigator:
    def __init__(self, goal_xz, map_center_xz, map_size_m: float = 16.0, max_linear_mps: float = 0.35,
                 max_angular_radps: float = np.deg2rad(40.0), heading_gain: float = 1.5,
                 look_around_s: float = 8.0, safety: Optional[ReactiveSafetyLayer] = None,
                 max_linear_accel: float = 0.4, max_linear_decel: float = 1.5,
                 max_angular_accel: float = np.deg2rad(120.0)):
        """look_around_s: spin in place (~360 deg at max_angular_radps) to seed the map, the usual start of an
        exploration run; both conditions do it. The same look is repeated when the route suddenly reverses
        (a new obstacle closed the way it was heading: look before committing to a long detour) or the robot
        has stopped making progress. The accel limits ramp commands the way a real
        base would (braking is allowed to be sharper than speeding up)."""
        self.goal = np.asarray(goal_xz, float)
        self.map = OccupancyMap(np.asarray(map_center_xz, float), size_m=map_size_m)
        self.scale = FloorScale()
        self.max_linear_mps, self.max_angular_radps, self.heading_gain = max_linear_mps, max_angular_radps, heading_gain
        self.look_around_s = look_around_s
        # The planner already routes around mapped obstacles, so this is a last-resort guard: tighter
        # margins than the reactive-only controllers use, and robust to stray door-frame pixels.
        self.safety = safety or ReactiveSafetyLayer(stop_distance_m=0.3, slow_distance_m=0.7, clearance_quantile=0.02)
        self.max_linear_accel, self.max_linear_decel = max_linear_accel, max_linear_decel
        self.max_angular_accel = max_angular_accel
        self._last = None  # (t, linear, angular) actually commanded
        self._look = (0.0, look_around_s, "start")  # (from, until, reason) of the current/last look-around
        self._look_spots = []  # positions where a look-around already happened
        self._prev_dir = None  # unit direction of the route ~1.5 m ahead, last tick
        self._trail = []  # (t, pos) for the stuck check
        self.scans: dict = {}  # sim time -> (pos, fwd, bearings, ranges): reflection walls are drawn from the flagged view

    def step(self, depth: np.ndarray, pose: dict, t: float, face_xz=None):
        """face_xz: if given, hold position and turn to face this direction instead of following the route
        (FORESIGHT uses it to look at a flagged view again); an ongoing look-around is paused meanwhile."""
        s = self.scale.update(depth)
        bearings, ranges = depth_scan(depth, s)
        pos = np.asarray(pose["position"], float)[[0, 2]]
        fwd = np.asarray(pose["forward_xz"], float)
        self.map.integrate(pos, fwd, bearings, ranges)
        self.scans[t] = (pos, fwd, bearings, ranges)
        pl = plan(self.map, pos, self.goal)
        right = np.array([-fwd[1], fwd[0]])
        looking = False
        if face_xz is not None:
            self._trail = []  # holding still on purpose is not being stuck
            if t < self._look[1] and self._last is not None:
                self._look = (self._look[0], self._look[1] + (t - self._last[0]), self._look[2])
            face = np.asarray(face_xz, float)
            err = float(np.arctan2(face @ right, face @ fwd))
            linear, angular = 0.0, float(np.clip(self.heading_gain * err, -self.max_angular_radps, self.max_angular_radps))
        elif self._maybe_look(t, pos, pl) or t < self._look[1]:
            looking = True
            linear, angular = 0.0, self.max_angular_radps
        elif pl.carrot_xz is not None:
            v = pl.carrot_xz - pos
            heading = float(np.arctan2(v @ right, v @ fwd))
            linear, angular = heading_to_velocity(heading, self.max_linear_mps, self.max_angular_radps, self.heading_gain)
        else:  # no route at all: turn on the spot to look around
            linear, angular = 0.0, self.max_angular_radps
        scaled = depth * s
        safe = self.safety.apply(scaled, linear, angular)
        # On a stop, only forward motion is vetoed: turning in place can't hit anything, and the planner's
        # turn (or the look-around spin) is what gets the robot unstuck, not the guard's left/right guess.
        linear, angular = self._ramp(t, safe.linear_velocity, angular if safe.level == "stop" else safe.angular_velocity)
        h, w = depth.shape
        return linear, angular, {
            "depth_scale": s,
            "depth_ahead_m": float(np.median(scaled[int(.4 * h):int(.9 * h), int(.25 * w):int(.75 * w)])),
            "safety_level": safe.level,
            "min_clearance_m": safe.min_clearance_m,
            "plan_path": [p.tolist() for p in pl.path_xz[::3]],
            "plan_cost": pl.cost_to_goal,
            "looking_around": self._look[2] if looking else None,
            "facing": face_xz is not None,
            "map_score": np.clip(self.map.score * 30, -127, 127).astype(np.int8),
            "map_walls": list(self.map.walls),
        }

    def _maybe_look(self, t, pos, pl, flip_deg: float = 100.0, spot_m: float = 1.5, stuck_s: float = 6.0,
                    stuck_m: float = 0.15):
        if t < self._look[1]:
            return False
        self._trail = [(tt, p) for tt, p in self._trail if t - tt <= stuck_s] + [(t, pos)]
        ahead = [p for p in pl.path_xz if np.linalg.norm(p - pos) >= spot_m]
        d = None if not ahead else (ahead[0] - pos) / np.linalg.norm(ahead[0] - pos)
        reason = None
        fresh = all(np.linalg.norm(pos - q) > spot_m for q in self._look_spots)
        if d is not None and self._prev_dir is not None and d @ self._prev_dir < np.cos(np.deg2rad(flip_deg)) and fresh:
            reason = "replan"
        elif (t - self._look[1] > stuck_s and self._trail[0][0] <= t - stuck_s + 0.2
              and np.linalg.norm(pos - self._trail[0][1]) < stuck_m):
            reason = "stuck"
        self._prev_dir = d
        if reason:
            self._look = (t, t + self.look_around_s, reason)
            self._look_spots.append(pos)
            self._trail = []
            self._prev_dir = None
            return True
        return False

    def _ramp(self, t, linear, angular):
        if self._last is not None:
            t0, l0, a0 = self._last
            dt = max(t - t0, 1e-3)
            linear = float(np.clip(linear, l0 - self.max_linear_decel * dt, l0 + self.max_linear_accel * dt))
            angular = float(np.clip(angular, a0 - self.max_angular_accel * dt, a0 + self.max_angular_accel * dt))
        self._last = (t, linear, angular)
        return linear, angular

    def view_walled(self, seen_t: float, center_deg: float = 12.0, max_m: float = 4.0) -> bool:
        """Whether the centre of the view seen at `seen_t` already runs into a reflection wall."""
        if seen_t not in self.scans:
            return False
        pos, fwd = self.scans[seen_t][:2]
        dirs = _rotate(fwd, np.deg2rad(np.array([-center_deg, 0.0, center_deg])))
        steps = np.arange(0.2, max_m, self.map.res / 2)
        hits = [self.map.virtual[self.map.cell(pos[None, :] + d[None, :] * steps[:, None])].any() for d in dirs]
        return sum(hits) >= 2

    def add_reflection_wall(self, seen_t: float, center_deg: float = 12.0, flank_deg=(15.0, 40.0),
                            min_gap_ratio: float = 1.3) -> bool:
        """Draw a wall across the view seen at `seen_t`, at the distance of the surfaces flanking its
        centre — but only if the centre reads as a too-deep gap (what a mirror's phantom hallway looks like).
        A flag on a view with no such gap (e.g. 'the artwork reflects light') adds nothing."""
        if seen_t not in self.scans:
            return False
        pos, fwd, b, r = self.scans[seen_t]
        deg = np.rad2deg(np.abs(b))
        centre = np.where(np.isfinite(r[deg < center_deg]), r[deg < center_deg], 4.0)
        flank = r[(deg >= flank_deg[0]) & (deg <= flank_deg[1])]
        flank = flank[np.isfinite(flank)]
        if centre.size == 0 or flank.size == 0:
            return False
        d_center, d_edge = float(np.median(centre)), float(np.median(flank))
        if d_center < min_gap_ratio * d_edge:
            return False
        d_wall = float(np.clip(d_edge, 0.8, 3.0))
        self.map.add_wall(pos, fwd, d_wall, d_wall * np.tan(np.deg2rad(35.0)) + 0.3)
        return True
