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
                 max_angular_accel: float = np.deg2rad(120.0), replan_look: bool = True, plan_kw: Optional[dict] = None,
                 floor_scale: bool = True, worker=None, escape: bool = False, escape_after_s: float = 2.0,
                 escape_s: float = 1.2, escape_mps: float = 0.15, robust_scan: bool = False):
        """look_around_s: spin in place (~360 deg at max_angular_radps) to seed the map, the usual start of an
        exploration run; both conditions do it. The same look is repeated when the route suddenly reverses
        (a new obstacle closed the way it was heading: look before committing to a long detour) or the robot
        has stopped making progress (replan_look=False keeps only the stuck trigger, for an explorer whose goal
        legitimately jumps between frontiers). The accel limits ramp commands the way a real
        base would (braking is allowed to be sharper than speeding up). plan_kw: overrides for `local_map.plan`.
        floor_scale=False uses the depth as given (no floor-plane rescaling): the depth-error study switches the
        self-calibration off to see what a uniform scale error costs without it. worker: a `PerceptionWorker` whose
        looks ground "mirror" (its `results` feed `reflection_segments`), for the fast/slow mirror guard.
        escape: if the safety layer has vetoed forward motion for `escape_after_s` while the route still points ahead
        (within 35 deg), reverse at `escape_mps` for `escape_s` (the safety layer is not consulted: the camera can't
        see behind), then replan. Without it a robot that starts or ends up ~0.3 m from a wall can face it forever:
        the route's first leg runs through the inflated wall, the planner's turn is ~0 and the stop never clears
        (point-goal pilot, a-00800-02 with ground-truth depth: 109 s without moving, 2026-11-02). Off by default, so
        object search keeps its evaluated behaviour. robust_scan: `local_map.depth_scan(robust=True)` (median-filtered
        depth, a column's range needs supporting points), which removes the phantom obstacles zero-mean depth noise
        makes in the plain nearest-point scan. Off by default: deliverable A's results use the plain scan."""
        self.goal = np.asarray(goal_xz, float)
        self.map = OccupancyMap(np.asarray(map_center_xz, float), size_m=map_size_m)
        self.scale = FloorScale()
        self.use_floor_scale = floor_scale
        self.worker = worker
        self.max_linear_mps, self.max_angular_radps, self.heading_gain = max_linear_mps, max_angular_radps, heading_gain
        self.look_around_s, self.replan_look = look_around_s, replan_look
        self.plan_kw = dict(plan_kw or {})
        self.last_path: list = []  # the route planned on the latest tick (world xz points)
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
        self.escape, self.escape_after_s, self.escape_s, self.escape_mps = escape, escape_after_s, escape_s, escape_mps
        self._blocked_since = None
        self._escape_until = -np.inf
        self.n_escapes = 0
        self.robust_scan = robust_scan

    def step(self, depth: np.ndarray, pose: dict, t: float, face_xz=None, creep_mps: float = 0.0):
        """face_xz: if given, hold position and turn to face this direction instead of following the route
        (FORESIGHT uses it to look at a flagged view again); an ongoing look-around is paused meanwhile.
        creep_mps: while facing, drive straight at this speed once within 15 deg of face_xz (closing in on an
        object; the safety layer still applies)."""
        s = self.scale.update(depth) if self.use_floor_scale else 1.0
        bearings, ranges = depth_scan(depth, s, robust=self.robust_scan)
        pos = np.asarray(pose["position"], float)[[0, 2]]
        fwd = np.asarray(pose["forward_xz"], float)
        self.map.integrate(pos, fwd, bearings, ranges)
        self.scans[t] = (pos, fwd, bearings, ranges)
        pl = plan(self.map, pos, self.goal, **self.plan_kw)
        self.last_path = pl.path_xz
        right = np.array([-fwd[1], fwd[0]])
        looking = False
        if face_xz is not None:
            self._trail = []  # holding still on purpose is not being stuck
            if t < self._look[1] and self._last is not None:
                self._look = (self._look[0], self._look[1] + (t - self._last[0]), self._look[2])
            face = np.asarray(face_xz, float)
            err = float(np.arctan2(face @ right, face @ fwd))
            linear = creep_mps if abs(err) < np.deg2rad(15.0) else 0.0
            angular = float(np.clip(self.heading_gain * err, -self.max_angular_radps, self.max_angular_radps))
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
        escaping = False
        if self.escape:
            route_ahead = (not looking and face_xz is None and pl.carrot_xz is not None
                           and abs(np.arctan2((pl.carrot_xz - pos) @ right, (pl.carrot_xz - pos) @ fwd)) < np.deg2rad(35.0))
            if safe.level == "stop" and route_ahead and t >= self._escape_until:
                self._blocked_since = t if self._blocked_since is None else self._blocked_since
                if t - self._blocked_since >= self.escape_after_s:
                    self._escape_until, self._blocked_since = t + self.escape_s, None
                    self.n_escapes += 1
            elif safe.level != "stop":
                self._blocked_since = None
            escaping = t < self._escape_until
        if escaping:
            linear, angular = self._ramp(t, -self.escape_mps, 0.0)
        else:
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
            "escaping": escaping,
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
        if (self.replan_look and d is not None and self._prev_dir is not None
                and d @ self._prev_dir < np.cos(np.deg2rad(flip_deg)) and fresh):
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

    def through_wall(self, a_xz, b_xz) -> bool:
        if not self.map.walls:
            return False
        a, b = np.asarray(a_xz, float), np.asarray(b_xz, float)
        n = max(int(np.linalg.norm(b - a) / (self.map.res / 2)), 2)
        pts = a[None, :] + (b - a)[None, :] * np.linspace(0.05, 0.95, n)[:, None]
        return bool(self.map.virtual[self.map.cell(pts)].any())

    def reflection_segments(self, seen_t: float, min_gap_ratio: float = 1.3, far_m: float = 4.0,
                            extend_m: float = 0.25, hfov_deg: float = 90.0) -> list:
        """Wall off a confirmed reflection where Qwen boxed the mirror in that view (not across the whole view: a
        note can be about something off to the side). Per mirror box: the scan bearings inside the box seed a
        "phantom" region, grown sideways while the scan stays about as deep; the first shallower hits on either
        side are the real surfaces framing the mirror, and the wall joins them (so it lies roughly on the mirror's
        plane and leaves no gap at its ends). Only if the phantom reads `min_gap_ratio` x deeper than its frame.
        No mirror box, or no gap -> no wall. -> [(a_xz, b_xz, view fwd)]."""
        res = self.worker.results.get(seen_t)
        if res is None or seen_t not in self.scans:
            return []
        pos, fwd, b, r = self.scans[seen_t]
        r = np.where(np.isfinite(r), r, far_m)
        tan_h = np.tan(np.deg2rad(hfov_deg) / 2)
        segs = []
        for box in res["boxes"].get("mirror", []):
            a1, a2 = (np.arctan((box[k] / 1000 - 0.5) * 2 * tan_h) for k in (0, 2))
            span = a2 - a1
            inner = np.nonzero((b > a1 + 0.1 * span) & (b < a2 - 0.1 * span))[0]
            if span < np.deg2rad(6) or inner.size == 0:
                continue
            d_in = float(np.median(r[inner]))
            deep = r > d_in / min_gap_ratio
            lo, hi = int(inner.min()), int(inner.max())
            while lo > 0 and deep[lo - 1]:
                lo -= 1
            while hi < len(b) - 1 and deep[hi + 1]:
                hi += 1
            frame = [i for i in (lo - 1, hi + 1) if 0 <= i < len(b)]  # first real surfaces beside the phantom
            if not frame:
                continue
            d_edge = float(np.mean(r[frame]))
            if d_in < min_gap_ratio * d_edge:
                continue
            # If one "frame" reads much nearer than the other it is something standing in front of the mirror (a door
            # frame on the near side), not the mirror's own frame: joining it gave a slanted wall across a real doorway
            # (00808, 2026-11-01). Then both ends go at the farther frame's distance, roughly the mirror's plane.
            d_frame = {i: float(r[i]) for i in frame}
            if len(frame) == 2 and max(d_frame.values()) > 1.35 * min(d_frame.values()):
                d_frame = {i: max(d_frame.values()) for i in frame}
            ends = []
            for i, edge in ((lo - 1, lo), (hi + 1, hi)):
                k = i if 0 <= i < len(b) else edge  # phantom runs off the image: guess the frame's distance
                ends.append(pos + _rotate(fwd, np.array(b[k])) * (d_frame[k] if k == i else max(d_frame.values())))
            a, c = ends
            u = (c - a) / max(np.linalg.norm(c - a), 1e-6)
            segs.append((a - u * extend_m, c + u * extend_m, fwd))
        return segs
