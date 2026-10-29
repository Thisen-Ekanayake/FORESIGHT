"""Language-goal object search ("find the sofa, then go back to the bed") on a self-built map.

No prior map and no goal coordinates: the robot explores frontiers of its own occupancy map (built from
zero-shot monocular depth, `foresight.planning.local_map`), and a background Qwen3-VL worker grounds the
object names it was given in what the camera sees. Each box is lifted onto the map with the box's own
depth pixels, and sightings are fused in an `ObjectMemory`:

  - a 2B VLM over-reports objects (a box on ~15-35% of frames without the object in the probe,
    results/runs/object_search/probe_grounding.py), so a remembered object only becomes a goal after two
    sightings, and a look that has it in view without boxing it counts against it;
  - on arrival the robot faces it and waits for one more look before declaring it found;
  - a sighting whose line of sight crosses a reflection wall on the map (FORESIGHT draws one across a
    confirmed mirror) was seen *in* the mirror, and is ignored.

Later instructions can be answered from memory: an object seen while looking for something else is
already on the map, so "go back to the bed" needs no new exploring.

Both conditions use this navigator (same explorer, same grounding, same memory); FORESIGHT additionally
wraps it in `laya_fusion.ForesightNavigator` (VLM scene notes + Laya flagging reflections), which is what
draws reflection walls. The one perception worker runs every VLM call of a look back-to-back, as a single
on-board GPU would: FORESIGHT's looks take longer (description + grounding) than the baseline's.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from foresight.planning.laya_fusion import BackgroundDescriber
from foresight.planning.local_map import _rotate, cost_grid, distances_from, frontiers
from foresight.planning.map_navigator import MapNavigator
from foresight.safety.reactive import ReactiveSafetyLayer


class PerceptionWorker(BackgroundDescriber):
    """One look = [scene description (FORESIGHT only)] + one grounding query per label, run back-to-back on
    the frame seen at the look's start, delivered (measured latency) sim-seconds later. Delivered looks
    queue in `inbox` as (seen_at, {"text", "boxes": {label: [box, ...]}, "ms": {...}})."""

    def __init__(self, locate: Callable[[np.ndarray, str], list], labels: Callable[[], List[str]],
                 describe: Optional[Callable[[np.ndarray], str]] = None, min_latency_s: float = 0.0,
                 direction: Optional[Callable[[np.ndarray, str], np.ndarray]] = None,
                 present: Optional[Callable[[np.ndarray, str], float]] = None, present_threshold: float = 0.5,
                 extra_labels: List[str] = ()):
        """direction(rgb, label) -> P(left, centre, right) as the way to search for `label` (asked for the
        first label, the current target). present(rgb, label) -> P(yes) to "is there a <label>?": grounding
        only runs when it is above `present_threshold` (the yes/no answer is far more precise than the 2B
        model's boxes: results/runs/object_search, a black bench scored 0.00 as a bed, which grounding boxed).
        extra_labels: grounded on every look but never searched for (FORESIGHT: "mirror", to place reflection walls)."""
        def look(rgb):
            out = {"text": None, "boxes": {}, "ms": {}, "direction": None, "present": {}}
            if describe is not None:
                t0 = time.perf_counter()
                out["text"] = describe(rgb)
                out["ms"]["describe"] = (time.perf_counter() - t0) * 1000
            labs = labels()
            for lab in list(labs) + [l for l in extra_labels if l not in labs]:
                t0 = time.perf_counter()
                p = 1.0 if present is None else float(present(rgb, lab))
                out["present"][lab] = p
                out["boxes"][lab] = locate(rgb, lab) if p > present_threshold else []
                out["ms"][lab] = (time.perf_counter() - t0) * 1000
            if direction is not None and labs:
                t0 = time.perf_counter()
                out["direction"] = (labs[0], np.asarray(direction(rgb, labs[0]), float))
                out["ms"]["direction"] = (time.perf_counter() - t0) * 1000
            return out

        super().__init__(look, min_latency_s)
        self.inbox: list = []
        self.results: dict = {}  # seen_at -> look result, for later queries about that view

    def _deliver(self, job) -> None:
        self.latest = job.text["text"]
        self.latest_available_at = job.ready_at
        self.latest_seen_at = job.started_at
        self.inbox.append((job.started_at, job.text))
        self.results[job.started_at] = job.text


def box_to_xz(box, depth: np.ndarray, pos_xz, fwd_xz, hfov_deg: float = 90.0, shrink: float = 0.2,
              band_m: float = 0.35, max_range_m: float = 6.0):
    """Lift a 0-1000 box onto the floor plane: the box's nearest well-populated depth layer (its front
    surface, not the wall behind it), back-projected through the pinhole camera. -> world xz or None."""
    h, w = depth.shape
    x1, y1, x2, y2 = np.asarray(box, float) / 1000.0
    dx, dy = (x2 - x1) * shrink / 2, (y2 - y1) * shrink / 2
    c0, c1 = int((x1 + dx) * w), max(int((x2 - dx) * w), int((x1 + dx) * w) + 1)
    r0, r1 = int((y1 + dy) * h), max(int((y2 - dy) * h), int((y1 + dy) * h) + 1)
    patch = depth[r0:r1, c0:c1]
    ok = np.isfinite(patch) & (patch > 0.1)
    if ok.sum() < 6:
        return None
    d_front = float(np.percentile(patch[ok], 30))
    if d_front > max_range_m:
        return None
    sel = ok & (np.abs(patch - d_front) < band_m)
    rows, cols = np.nonzero(sel)
    f = (w / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    u = cols + c0 - w / 2 + 0.5
    d = patch[sel]
    x_cam, z_cam = float(np.median(u * d / f)), float(np.median(d))
    fwd = np.asarray(fwd_xz, float) / np.linalg.norm(fwd_xz)
    right = np.array([-fwd[1], fwd[0]])
    return np.asarray(pos_xz, float) + fwd * z_cam + right * x_cam


@dataclass
class Sighting:
    t: float
    view_xz: np.ndarray
    xz: np.ndarray


@dataclass
class Remembered:
    label: str
    sightings: list = field(default_factory=list)
    misses: int = 0
    rejected: bool = False

    @property
    def xz(self):
        return np.mean([s.xz for s in self.sightings], axis=0)


class ObjectMemory:
    """Sightings of named objects fused on the map. An object is a candidate goal once it has
    `min_hits` valid sightings (distinct looks) and more hits than misses."""

    def __init__(self, merge_m: float = 1.2, min_hits: int = 3):
        self.merge_m, self.min_hits = merge_m, min_hits
        self.objects: list = []

    def add(self, label, xz, view_xz, t) -> Remembered:
        near = [o for o in self.objects if o.label == label and np.linalg.norm(o.xz - xz) < self.merge_m]
        o = min(near, key=lambda o: np.linalg.norm(o.xz - xz)) if near else None
        if o is None:
            o = Remembered(label)
            self.objects.append(o)
        o.sightings.append(Sighting(t, np.asarray(view_xz, float), np.asarray(xz, float)))
        return o

    def hits(self, o: Remembered, through_wall: Callable) -> int:
        return len({s.t for s in o.sightings if not through_wall(s.view_xz, s.xz)})

    def best(self, label, pos_xz, through_wall: Callable) -> Optional[Remembered]:
        ok = [o for o in self.objects if o.label == label and not o.rejected
              and self.hits(o, through_wall) >= self.min_hits and self.hits(o, through_wall) > o.misses]
        return min(ok, key=lambda o: np.linalg.norm(o.xz - pos_xz)) if ok else None


class ObjectSearchNavigator(MapNavigator):
    """MapNavigator whose goal is picked every tick: the remembered target if confirmed, else a frontier.

    instructions: object labels to reach in order (e.g. ["sofa", "bed"]). worker: the `PerceptionWorker`;
    poll_worker=False when an outer controller (ForesightNavigator) already polls it each tick.
    Exposes the same `step(depth, pose, t, face_xz)` as MapNavigator, plus `rgb=` for polling."""

    def __init__(self, instructions: List[str], worker: PerceptionWorker, start_xz, map_size_m: float = 30.0,
                 poll_worker: bool = True, arrive_m: float = 0.8, arrive_blocked_m: float = 1.4,
                 frontier_every_s: float = 1.0, verify_looks: int = 2, verify_timeout_s: float = 12.0,
                 announce_s: float = 2.0, prior_weight_m: float = 4.0, **kw):
        kw.setdefault("replan_look", False)
        # Margin over the 0.3 m crash rule: monocular depth is ~0.1 m optimistic up close.
        kw.setdefault("safety", ReactiveSafetyLayer(stop_distance_m=0.4, slow_distance_m=0.8, clearance_quantile=0.02))
        # Plan through space the robot has seen: with a 90 deg camera, walls beside it are often still unmapped.
        kw.setdefault("plan_kw", {"unknown_cost": 3.0})
        super().__init__(goal_xz=start_xz, map_center_xz=start_xz, map_size_m=map_size_m, **kw)
        self.instructions, self.k = list(instructions), 0
        self.worker, self.poll_worker = worker, poll_worker
        self.memory = ObjectMemory()
        self.arrive_m, self.arrive_blocked_m = arrive_m, arrive_blocked_m
        self.frontier_every_s, self.verify_looks, self.announce_s = frontier_every_s, verify_looks, announce_s
        self.verify_timeout_s = verify_timeout_s
        self.mode = "explore"  # explore | approach | verify | announce | done | gave_up
        self._frontier = None  # (goal xz, chosen at t)
        self._frontier_t = -np.inf
        self._blacklist: list = []  # frontier spots already reached without resolving them
        self._target: Optional[Remembered] = None
        self._verify = None  # {"since", "looks", "need_after"}
        self._announce_until = -np.inf
        self._depths: dict = {}  # sim time -> (scaled depth, pos, fwd) for lifting boxes from that frame
        self.found: list = []  # (t, label, xz robot, xz object estimate)
        self._new_looks: list = []  # looks delivered this tick (logged for the video)
        self.events: list = []  # (t, kind, label)
        self.frontier_cells = np.zeros((0, 2), np.float32)
        # Search prior painted from the VLM's left/centre/right answers: 1 = no opinion, up to 3 = "go this way";
        # a frontier's cost is discounted by prior_weight_m * (value - 1).
        self.prior_weight_m = prior_weight_m
        self.value = np.ones_like(self.map.score)
        self._value_n = np.zeros_like(self.map.score)
        self._value_label = None

    def look_labels(self):
        """Objects to ground on each look: the current target first, then the rest (remembered for later legs)."""
        rest = [l for l in dict.fromkeys(self.instructions) if l != self.label]
        return ([self.label] if self.label else []) + rest

    def _paint_prior(self, probs, pos, fwd, r_max: float = 8.0, sectors=(-45.0, -15.0, 15.0, 45.0), cap: float = 4.0):
        m = self.map
        rows, cols = np.mgrid[0:m.n, 0:m.n]
        xz = m.origin[None, None, :] + (np.stack([cols, rows], -1) + 0.5) * m.res
        v = xz - pos
        dist = np.linalg.norm(v, axis=-1)
        right = np.array([-fwd[1], fwd[0]])
        bearing = np.rad2deg(np.arctan2(v @ right, v @ fwd))
        for k in range(3):
            sel = (dist > 0.5) & (dist < r_max) & (bearing >= sectors[k]) & (bearing < sectors[k + 1])
            n = self._value_n[sel]
            self.value[sel] = (self.value[sel] * n + 3.0 * probs[k]) / (n + 1)
            self._value_n[sel] = np.minimum(n + 1, cap)  # recent looks keep a say
    @property
    def label(self):
        return self.instructions[self.k] if self.k < len(self.instructions) else None

    def through_wall(self, a_xz, b_xz) -> bool:
        if not self.map.walls:
            return False
        a, b = np.asarray(a_xz, float), np.asarray(b_xz, float)
        n = max(int(np.linalg.norm(b - a) / (self.map.res / 2)), 2)
        pts = a[None, :] + (b - a)[None, :] * np.linspace(0.05, 0.95, n)[:, None]
        return bool(self.map.virtual[self.map.cell(pts)].any())

    # -- perception -----------------------------------------------------------------------------------
    def _consume(self, t):
        while self.worker.inbox:
            seen, res = self.worker.inbox.pop(0)
            d = res.get("direction")
            self._new_looks.append({"seen": seen, "boxes": res["boxes"], "present": res.get("present", {}),
                                    "direction": None if d is None else [d[0], [float(x) for x in d[1]]],
                                    "text": res.get("text"), "ms": res.get("ms", {})})
            if seen not in self._depths:
                continue
            depth, pos, fwd = self._depths[seen]
            depth = depth * self.scale.value
            if res.get("direction") is not None and res["direction"][0] == self.label:
                if self._value_label != self.label:  # a new target: the old prior says nothing about it
                    self.value[:], self._value_n[:], self._value_label = 1.0, 0.0, self.label
                self._paint_prior(res["direction"][1], pos, fwd)
            for lab, boxes in res["boxes"].items():
                if lab not in self.instructions:
                    continue
                pts = [p for p in (box_to_xz(b, depth, pos, fwd) for b in boxes) if p is not None]
                matched = set()
                for p in pts:
                    matched.add(id(self.memory.add(lab, p, pos, seen)))
                self._count_misses(lab, matched, seen, pos, fwd)
                v = self._verify
                if v is not None and lab == self.label and v["need_after"] is not None and seen >= v["need_after"]:
                    self._verify_result(t, any(np.linalg.norm(p - self._target.xz) < 1.5 for p in pts))
        for k in [k for k in self._depths if k < t - 15.0]:
            del self._depths[k]

    def _count_misses(self, lab, matched, seen, pos, fwd, half_fov_deg: float = 38.0, r=(0.6, 4.0)):
        scan = self.scans.get(seen)
        for o in self.memory.objects:
            if o.label != lab or id(o) in matched or o.rejected:
                continue
            v = o.xz - pos
            dist = float(np.linalg.norm(v))
            if not r[0] < dist < r[1]:
                continue
            right = np.array([-fwd[1], fwd[0]])
            bearing = float(np.arctan2(v @ right, v @ fwd))
            if abs(np.rad2deg(bearing)) > half_fov_deg:
                continue
            if scan is not None:  # occluded by something nearer on that bearing: not a miss
                b, rng = scan[2], scan[3]
                i = int(np.argmin(np.abs(b - bearing)))
                if np.isfinite(rng[i]) and rng[i] < dist - 0.6:
                    continue
            o.misses += 1

    def _verify_result(self, t, seen_again: bool):
        v, o = self._verify, self._target
        v["looks"] += 1
        if seen_again:
            self.found.append((t, self.label, self._pos.tolist(), o.xz.tolist()))
            self.events.append((t, "found", self.label))
            self._verify, self.mode = None, "announce"
            self._announce_until = t + self.announce_s
        elif v["looks"] >= self.verify_looks:
            o.rejected = True
            self.events.append((t, "rejected", self.label))
            self._verify, self._target, self.mode = None, None, "explore"

    # -- goal selection -----------------------------------------------------------------------------------
    def _pick_frontier(self, t, pos):
        cost, blocked = cost_grid(self.map, pos, **{k: v for k, v in self.plan_kw.items() if k != "lookahead_m"})
        fr = frontiers(self.map, blocked)
        cells = [np.stack([r, c], 1) for r, c in fr]
        self.frontier_cells = (np.concatenate(cells)[::2].astype(np.float32) if cells else np.zeros((0, 2), np.float32))
        if not fr:
            return None
        dist = distances_from(self.map, cost, pos)
        best = None
        for rows, cols in fr:
            d = dist[rows, cols]
            i = int(np.argmin(d))
            xz = self.map.world(rows[i], cols[i])
            if not np.isfinite(d[i]) or any(np.linalg.norm(xz - b) < 0.8 for b in self._blacklist):
                continue
            # cheap to reach, big (a doorway beats a speck), and where the VLM's hint points
            score = d[i] - 0.15 * min(rows.size, 40) - self.prior_weight_m * (float(self.value[rows[i], cols[i]]) - 1.0)
            if best is None or score < best[0]:
                best = (score, xz)
        if best is None:
            return None
        if self._frontier is not None:  # hysteresis: keep heading for the current frontier unless much worse
            cur = self._frontier[0]
            r, c = self.map.cell(cur)
            near = any(np.min(np.hypot(rows - r, cols - c)) * self.map.res < 0.6 for rows, cols in fr)
            cur_score = dist[r, c] - 0.15 * 20 - self.prior_weight_m * (float(self.value[r, c]) - 1.0)
            if near and cur_score <= best[0] + 3.0:
                return cur
        return best[1]

    def _choose_goal(self, t, pos):
        if self.mode in ("announce",):
            if t >= self._announce_until:
                self.k += 1
                self._target, self._frontier = None, None
                self.mode = "explore" if self.label is not None else "done"
                if self.label is not None:
                    self.events.append((t, "instruction", self.label))
            return
        if self.mode in ("done", "gave_up", "verify"):
            return
        o = self.memory.best(self.label, pos, self.through_wall)
        if o is not None:
            if self._target is not o:
                self.events.append((t, "target_on_map", self.label))
            self._target, self.mode = o, "approach"
            self.goal = o.xz
            return
        if self.mode == "approach":  # the target lost its support (misses / reflection): back to exploring
            self.events.append((t, "target_lost", self.label))
            self._target, self.mode = None, "explore"
        if self._frontier is not None and np.linalg.norm(self._frontier[0] - pos) < 0.5:
            self._blacklist.append(self._frontier[0])
            self._frontier, self._frontier_t = None, -np.inf
        if t - self._frontier_t >= self.frontier_every_s or self._frontier is None:
            f = self._pick_frontier(t, pos)
            self._frontier_t = t
            if f is None:
                if t > self.look_around_s + 2.0:
                    self.mode = "gave_up"
                    self.events.append((t, "gave_up", self.label))
                self._frontier = None
                return
            self._frontier = (np.asarray(f, float), t)
        self.goal = self._frontier[0]

    # -- reflections ------------------------------------------------------------------------------------------
    def add_reflection_wall(self, seen_t: float, min_gap_ratio: float = 1.3, far_m: float = 4.0,
                            extend_m: float = 0.25, hfov_deg: float = 90.0) -> bool:
        """Wall off a confirmed reflection where Qwen boxed the mirror in that view (not across the whole view: a
        note can be about something off to the side). Per mirror box: the scan bearings inside the box seed a
        "phantom" region, grown sideways while the scan stays about as deep; the first shallower hits on either
        side are the real surfaces framing the mirror, and the wall joins them (so it lies roughly on the mirror's
        plane and leaves no gap at its ends). Only if the phantom reads `min_gap_ratio` x deeper than its frame.
        No mirror box, or no gap -> no wall."""
        res = self.worker.results.get(seen_t)
        if res is None or seen_t not in self.scans:
            return False
        pos, fwd, b, r = self.scans[seen_t]
        r = np.where(np.isfinite(r), r, far_m)
        tan_h = np.tan(np.deg2rad(hfov_deg) / 2)
        added = False
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
            ends = []
            for i, edge in ((lo - 1, lo), (hi + 1, hi)):
                k = i if 0 <= i < len(b) else edge  # phantom runs off the image: guess the frame's distance
                ends.append(pos + _rotate(fwd, np.array(b[k])) * (r[k] if k == i else d_edge))
            a, c = ends
            u = (c - a) / max(np.linalg.norm(c - a), 1e-6)
            self.map.add_segment(a - u * extend_m, c + u * extend_m, fwd)
            added = True
        return added

    # -- control ------------------------------------------------------------------------------------------
    def step(self, depth, pose, t, face_xz=None, rgb=None):
        pos = np.asarray(pose["position"], float)[[0, 2]]
        fwd = np.asarray(pose["forward_xz"], float)
        self._pos = pos
        self._depths[t] = (depth, pos, fwd)  # lifted with the floor scale current when its look lands
        background_ms = self.worker.poll(rgb, t) if self.poll_worker and rgb is not None else 0.0
        self._consume(t)
        self._choose_goal(t, pos)
        if self.mode == "approach" and self._target is not None:
            d = float(np.linalg.norm(self._target.xz - pos))
            blocked = self._last is not None and self._last[1] < 0.02 and t > self.look_around_s
            if d < self.arrive_m or (d < self.arrive_blocked_m and blocked):
                self.mode = "verify"
                self._verify = {"since": t, "looks": 0, "need_after": None}
                self.events.append((t, "arrived", self.label))
        if self.mode == "verify" and t - self._verify["since"] > self.verify_timeout_s:
            self._verify["since"] = t  # no usable look in time (e.g. a mirror check held the robot): one failed look
            self._verify_result(t, False)
        if self.mode == "verify" and face_xz is None:
            v_to = self._target.xz - pos
            face_xz = v_to
            if self._verify["need_after"] is None:  # wait for a look that starts once we face it
                ang = np.arccos(np.clip(v_to @ fwd / (np.linalg.norm(v_to) * np.linalg.norm(fwd) + 1e-9), -1, 1))
                if np.rad2deg(ang) < 12.0:
                    self._verify["need_after"] = t
        if self.mode in ("announce", "done", "gave_up") and face_xz is None:
            face_xz = fwd  # hold still
        linear, angular, diag = super().step(depth, pose, t, face_xz=face_xz)
        if self.mode in ("announce", "done", "gave_up", "verify"):
            linear = 0.0
        diag.update({
            "mode": self.mode,
            "target_label": self.label,
            "goal_xz": None if self.goal is None else np.asarray(self.goal, float).tolist(),
            "frontier_xz": None if self._frontier is None else self._frontier[0].tolist(),
            "memory": [{"label": o.label, "xz": o.xz.tolist(), "hits": self.memory.hits(o, self.through_wall),
                        "raw_hits": len(o.sightings), "misses": o.misses, "rejected": o.rejected}
                       for o in self.memory.objects],
            "found": list(self.found),
            "search_events": list(self.events),
            "episode_done": self.mode in ("done", "gave_up"),
            "looks": self._new_looks,
        })
        self._new_looks = []
        if self.poll_worker:
            diag["background_ms"] = background_ms
            diag["vlm_progress"] = self.worker.progress(t)
        return linear, angular, diag
