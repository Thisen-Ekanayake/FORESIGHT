"""Fast/slow reflection handling for object search: a cheap check on every look, the slow VLM note + Laya only
when it matters.

FORESIGHT v1 (`laya_fusion.ForesightNavigator`) put a ~2 s Qwen3-VL scene description on *every* look, so its
looks took ~2.6 s against the baseline's ~0.34 s: it crashed less (1 vs 5 runs at a mirror) but found objects
later and timed out more (7 vs 2 of 15, results/runs/object_search/eval, 2026-10-30). Here:

  fast (every look, ~0.1-0.4 s)  Qwen3-VL yes/no "is there a mirror?" (one forward pass); only if yes, ground
                                 the mirror's box. The box plus the depth scan of that frame give a *candidate*
                                 wall (`ObjectSearchNavigator.reflection_segments`: the boxed region reads >= 1.3x
                                 deeper than the real surfaces framing it, i.e. it looks like a phantom room).
  in the way?                    A candidate matters only if the planned route crosses it within `path_m`, or
                                 the robot's heading does within `ray_m`. Others wait (`keep_s`) in case the
                                 route turns into them, and cost nothing.
  slow (only then)               The robot stops forward motion, faces the view, and its next look adds the
                                 reflection-aware scene description; Laya reads it. P(mirror) > threshold -> the
                                 candidate becomes a wall on the map (the planner routes around it, objects seen
                                 through it are ignored); otherwise it is dismissed and not raised again.

So the two independent signals are the same as v1's double check (a second opinion before a permanent wall), but
the expensive one is paid only for reflections in the robot's way.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from foresight.planning.laya_client import LayaClient
from foresight.planning.laya_fusion import MIRROR_QUESTIONS


def _seg_dist(p, a, b) -> float:
    """Distance from point p to segment ab (2D)."""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    u = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(a + u * ab - p))


def _cross(a, b, c, d) -> bool:
    """Whether segments ab and cd intersect (2D)."""
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return orient(a, b, c) != orient(a, b, d) and orient(c, d, a) != orient(c, d, b)


class FastSlowMirrorGuard:
    """Wraps an `ObjectSearchNavigator` (built with poll_worker=False) whose `PerceptionWorker` grounds "mirror"
    on every look and was given `describe` + `describe_when=guard.want_note`. Same `step` signature as
    `ForesightNavigator`, and the same diagnostics keys, so runners and videos treat them alike."""

    def __init__(self, laya: LayaClient, worker, navigator, questions: dict = MIRROR_QUESTIONS,
                 hazard_question: str = "mirror", hazard_option: str = "A", threshold: float = 0.6,
                 path_m: float = 5.0, ray_m: float = 2.5, keep_s: float = 15.0, verify_timeout_s: float = 14.0,
                 align_deg: float = 8.0, banner_s: float = 3.0, same_m: float = 0.4,
                 wedge_p: float = 0.9, wedge_m: float = 3.0, wedge_pad_deg: float = 3.0,
                 near_m: Optional[float] = None, walled_m: Optional[float] = None, gap_s: float = 0.0):
        """same_m: a candidate is already handled if it lies along an existing wall or a dismissed candidate (both ends
        and its middle within same_m of it). wedge_*: a look with P(mirror) >= wedge_p also makes a "wedge" candidate
        from each mirror box: the box's bearing span from that pose, out to wedge_m; it is in the way if the route
        or heading enters it. On confirmation, a wedge without gap geometry is walled across its span at the nearest
        depth read inside it (clamped 0.6-2.0 m).

        v2.1 (2026-11-01), after the fresh-set evaluation of v2 (commit b38b4ed), whose two mirror crashes were both at
        a 00808 closet mirror: (1) v2 called a candidate "known" if its midpoint was within 0.8 m of any wall's
        midpoint, so an earlier wall at the wrong angle hid the real candidate; (2) Qwen boxed objects *inside* the
        reflection (a TV, pictures), the region read about as near as its frame, so no gap-based candidate was made,
        and the robot turned into the boxed mirror. Both found by replaying those looks offline.

        v2.2 (2026-11-10), against over-checking: on the point-goal pilot (dev set A_3001) v2.1 ran 4-18 slow checks per
        episode, confirmed 1-6 walls, and succeeded 33% against 63% for the same navigator without it (p = 0.006);
        object-search set 3 showed the same in 00808. Off by default (v2.1 behaviour):
          near_m    a candidate is in the way only where the route or heading meets it within near_m of the robot
                    (v2.1: 5 m along the route, 2.5 m along the heading, wedges out to 3 m), so a mirror across the room
                    costs nothing until the robot is about to reach it
          walled_m  a candidate whose midpoint is within walled_m of an existing wall is already handled, whatever its
                    angle (v2.1 needed it to lie *along* the wall, so a mirror walled from one side was re-checked from
                    the next)
          gap_s     at least gap_s between the end of one slow check and the start of the next"""
        self.laya, self.worker, self.nav, self.questions = laya, worker, navigator, questions
        self.hazard_question, self.hazard_option, self.threshold = hazard_question, hazard_option, threshold
        self.path_m, self.ray_m, self.keep_s, self.verify_timeout_s = path_m, ray_m, keep_s, verify_timeout_s
        self.align_cos, self.banner_s, self.same_m = np.cos(np.deg2rad(align_deg)), banner_s, same_m
        self.wedge_p, self.wedge_m, self.wedge_pad = wedge_p, wedge_m, np.deg2rad(wedge_pad_deg)
        self._done_looks: set = set()
        self.candidates: list = []  # {"a", "b", "fwd", "seen", "t"} candidate walls not yet checked
        self._dismissed: list = []  # (a, b) of candidates the slow check rejected
        self._verify: Optional[dict] = None
        self._wall_at = -np.inf
        self._p, self._note, self._note_seen = 0.0, None, None
        self._infer_ms = self._roundtrip_ms = 0.0
        self.events: list = []  # (t, kind, seen, p): suspect / confirmed / dismissed / timeout
        self.n_slow_looks = 0
        self.near_m, self.walled_m, self.gap_s = near_m, walled_m, gap_s
        if near_m is not None:
            self.path_m, self.ray_m, self.wedge_m = min(path_m, near_m), min(ray_m, near_m), min(wedge_m, near_m)
        self._last_check_end = -np.inf

    def want_note(self) -> bool:
        """Asked by the worker as a look starts: describe this view? Only once facing a suspected reflection."""
        v = self._verify
        want = v is not None and v["aligned_at"] is not None
        if want:
            self.n_slow_looks += 1
        return want

    # -- geometry -----------------------------------------------------------------------------------------------
    def _known(self, a, b) -> bool:
        """Lies along an existing wall, a dismissed candidate, or a pending candidate (v2.2: or near any wall)."""
        mid = (a + b) / 2
        if self.walled_m is not None and any(_seg_dist(mid, np.asarray(w0), np.asarray(w1)) < self.walled_m
                                             for w0, w1 in self.nav.map.walls):
            return True
        segs = [(np.asarray(w0), np.asarray(w1)) for w0, w1 in self.nav.map.walls] + list(self._dismissed) \
            + [(c["a"], c["b"]) for c in self.candidates]
        return any(max(_seg_dist(q, s0, s1) for q in (a, b, mid)) < self.same_m for s0, s1 in segs)

    def _wedge(self, seen, box, p_mirror):
        """A wedge candidate from one mirror box (or None): bearing span, origin pose, and its fallback wall."""
        if p_mirror < self.wedge_p or seen not in self.nav.scans:
            return None
        pos, fwd, b, r = self.nav.scans[seen]
        a1, a2 = (float(np.arctan((box[k] / 1000 - 0.5) * 2)) for k in (0, 2))  # 90 deg hfov: tan(45) = 1
        inside = (b >= a1) & (b <= a2) & np.isfinite(r)
        d = float(np.clip(np.min(r[inside]) if inside.any() else 1.0, 0.6, 2.0))
        from foresight.planning.local_map import _rotate
        ends = [pos + _rotate(fwd, np.array(x)) * d for x in (a1, a2)]
        u = (ends[1] - ends[0]) / max(np.linalg.norm(ends[1] - ends[0]), 1e-6)
        return {"kind": "wedge", "pos": pos, "fwd": fwd, "b1": a1 - self.wedge_pad, "b2": a2 + self.wedge_pad,
                "a": ends[0] - u * 0.2, "b": ends[1] + u * 0.2, "seen": seen}

    def _in_wedge(self, c, q) -> bool:
        v = np.asarray(q, float) - c["pos"]
        dist = float(np.linalg.norm(v))
        if not 0.3 < dist < self.wedge_m:
            return False
        f = c["fwd"] / np.linalg.norm(c["fwd"])
        bearing = float(np.arctan2(v @ np.array([-f[1], f[0]]), v @ f))
        return c["b1"] <= bearing <= c["b2"]

    def _in_way(self, c, pos, fwd) -> bool:
        a, b = c["a"], c["b"]
        if c.get("kind") == "wedge":
            ray = [pos + fwd / np.linalg.norm(fwd) * x for x in np.arange(0.2, min(2.0, self.ray_m), 0.2)]
            path, run, pts = [pos] + [np.asarray(p, float) for p in self.nav.last_path], 0.0, []
            for p, q in zip(path[:-1], path[1:]):
                pts += [p + (q - p) * x for x in np.linspace(0, 1, 4)]
                run += float(np.linalg.norm(q - p))
                if run > self.path_m:
                    break
            return any(self._in_wedge(c, q) for q in ray + pts)
        if _cross(pos, pos + fwd / np.linalg.norm(fwd) * self.ray_m, a, b):
            return True
        path = [pos] + [np.asarray(p, float) for p in self.nav.last_path]
        run = 0.0
        for p, q in zip(path[:-1], path[1:]):
            if _cross(p, q, a, b):
                return True
            run += float(np.linalg.norm(q - p))
            if run > self.path_m:
                break
        return False

    # -- control ------------------------------------------------------------------------------------------------
    def _laya(self, text, t, seen, goal_bearing_rad):
        state = {"scene_description": text, "description_age_s": round(t - seen, 1),
                 "goal_direction_deg": round(float(np.rad2deg(goal_bearing_rad)), 1)}
        d = self.laya.predict(state, self.questions)
        self._infer_ms, self._roundtrip_ms = d.infer_ms, d.roundtrip_ms
        probs = {q: a.get("probabilities", {}) for q, a in d.answers.items()}
        return float(probs[self.hazard_question].get(self.hazard_option, 0.0)), probs

    def step(self, rgb: np.ndarray, depth: np.ndarray, goal_bearing_rad: float, t: float, pose: dict):
        background_ms = self.worker.poll(rgb, t)
        pos = np.asarray(pose["position"], float)[[0, 2]]
        fwd = np.asarray(pose["forward_xz"], float)
        event, wall_added, probs = None, False, None
        for seen in sorted(k for k in self.worker.results if k not in self._done_looks):
            self._done_looks.add(seen)
            res = self.worker.results[seen]
            v = self._verify
            if v is not None and res.get("text") and v["aligned_at"] is not None and seen >= v["aligned_at"]:
                self._note, self._note_seen = res["text"], seen  # the slow check's look
                self._p, probs = self._laya(res["text"], t, seen, goal_bearing_rad)
                if self._p > self.threshold:
                    for c in v["cands"]:
                        self.nav.map.add_segment(c["a"], c["b"], c["fwd"])
                    wall_added, self._wall_at, event = True, t, "confirmed"
                else:
                    self._dismissed += [(c["a"], c["b"]) for c in v["cands"]]
                    event = "dismissed"
                self.events.append((t, event, seen, self._p))
                self._verify, self._last_check_end = None, t
                continue
            segs = self.nav.reflection_segments(seen)  # fast path: candidate walls from this look
            for a, b, f in segs:
                if not self._known(a, b):
                    self.candidates.append({"a": a, "b": b, "fwd": f, "seen": seen, "t": t})
            if not segs:  # no gap geometry: a confident mirror box still marks where not to drive
                for box in res.get("boxes", {}).get("mirror", []):
                    c = self._wedge(seen, box, res.get("present", {}).get("mirror", 0.0))
                    if c is not None and not self._known(c["a"], c["b"]):
                        self.candidates.append(dict(c, t=t))
        self.candidates = [c for c in self.candidates if t - c["t"] < self.keep_s]
        if self._verify is None and event is None and t - self._last_check_end >= self.gap_s:
            # (one event per tick: a fresh check starts next tick)
            hits = [c for c in self.candidates if self._in_way(c, pos, fwd)]
            if hits:
                c0 = hits[0]
                self.candidates = [c for c in self.candidates if not any(c is h for h in hits)]
                self._verify = {"seen": c0["seen"], "fwd": self.nav.scans[c0["seen"]][1], "since": t,
                                "aligned_at": None, "cands": hits}
                event = "suspect"
                self.events.append((t, event, c0["seen"], None))
        elif self._verify is not None and t - self._verify["since"] > self.verify_timeout_s:
            # No usable look in time: stop holding the robot and don't raise these candidates again.
            self._dismissed += [(c["a"], c["b"]) for c in self._verify["cands"]]
            event, self._verify, self._last_check_end = "timeout", None, t
            self.events.append((t, event, None, None))
        v = self._verify
        linear, angular, nav = self.nav.step(depth, pose, t, face_xz=None if v is None else v["fwd"])
        if v is not None:
            linear = 0.0  # a suspected reflection in the way vetoes forward motion until checked
            if v["aligned_at"] is None and fwd @ v["fwd"] / (np.linalg.norm(fwd) * np.linalg.norm(v["fwd"])) > self.align_cos:
                v["aligned_at"] = t
        return linear, angular, {
            "background_ms": background_ms,
            "laya_infer_ms": self._infer_ms if probs is not None else 0.0,
            "laya_roundtrip_ms": self._roundtrip_ms if probs is not None else 0.0,
            "laya_called": probs is not None,
            "p_hazard": self._p,
            "probabilities": probs,
            "hazard": t - self._wall_at < self.banner_s,
            "verifying": v is not None,
            "verify_aligned": v is not None and v["aligned_at"] is not None,
            "event": event,
            "wall_added": wall_added,
            "description": self._note,
            "description_seen_at": self._note_seen,
            "candidates": [[c["a"].tolist(), c["b"].tolist()] for c in self.candidates]
                          + ([[c["a"].tolist(), c["b"].tolist()] for c in v["cands"]] if v else []),
            "vlm_progress": self.worker.progress(t),
            **nav,
        }
