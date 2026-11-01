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
                 align_deg: float = 8.0, banner_s: float = 3.0, same_m: float = 0.8):
        self.laya, self.worker, self.nav, self.questions = laya, worker, navigator, questions
        self.hazard_question, self.hazard_option, self.threshold = hazard_question, hazard_option, threshold
        self.path_m, self.ray_m, self.keep_s, self.verify_timeout_s = path_m, ray_m, keep_s, verify_timeout_s
        self.align_cos, self.banner_s, self.same_m = np.cos(np.deg2rad(align_deg)), banner_s, same_m
        self._done_looks: set = set()
        self.candidates: list = []  # {"a", "b", "fwd", "seen", "t"} candidate walls not yet checked
        self._dismissed: list = []  # midpoints of candidates the slow check rejected
        self._verify: Optional[dict] = None
        self._wall_at = -np.inf
        self._p, self._note, self._note_seen = 0.0, None, None
        self._infer_ms = self._roundtrip_ms = 0.0
        self.events: list = []  # (t, kind, seen, p): suspect / confirmed / dismissed / timeout
        self.n_slow_looks = 0

    def want_note(self) -> bool:
        """Asked by the worker as a look starts: describe this view? Only once facing a suspected reflection."""
        v = self._verify
        want = v is not None and v["aligned_at"] is not None
        if want:
            self.n_slow_looks += 1
        return want

    # -- geometry -----------------------------------------------------------------------------------------------
    def _known(self, mid) -> bool:
        for a, b in self.nav.map.walls:
            if np.linalg.norm((np.asarray(a) + np.asarray(b)) / 2 - mid) < self.same_m:
                return True
        if any(np.linalg.norm(m - mid) < self.same_m for m in self._dismissed):
            return True
        return any(np.linalg.norm((c["a"] + c["b"]) / 2 - mid) < self.same_m for c in self.candidates)

    def _in_way(self, c, pos, fwd) -> bool:
        a, b = c["a"], c["b"]
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
                    self._dismissed += [(c["a"] + c["b"]) / 2 for c in v["cands"]]
                    event = "dismissed"
                self.events.append((t, event, seen, self._p))
                self._verify = None
                continue
            for a, b, f in self.nav.reflection_segments(seen):  # fast path: a candidate wall from this look
                if not self._known((a + b) / 2):
                    self.candidates.append({"a": a, "b": b, "fwd": f, "seen": seen, "t": t})
        self.candidates = [c for c in self.candidates if t - c["t"] < self.keep_s]
        if self._verify is None and event is None:  # (one event per tick: a fresh check starts next tick)
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
            self._dismissed += [(c["a"] + c["b"]) / 2 for c in self._verify["cands"]]
            event, self._verify = "timeout", None
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
