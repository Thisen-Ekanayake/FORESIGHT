"""Fast-slow hazard handling: a VLM describes the scene in the background (~0.5 Hz), and the Laya
typed-decision model re-decides every control tick from the latest description.

Division of labour, stated plainly because it is easy to oversell:
  - the VLM (Qwen3-VL-2B) is the only component that looks at pixels semantically, and it is slow
    (~2 s per description on an RTX 4060);
  - Laya never sees the image: it turns the latest description into calibrated probabilities over typed
    questions in one ~20-40 ms forward pass;
  - the *action* is a hand-written rule on Laya's output: if P(mirror) crosses `threshold` on a note, the robot
    turns back to that view for a second look, and only if that note is flagged too does its self-built map
    get a wall across the view (see `MapNavigator.add_reflection_wall`), so its planner stops routing
    through the phantom hallway.
Mapping, planning, control law and safety layer are the same `MapNavigator` the depth-only baseline uses.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from foresight.planning.laya_client import LayaClient
from foresight.planning.map_navigator import MapNavigator

MIRROR_QUESTIONS = {
    "mirror": {"type": "choice", "instructions": "Does the scene contain a mirror or glass?",
               "criteria": {"A": "yes, a mirror or glass is present", "B": "no mirror or glass"}},
    "doorway": {"type": "choice", "instructions": "Is there a doorway or opening the robot could pass through?",
                "criteria": {"A": "yes, a doorway or opening is visible", "B": "no doorway or opening"}},
}


@dataclass
class _Job:
    text: str
    started_at: float
    ready_at: float


class BackgroundDescriber:
    """Runs `describe(rgb)` back-to-back as if on a parallel worker: a job started at sim time t on
    the frame seen at t delivers its text at t + (measured latency). Compute happens synchronously
    inside `poll`, which returns those milliseconds so the episode loop can exclude them from the tick.
    """

    def __init__(self, describe: Callable[[np.ndarray], str], min_latency_s: float = 0.0):
        """min_latency_s: deliver no sooner than this (for scripted stand-ins that compute instantly)."""
        self._describe = describe
        self.min_latency_s = min_latency_s
        self._job: Optional[_Job] = None
        self.latest: Optional[str] = None
        self.latest_available_at: Optional[float] = None
        self.latest_seen_at: Optional[float] = None
        self.latencies_s: list = []
        self.jobs: list = []  # (started_at, ready_at) sim times, for timelines

    def poll(self, rgb: np.ndarray, t: float) -> float:
        if self._job is not None and t >= self._job.ready_at:
            self._deliver(self._job)
            self._job = None
        if self._job is None:
            t0 = time.perf_counter()
            text = self._describe(rgb)
            latency = max(time.perf_counter() - t0, self.min_latency_s)
            self.latencies_s.append(latency)
            self._job = _Job(text, t, t + latency)
            self.jobs.append((t, t + latency))
            return latency * 1000
        return 0.0

    def _deliver(self, job: _Job) -> None:
        self.latest = job.text
        self.latest_available_at = job.ready_at
        self.latest_seen_at = job.started_at

    def progress(self, t: float) -> float:
        """Fraction (0-1) of the in-flight description done at sim time t."""
        if self._job is None:
            return 1.0
        span = self._job.ready_at - self._job.started_at
        return float(np.clip((t - self._job.started_at) / span, 0.0, 1.0)) if span > 0 else 1.0


class ForesightNavigator:
    """FORESIGHT's mirror-trap agent: the shared `MapNavigator`, plus Laya deciding every tick whether the
    latest VLM note describes a reflection.

    A 2B VLM also "sees" reflections in glossy artwork and screens, so one flag is not acted on alone: on a
    new note with P(hazard) > threshold the robot stops, turns back to face the view that note was about,
    and waits for the VLM's next look at it. If Laya flags that note too, the navigator draws a wall across
    the view (only if it, or the first flagged view, shows a too-deep gap, see `MapNavigator.add_reflection_wall`)
    and replans;
    otherwise the flag is dismissed. Gives up after `verify_timeout_s`. A flag on a view whose centre already
    runs into a reflection wall is old news and is not re-checked."""

    def __init__(self, laya: LayaClient, describer: BackgroundDescriber, navigator: MapNavigator,
                 questions: dict = MIRROR_QUESTIONS, hazard_question: str = "mirror", hazard_option: str = "A",
                 threshold: float = 0.6, banner_s: float = 3.0, align_deg: float = 8.0, verify_timeout_s: float = 8.0,
                 in_path_only: bool = False):
        """in_path_only: only double-check a flagged view if the robot's current route runs into it (within
        `path_m` ahead and `path_deg` of the view's centre); flags about views off the route are logged as
        "off_path" and cost nothing — a reflection matters only where the robot is about to drive."""
        self.laya, self.describer, self.nav, self.questions = laya, describer, navigator, questions
        self.hazard_question, self.hazard_option = hazard_question, hazard_option
        self.threshold, self.banner_s = threshold, banner_s
        self.align_cos, self.verify_timeout_s = np.cos(np.deg2rad(align_deg)), verify_timeout_s
        self.in_path_only = in_path_only
        self._handled: set = set()
        self._resolved: list = []  # (pos, fwd) of views whose check ended without a new wall: don't re-check them
        self._verify: Optional[dict] = None
        self._wall_at = -np.inf
        self.events: list = []  # (t, kind, note seen_at, p): flag / confirmed / dismissed / timeout / known

    def _route_into(self, seen, path_m: float = 4.0, path_deg: float = 25.0) -> bool:
        pos, fwd = self.nav.scans[seen][:2]
        f = np.asarray(fwd, float) / np.linalg.norm(fwd)
        for p in self.nav.last_path:
            v = np.asarray(p, float) - pos
            d = float(np.linalg.norm(v))
            if 0.3 < d < path_m and v @ f > d * np.cos(np.deg2rad(path_deg)):
                return True
        return False

    def _was_resolved(self, seen, near_m: float = 1.0, within_deg: float = 30.0) -> bool:
        if seen not in self.nav.scans:
            return False
        pos, fwd = self.nav.scans[seen][:2]
        for p, f in self._resolved:
            cos = f @ fwd / (np.linalg.norm(f) * np.linalg.norm(fwd) + 1e-9)
            if np.linalg.norm(p - pos) < near_m and cos > np.cos(np.deg2rad(within_deg)):
                return True
        return False

    def step(self, rgb: np.ndarray, depth: np.ndarray, goal_bearing_rad: float, t: float, pose: dict):
        background_ms = self.describer.poll(rgb, t)
        desc = self.describer.latest
        seen = self.describer.latest_seen_at
        state = {
            "scene_description": desc or "No scene description yet.",
            "description_age_s": round(t - seen, 1) if desc else None,
            "goal_direction_deg": round(float(np.rad2deg(goal_bearing_rad)), 1),
        }
        decision = self.laya.predict(state, self.questions)
        probs = {q: a.get("probabilities", {}) for q, a in decision.answers.items()}
        p_hazard = float(probs[self.hazard_question].get(self.hazard_option, 0.0)) if desc else 0.0
        new_note = seen is not None and seen not in self._handled
        if new_note:
            self._handled.add(seen)
        flagged = p_hazard > self.threshold
        event, wall_added = None, False
        v = self._verify
        if v is None:
            if new_note and flagged and (self.nav.view_walled(seen) or self._was_resolved(seen)):
                event = "known"  # already walled off, or already checked from about here: nothing new to check
            elif new_note and flagged and seen in self.nav.scans and self.in_path_only and not self._route_into(seen):
                event = "off_path"
            elif new_note and flagged and seen in self.nav.scans:
                self._verify = {"seen": seen, "fwd": self.nav.scans[seen][1], "since": t, "aligned_at": None}
                event = "flag"
        elif new_note and v["aligned_at"] is not None and seen >= v["aligned_at"]:  # the second look
            if flagged:
                # The confirming view can be too close to show a gap (a mirror filling the whole view reads deep
                # everywhere); the first flagged view of the same reflection, from further back, then decides.
                wall_added = self.nav.add_reflection_wall(seen) or self.nav.add_reflection_wall(v["seen"])
                if wall_added:
                    self._wall_at = t
                event = "confirmed"
            else:
                event = "dismissed"
            if not wall_added and seen in self.nav.scans:
                self._resolved.append(self.nav.scans[seen][:2])
            self._verify = None
        elif t - v["since"] > self.verify_timeout_s:
            event, self._verify = "timeout", None
        if event:
            self.events.append((t, event, seen, p_hazard))
        v = self._verify
        linear, angular, nav = self.nav.step(depth, pose, t, face_xz=None if v is None else v["fwd"])
        if v is not None and v["aligned_at"] is None:
            f = np.asarray(pose["forward_xz"], float)
            if f @ v["fwd"] / (np.linalg.norm(f) * np.linalg.norm(v["fwd"])) > self.align_cos:
                v["aligned_at"] = t  # a look starting from this tick on sees the flagged view again
        return linear, angular, {
            "background_ms": background_ms,
            "laya_infer_ms": decision.infer_ms,
            "laya_roundtrip_ms": decision.roundtrip_ms,
            "p_hazard": p_hazard,
            "probabilities": probs,
            "hazard": t - self._wall_at < self.banner_s,
            "verifying": v is not None,
            "verify_aligned": v is not None and v["aligned_at"] is not None,
            "event": event,
            "wall_added": wall_added,
            "description": desc,
            "description_seen_at": seen,
            "vlm_progress": self.describer.progress(t),
            **nav,
        }
