"""Depth sources for the closed loop: what the navigator is given as "depth" each tick.

The depth-error-propagation study (deliverable A, PLAN.md) holds the navigation stack fixed and changes only this:

  da                     Depth Anything V2 (the real zero-shot estimator)
  gt                     simulator ground-truth depth: the spec's depth-sensor baseline
  gt:<errors>            ground truth with controlled, injected error (`InjectedDepth`)
  mix:<base>@<mask>      one source everywhere, the other inside a pixel mask (`MaskedSwapDepth`): e.g. ground truth
                         with Depth Anything's own error only on reflective pixels, which isolates how much of the
                         estimator's navigation failure comes from those pixels

Error types (lit review, docs/lit_review/positioning.md): *uniform* errors (scale, bias, noise) that LG-VLN argues
navigation tolerates; NavTrust's sensor corruptions (missing data, quantization) for comparability; and *structured*
errors confined to reflective or textureless pixels, where monocular depth actually fails.

A source is called as `source(rgb, obs)` with the frame the models see and the simulator observation of the same
tick (`obs["depth"]`, and `obs["semantic"]` when the episode renders it). Ground truth never reaches a navigator
except through a source built from it; `run_timed_episode` still logs GT for scoring only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

# HM3D categories that are mirrors or see-through glass (4 annotated minival homes: mirror 19, glass 10, window 81,
# shower door/cabin 4, stained glass 1). Windows are glass panes in a wall; most HM3D window frames are separate
# "window frame" objects and are not included.
REFLECTIVE_CATEGORIES = ("mirror", "glass", "window", "shower door", "shower cabin", "stained glass")


def _resize_nearest(a: np.ndarray, hw) -> np.ndarray:
    if a.shape[:2] == tuple(hw):
        return a
    return np.asarray(Image.fromarray(a).resize((hw[1], hw[0]), Image.NEAREST))


def reflective_ids(sim, categories=REFLECTIVE_CATEGORIES) -> np.ndarray:
    """Semantic instance ids whose category is a reflective/transparent surface."""
    ids = [o.semantic_id for o in sim.semantic_scene.objects
           if o is not None and o.category is not None and o.category.name().lower().strip() in categories]
    return np.asarray(sorted(ids), np.int64)


def texture_energy(rgb: np.ndarray, window: int = 15) -> np.ndarray:
    """Local mean gradient magnitude of the grey image (0-255 units per pixel), the textureless measure of the
    stratified benchmark (deliverable C)."""
    g = rgb.astype(np.float32).mean(axis=2)
    gy, gx = np.gradient(g)
    return uniform_filter(np.hypot(gx, gy), size=window)


def textureless_mask(rgb: np.ndarray, threshold: float = 2.0, window: int = 15) -> np.ndarray:
    return texture_energy(rgb, window) < threshold


class DepthSource:
    name = "source"

    def __call__(self, rgb: np.ndarray, obs: dict) -> np.ndarray:
        raise NotImplementedError

    def reset(self, sim=None, seed: int = 0) -> None:
        """Called at the start of each episode (the simulator is needed for semantic ids)."""


class EstimatedDepth(DepthSource):
    name = "da"

    def __init__(self, estimator):
        self.estimator = estimator

    def __call__(self, rgb, obs):
        return self.estimator.estimate(rgb)


class GroundTruthDepth(DepthSource):
    """Simulator depth at the model's resolution. `latency_ms`: wait this long per frame, so a comparison with the
    estimator changes the error only, not the control rate (sim time advances by measured compute)."""
    name = "gt"

    def __init__(self, latency_ms: float = 0.0):
        self.latency_ms = latency_ms

    def __call__(self, rgb, obs):
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        return _resize_nearest(np.asarray(obs["depth"], np.float32), rgb.shape[:2])


@dataclass
class ErrorSpec:
    """Injected on top of ground truth, in this order. All zero / one = no error.

    scale        global multiplicative scale (monocular metric depth is off by a scene-dependent factor: DA-V2
                 read 1.9-2.5x too far across HM3D homes)
    bias_m       additive offset
    noise        depth-proportional Gaussian noise, sigma = noise * depth, smoothed over `noise_px` pixels (0 = iid)
    holes        fraction of pixels set invalid (0), in blobs of ~`hole_px` (NavTrust "missing data")
    quant_m      depth quantised to this step (NavTrust "quantization")
    reflect      on reflective pixels (semantic mask), depth multiplied by this factor: a mirror read as the
                 room it reflects (> 1) is what DA-V2 did at the 00808 hallway mirror (4.5-7.9 m at 2.5 m)
    flat_tex     on textureless pixels, depth replaced by its local mean over `flat_px` (structure lost)
    """
    scale: float = 1.0
    bias_m: float = 0.0
    noise: float = 0.0
    noise_px: int = 0
    holes: float = 0.0
    hole_px: int = 8
    quant_m: float = 0.0
    reflect: float = 1.0
    flat_tex: bool = False
    flat_px: int = 41

    @classmethod
    def parse(cls, text: str) -> "ErrorSpec":
        """'scale=1.5,noise=0.1' -> ErrorSpec."""
        kw = {}
        for part in filter(None, text.split(",")):
            k, v = part.split("=")
            kw[k] = (v.lower() in ("1", "true", "yes")) if k == "flat_tex" else (int(v) if k.endswith("_px") else float(v))
        return cls(**kw)

    def tag(self) -> str:
        d = {k: v for k, v in self.__dict__.items() if v != getattr(ErrorSpec(), k)}
        return ",".join(f"{k}={v}" for k, v in d.items()) or "none"


class InjectedDepth(GroundTruthDepth):
    """Ground truth plus an `ErrorSpec`. Random parts are seeded per episode (`reset`), so every condition sees the
    same noise on the same episode."""

    def __init__(self, spec: ErrorSpec, latency_ms: float = 0.0, reflective: Optional[np.ndarray] = None):
        super().__init__(latency_ms)
        self.spec = spec
        self.name = f"gt:{spec.tag()}"
        self.reflective = reflective  # semantic ids; set by reset() when the episode renders semantics
        self.rng = np.random.default_rng(0)

    def reset(self, sim=None, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        if sim is not None and self.spec.reflect != 1.0:
            self.reflective = reflective_ids(sim)

    def __call__(self, rgb, obs):
        d = super().__call__(rgb, obs).copy()
        s = self.spec
        valid = d > 0
        if s.reflect != 1.0:
            if "semantic" not in obs or self.reflective is None:
                raise ValueError("reflect= needs a semantic sensor (run the episode with semantic=True)")
            m = np.isin(_resize_nearest(obs["semantic"], d.shape), self.reflective) & valid
            d[m] *= s.reflect
        if s.flat_tex:
            m = textureless_mask(rgb) & valid
            d[m] = uniform_filter(d, size=s.flat_px)[m]
        d[valid] = d[valid] * s.scale + s.bias_m
        if s.noise > 0:
            n = self.rng.standard_normal(d.shape).astype(np.float32)
            if s.noise_px > 1:
                n = uniform_filter(n, size=s.noise_px) * s.noise_px  # keep unit variance after averaging
            d[valid] *= 1.0 + s.noise * n[valid]
        if s.quant_m > 0:
            d[valid] = np.round(d[valid] / s.quant_m) * s.quant_m
        d[valid & (d < 0.05)] = 0.05  # (before holes: a hole is the only invalid pixel an error makes)
        if s.holes > 0:
            h, w = d.shape
            k = max(int(s.hole_px), 1)
            coarse = self.rng.random((h // k + 1, w // k + 1)) < s.holes
            d[np.kron(coarse, np.ones((k, k), bool))[:h, :w]] = 0.0
        return d


class MaskedSwapDepth(DepthSource):
    """`outside` source everywhere, `inside` source on a pixel mask. mask: 'reflective' (semantic) or 'textureless'.

    gt outside, da inside  -> only the estimator's error on those pixels (how much of its failure they explain)
    da outside, gt inside  -> the estimator with those pixels fixed (what a perfect mirror/glass detector would buy)

    align: the inside source is first rescaled to the outside one's global scale (median ratio over the unmasked
    pixels), so the swap moves only the *structured* error on the mask. Without it, pasting ground truth into DA-V2's
    2x-too-far depth left a ring of falsely near pixels at every window, and the robot never reached a goal plain DA
    reached (s-00808-00 smoke test, 2026-11-02)."""

    def __init__(self, outside: DepthSource, inside: DepthSource, mask: str, align: bool = True):
        self.outside, self.inside, self.mask, self.align = outside, inside, mask, align
        self.name = f"mix:{outside.name}@{mask}={inside.name}"
        self.reflective: Optional[np.ndarray] = None

    def reset(self, sim=None, seed: int = 0):
        for s in (self.outside, self.inside):
            s.reset(sim, seed)
        if sim is not None and self.mask == "reflective":
            self.reflective = reflective_ids(sim)

    def __call__(self, rgb, obs):
        a = self.outside(rgb, obs)
        b = self.inside(rgb, obs)
        if self.mask == "reflective":
            m = np.isin(_resize_nearest(obs["semantic"], a.shape), self.reflective)
        elif self.mask == "textureless":
            m = textureless_mask(rgb)
        else:
            raise ValueError(f"unknown mask {self.mask!r}")
        if self.align:
            ok = ~m & (a > 0) & (b > 0)
            if ok.sum() > 100:
                b = b * float(np.median(a[ok] / b[ok]))
        out = a.copy()
        out[m] = b[m]
        return out


def make_source(spec: str, estimator=None, latency_ms: float = 0.0) -> DepthSource:
    """'da' | 'gt' | 'gt:scale=1.5,noise=0.1' | 'mix:gt@reflective=da' | 'mix:da@reflective=gt' (also @textureless)."""
    if spec == "da":
        return EstimatedDepth(estimator)
    if spec == "gt":
        return GroundTruthDepth(latency_ms)
    if spec.startswith("gt:"):
        return InjectedDepth(ErrorSpec.parse(spec[3:]), latency_ms)
    if spec.startswith("mix:"):
        outside, rest = spec[4:].split("@")
        mask, inside = rest.split("=")
        return MaskedSwapDepth(make_source(outside, estimator, latency_ms), make_source(inside, estimator, latency_ms), mask)
    raise ValueError(f"unknown depth source {spec!r}")


def needs_semantic(spec: str) -> bool:
    return "reflect=" in spec or "@reflective" in spec


@dataclass
class DepthErrorStats:
    """Per-frame error of the depth a navigator was given, against ground truth (valid GT 0.1-10 m, stride 4)."""
    rmse: float
    absrel: float
    delta1: float
    rmse_aligned: float  # after median-scale alignment: the structure of the error, with the scale taken out
    scale: float  # median(given / GT)
    extra: dict = field(default_factory=dict)


def depth_error(given: np.ndarray, gt: np.ndarray, stride: int = 4) -> Optional[DepthErrorStats]:
    g = given[::stride, ::stride].astype(np.float32)
    t = _resize_nearest(np.asarray(gt, np.float32), given.shape)[::stride, ::stride]
    ok = (t > 0.1) & (t < 10.0) & np.isfinite(g)
    if ok.sum() < 20:
        return None
    g, t = g[ok], t[ok]
    gv = np.where(g > 0, g, np.nan)  # holes count as maximal error in RMSE (as 0 m), excluded from ratios
    ratio = gv / t
    scale = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else 1.0
    delta = np.maximum(ratio, 1 / ratio)
    aligned = np.where(np.isfinite(gv), gv / max(scale, 1e-6), 0.0)
    return DepthErrorStats(
        rmse=float(np.sqrt(np.mean((g - t) ** 2))),
        absrel=float(np.nanmean(np.abs(gv - t) / t)) if np.isfinite(gv).any() else 1.0,
        delta1=float(np.nanmean(delta < 1.25)) if np.isfinite(delta).any() else 0.0,
        rmse_aligned=float(np.sqrt(np.mean((aligned - t) ** 2))),
        scale=scale,
    )
