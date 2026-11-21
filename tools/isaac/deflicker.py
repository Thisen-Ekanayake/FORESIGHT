"""Temporal deflicker for path-traced frame sequences.

The RTX path tracer occasionally lifts a whole frame (brighter, hazier) for one to
a few frames and then settles back. This removes those global flashes without
touching per-pixel content: each frame's per-channel tone curve (64 quantiles)
is matched to a running median (then Gaussian) of the neighbouring frames' curves. Slow, real
lighting changes from the camera moving survive; short flashes, including their
lifted blacks, are pulled back to their neighbours.

    python tools/isaac/deflicker.py IN_DIR OUT_DIR [--win 31] [--sigma 3] [--gain 1.0]

IN_DIR holds %05d.jpg frames; OUT_DIR gets the corrected frames under the same names.
--gain multiplies the target curve (e.g. 1.15 for a slight overall lift).
"""
import argparse, glob, os
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter1d

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--win", type=int, default=31, help="running-median window, frames (odd)")
ap.add_argument("--sigma", type=float, default=3.0, help="Gaussian smoothing after the median, frames")
ap.add_argument("--gain", type=float, default=1.0, help="scale on the target tone curve")
ap.add_argument("--quality", type=int, default=95)
a = ap.parse_args()

fs = sorted(glob.glob(os.path.join(a.inp, "*.jpg")))
os.makedirs(a.out, exist_ok=True)
Q = np.linspace(0, 100, 65)
st = []
for f in fs:
    x = np.asarray(Image.open(f).reduce(4), dtype=np.float32).reshape(-1, 3)
    st.append(np.percentile(x, Q, axis=0).T)        # (3, 65) tone curve per channel
st = np.array(st)                                   # (n, 3, 65)
tgt = gaussian_filter1d(median_filter(st, size=(a.win, 1, 1), mode="nearest"), a.sigma, axis=0, mode="nearest")
tgt = np.maximum.accumulate(np.clip(tgt * a.gain, 0, 255), axis=2)
for f, s_, t in zip(fs, st, tgt):
    x = np.asarray(Image.open(f), dtype=np.float32)
    y = np.empty_like(x)
    for c in range(3):
        src = s_[c] + np.arange(65) * 1e-3          # strictly increasing for interp
        y[..., c] = np.interp(x[..., c], src, t[c])
    Image.fromarray(np.clip(y + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(a.out, os.path.basename(f)), quality=a.quality)
print(f"{len(fs)} frames -> {a.out}; max |median shift| {np.abs(tgt[:, :, 32] - st[:, :, 32]).max():.1f}")
