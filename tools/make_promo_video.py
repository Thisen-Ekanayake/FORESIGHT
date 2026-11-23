#!/usr/bin/env python
"""FORESIGHT promo, "One Camera, No Map": the narrated 1920x1920 video (foresight.viz.promo_video).

  python tools/make_promo_video.py --preview 40.0          # one composed frame at 40 s -> results/isaac/promo/preview_*.jpg
  python tools/make_promo_video.py --scale 0.5             # 960x960 preview cut (the plan's sign-off step)
  python tools/make_promo_video.py                         # the 1920x1920 master -> results/isaac/promo/foresight_promo.mp4

Needs the live runs (tools/isaac/promo/live_runs.sh) and the path-traced shots (tools/isaac/promo/render.sh full);
a segment whose shots aren't rendered yet shows a placeholder card, so previews work part-way. Voice: Kokoro af_heart
(local, cached per line); music: Kevin MacLeod, "Ascending the Vale" (CC BY 4.0, data/raw/music/; not used in
earlier FORESIGHT videos), ducked under the voice.
"""
import argparse
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

import foresight.viz.glass_demo_video as G  # noqa: E402
from foresight.viz.glass_demo_video import FPS, S, mix_music  # noqa: E402
from foresight.viz.mirror_trap_video import Voice  # noqa: E402
from foresight.viz.promo_video import Film  # noqa: E402
from foresight.viz.showcase_video import Writer  # noqa: E402

# The plan's narration (draft 2). Numbers: docs/results/depth_error_A.md (5,600 runs; DA 79% vs GT 83% with floor
# calibration; GT + 10% noise 5%).
LINES = {
    "hook": "To one camera, this mirror is a doorway. There's a whole room back there. It just isn't real.",
    "sys1": "FORESIGHT navigates with a single camera. No LiDAR, no map.",
    "sys2": "A depth model reads every frame and builds the map as the robot walks. A small vision-language model on "
            "board looks every couple of seconds and tells it what it's seeing.",
    "depth1": "We tested how depth error turns into navigation error, over five thousand runs. The surprise: the size "
              "of the error doesn't predict a crash. Its type does.",
    "depth2": "A depth map that's two times off in scale is harmless once you calibrate it on the floor. Ten percent "
              "of random noise, a far smaller error, stops the robot cold.",
    "mirror1": "The camera can't tell a mirror from a door. The language model can.",
    "mirror2": "FORESIGHT stops, takes a second look, marks the mirror as a wall, and finds the real way round.",
    "search": "Give it a goal in plain words, and it searches the home on its own. No coordinates, nothing mapped in "
              "advance.",
    "end": "FORESIGHT. One camera. No map. On board.",
}

_W = {}


def _init(music_title):
    _W["film"] = Film(ROOT, music_title=music_title)


def _frame(job):
    """One composed frame (worker process): segment `name` of n frames, frame k -> raw RGB bytes."""
    name, n, k, size = job
    G.PREVIEW["k"] = k
    im = next(getattr(_W["film"], "seg_" + name)(n, [])).convert("RGB")
    return (im.resize((size, size), Image.LANCZOS) if size != S else im).tobytes()


def plan(film, dur):
    """(segment, seconds, [(line, start s)]): each segment lasts as long as its shots or its narration needs."""
    L = lambda n: film.sh[n].length if n in film.sh else 0.0  # noqa: E731
    hook = max(12.0, dur["hook"] + 2.0)
    system = max(L("sys_front") + L("sys_shoulder"), dur["sys1"] + dur["sys2"] + 2.5, 20.0)
    depth = max(L("depth_crane") + 7.0 + L("depth_goal") + L("depth_noise_stall"),
                dur["depth1"] + dur["depth2"] + 3.0, 36.0)
    mirror = max(L("mirror_duel") + L("mirror_check") + L("mirror_exit"), dur["mirror1"] + dur["mirror2"] + 3.0, 26.0)
    search = max(min(L("search_lapse"), 24.0), dur["search"] + 6.5, 20.0)
    end = max(L("end_pull") + 5.0, dur["end"] + 4.0)
    sys2_at = max(0.6 + dur["sys1"] + 0.6, 5.0)
    depth2_at = max(L("depth_crane") + 7.0 + 0.4, 0.6 + dur["depth1"] + 0.8)
    mirror2_at = max(L("mirror_duel") + 0.6, 0.6 + dur["mirror1"] + 0.6)
    return [("hook", hook, [("hook", 0.8)]),
            ("system", system, [("sys1", 0.6), ("sys2", sys2_at)]),
            ("depth", depth, [("depth1", 0.6), ("depth2", depth2_at)]),
            ("mirror", mirror, [("mirror1", 0.6), ("mirror2", mirror2_at)]),
            ("search", search, [("search", 6.0)]),  # after the on-screen instruction
            ("end", end, [("end", L("end_pull") + 1.2)])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/isaac/promo/foresight_promo.mp4")
    ap.add_argument("--preview", type=float, default=None, help="write one frame at this video time and stop")
    ap.add_argument("--scale", type=float, default=1.0, help="0.5 = the 960x960 preview cut")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--music", default="data/raw/music/Ascending the Vale.mp3")
    ap.add_argument("--tts-python", default="/ml/HelaBERT_Analysis/.venv/bin/python")
    ap.add_argument("--kokoro-home", default="/ml/Models/kokoro/hf")
    ap.add_argument("--voice", default="af_heart")
    a = ap.parse_args()
    out = ROOT / a.out
    if a.scale != 1.0:
        out = out.with_name(out.stem + f"_{int(S * a.scale)}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    film = Film(ROOT, music_title=Path(a.music).stem)
    vo = Voice(LINES, ROOT / "results/isaac/promo/narration", a.tts_python, a.kokoro_home, a.voice)
    segs = plan(film, vo.dur)
    t = 0.0
    for name, sec, cues in segs:
        print(f"  {t:6.1f}  {name:7s} {sec:5.1f} s  " + " ".join(f"{c}@{s:.1f}" for c, s in cues))
        t += sec
    print(f"  total {t:.1f} s (plan: 2:00-2:30)")
    if a.preview is not None:
        t0 = 0.0
        for name, sec, cues in segs:
            if a.preview < t0 + sec:
                n = int(round(sec * FPS))
                G.PREVIEW["k"] = int((a.preview - t0) * FPS)
                im = next(getattr(film, "seg_" + name)(n, cues)).convert("RGB")
                p = out.with_name(f"preview_{a.preview:06.1f}.jpg")
                im.save(p, quality=92)
                print("preview", name, "->", p)
                return
            t0 += sec
        return
    size = int(S * a.scale)
    silent = out.with_name(out.stem + ".silent.mp4")
    w = Writer(silent, fade=10)
    if size != S:  # the preview cut: ffmpeg scales the full-size stream on the way out
        w.proc.stdin.close()
        w.proc.wait()
        w.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(silent)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        w.bg = w.bg.resize((size, size))
    spans = []
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(Path(a.music).stem,)) as pool:
        for name, sec, cues in segs:
            n = int(round(sec * FPS))
            t0 = w.n / FPS
            for line, st in cues:
                vo.cues.append((t0 + st, line))
                spans.append((t0 + st, t0 + st + vo.dur[line]))
            frames = (Image.frombytes("RGB", (size, size), b) for b in
                      pool.imap(_frame, [(name, n, k, size) for k in range(n)], chunksize=4))
            w.segment(frames, n)
            print(f"  {name}: {n / FPS:.1f} s (total {w.n / FPS:.1f} s)", flush=True)
    w.close()
    total = w.n / FPS
    vwav = out.with_name(out.stem + ".voice.wav")
    vo.write_mix(vwav, total)
    mwav = out.with_name(out.stem + ".mix.wav")
    mix_music(vwav, ROOT / a.music, mwav, total, spans)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(mwav), "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out)], check=True)
    for f in (silent, vwav, mwav):
        f.unlink()
    print(f"wrote {out} ({total:.1f} s)")


if __name__ == "__main__":
    main()
