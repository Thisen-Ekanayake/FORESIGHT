#!/usr/bin/env python
"""FORESIGHT · Through the Glass: narrated 1920x1920 video of the live G1 runs (foresight.viz.glass_demo_video).

  python tools/make_glass_demo_video.py --out results/isaac/foresight_through_the_glass.mp4 [--preview 42.0]

Needs the pass-2 shots (tools/isaac/replay_render.py) under results/isaac/film/{home,office}/ and the live runs under
results/isaac/live/. Voice: Kokoro af_heart (local); music: a CC BY 4.0 track from data/raw/music/.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from foresight.viz.glass_demo_video import FPS, Film, mix_music  # noqa: E402
from foresight.viz.mirror_trap_video import Voice  # noqa: E402
from foresight.viz.showcase_video import Writer  # noqa: E402

LINES = {
    "open": "To a robot with one camera, glass and mirrors are the hardest things in a room.",
    "office1": "FORESIGHT navigates with a single ordinary camera. It estimates depth from every frame, builds its own "
               "map as it walks, and a small vision model watches for glass and mirrors.",
    "office2": "In NVIDIA's office, the glass shows its blinds and reflections. Depth reads it as a wall, and the robot "
               "walks round the meeting room to its goal.",
    "pov": "In a real, 3D-scanned home, a mirror is harder. Depth from pixels reads the reflection as open floor.",
    "duel1": "Two live runs, the same start, the same goal. The depth-only robot trusts what it sees,",
    "duel2": "and walks straight into the mirror.",
    "check": "FORESIGHT's vision model reads the reflection. The robot takes a second look, and walls the mirror off "
             "on its own map.",
    "exit": "Then it takes the real way round, out of the room and down the hall, and reaches the goal behind the "
            "mirror. No collisions.",
    "end": "FORESIGHT. One camera. No map.",
}


_W = {}


def _init(rate, music_title):
    _W["film"] = Film(ROOT, rate=rate, music_title=music_title)


def _frame(job):
    """One composed frame (worker process): segment `name` of n frames, frame k -> raw RGB bytes."""
    import foresight.viz.glass_demo_video as G
    name, n, k = job
    G.PREVIEW["k"] = k
    gen = getattr(_W["film"], "seg_" + name)
    return next(gen(n, [])).convert("RGB").tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/isaac/foresight_through_the_glass.mp4")
    ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--rate", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--music", default="data/raw/music/Lightless Dawn.mp3")
    ap.add_argument("--tts-python", default="/ml/HelaBERT_Analysis/.venv/bin/python")
    ap.add_argument("--kokoro-home", default="/ml/Models/kokoro/hf")
    ap.add_argument("--voice", default="af_heart")
    a = ap.parse_args()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    film = Film(ROOT, rate=a.rate, music_title=Path(a.music).stem)
    vo = Voice(LINES, ROOT / "results/isaac/film/narration", a.tts_python, a.kokoro_home, a.voice)
    dur = vo.dur
    L = lambda name: film.sh[name].length / a.rate  # noqa: E731
    # (name, generator, seconds, [(line, start s)])
    office_len = L("o_orbit") + L("o_walk")
    segs = [
        ("open", film.seg_open, max(L("o_pov"), dur["open"] + 1.4), [("open", 0.8)]),
        ("office", film.seg_office, max(office_len, dur["office1"] + dur["office2"] + 2.5),
         [("office1", 0.6), ("office2", max(L("o_orbit") + 0.4, 0.6 + dur["office1"] + 0.6))]),
        ("pov", film.seg_pov, max(L("h_pov_ghost"), dur["pov"] + 1.2), [("pov", 0.5)]),
        ("duel", film.seg_duel, max(L("h_duel"), dur["duel1"] + dur["duel2"] + 1.5),
         [("duel1", 0.4), ("duel2", max(0.4 + dur["duel1"] + 0.3, (28.8 - 9.0) / 1.6 / a.rate - dur["duel2"] * 0.6))]),
        ("check", film.seg_check, max(L("h_check"), dur["check"] + 1.2), [("check", 0.6)]),
        ("exit", film.seg_exit, max(L("h_exit"), dur["exit"] + 1.2), [("exit", 0.8)]),
        ("end", film.seg_end, max(7.0, dur["end"] + 3.0), [("end", 0.6)]),
    ]
    t = 0.0
    for name, _, sec, cues in segs:
        print(f"  {t:6.1f}  {name:7s} {sec:5.1f} s  " + " ".join(f"{c}@{s:.1f}" for c, s in cues))
        t += sec
    print(f"  total {t:.1f} s")
    if a.preview is not None:
        t0 = 0.0
        for name, gen, sec, cues in segs:
            if a.preview < t0 + sec:
                import foresight.viz.glass_demo_video as G
                n = int(round(sec * FPS))
                G.PREVIEW["k"] = int((a.preview - t0) * FPS)
                for im in gen(n, cues):
                    p = out.with_name(f"preview_{a.preview:06.1f}.jpg")
                    im.convert("RGB").save(p, quality=92)
                    print("preview", name, "->", p)
                    return
            t0 += sec
        return
    silent = out.with_name(out.stem + ".silent.mp4")
    w = Writer(silent, fade=10)
    spans = []
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.rate, Path(a.music).stem)) as pool:
        for name, gen, sec, cues in segs:
            n = int(round(sec * FPS))
            t0 = w.n / FPS
            for line, st in cues:
                vo.cues.append((t0 + st, line))
                spans.append((t0 + st, t0 + st + dur[line]))
            frames = (Image.frombytes("RGB", (1920, 1920), b) for b in
                      pool.imap(_frame, [(name, n, k) for k in range(n)], chunksize=4))
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
    silent.unlink()
    vwav.unlink()
    mwav.unlink()
    print(f"wrote {out} ({total:.1f} s)")


if __name__ == "__main__":
    main()
