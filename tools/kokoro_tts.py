#!/usr/bin/env python
"""Voice-over lines with Kokoro-82M (hexgrad/Kokoro-82M), for the demo videos.

Kokoro isn't installed in the habitat env (Python 3.9), so this runs in an interpreter that has it, e.g.
  HF_HOME=/ml/Models/kokoro/hf HF_HUB_OFFLINE=1 /ml/HelaBERT_Analysis/.venv/bin/python tools/kokoro_tts.py \
      lines.json out_dir
lines.json: [{"id": ..., "text": ...}, ...]. Writes out_dir/<id>.wav (24 kHz mono) and out_dir/durations.json.
Names the English G2P gets wrong are given explicit phonemes (misaki's [word](/phonemes/) syntax).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000
PRONOUNCE = {
    "Qwen": "kwˈɛn",  # otherwise read letter by letter, "Q-wen"
    "Laya": "lˈɑjə",
}


def _phonemize_names(text: str) -> str:
    for word, ph in PRONOUNCE.items():
        text = re.sub(rf"\b{word}\b", f"[{word}](/{ph}/)", text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lines")
    ap.add_argument("out_dir")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    durations = {}
    for line in json.loads(Path(args.lines).read_text()):
        chunks = [a.numpy() for _, _, a in pipe(_phonemize_names(line["text"]), voice=args.voice, speed=args.speed)]
        audio = np.concatenate(chunks).astype(np.float32)
        sf.write(out / f"{line['id']}.wav", audio, SR)
        durations[line["id"]] = len(audio) / SR
    (out / "durations.json").write_text(json.dumps(durations, indent=1))


if __name__ == "__main__":
    main()
