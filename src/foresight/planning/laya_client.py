"""Client for the Laya typed-decision model, run out-of-process.

Laya (convaiinnovations/laya: a non-autoregressive ModernBERT decision model returning calibrated
probabilities over typed questions in one forward pass) needs Python >=3.10, while habitat-sim
0.3.3 pins this stack to 3.9. So it runs in its own interpreter (`.venv-laya`) as
`tools/laya_worker.py`, and this client talks to it over stdin/stdout JSON lines.

Timing is reported two ways so the cross-process hop isn't hidden: `infer_ms` is the model's own
forward pass (measured inside the worker, CUDA-synchronised), `roundtrip_ms` is what the caller
actually waits, including JSON encoding and the pipe.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PYTHON = REPO_ROOT / ".venv-laya" / "bin" / "python"
WORKER = REPO_ROOT / "tools" / "laya_worker.py"


@dataclass
class LayaDecision:
    answers: dict
    infer_ms: float
    roundtrip_ms: float


class LayaClient:
    def __init__(self, model: str = "models/laya", device: str = "cuda", python: Path = DEFAULT_PYTHON):
        self._proc = subprocess.Popen(
            [str(python), str(WORKER), "--model", model, "--device", device],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"laya worker failed to start: {ready}")
        self.load_s = ready["load_s"]

    def _read(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("laya worker exited (run it by hand to see its stderr)")
        return json.loads(line)

    def predict(self, state, questions: dict) -> LayaDecision:
        t = time.perf_counter()
        self._proc.stdin.write(json.dumps({"state": state, "questions": questions}) + "\n")
        self._proc.stdin.flush()
        reply = self._read()
        roundtrip_ms = (time.perf_counter() - t) * 1000
        if "error" in reply:
            raise RuntimeError(f"laya worker: {reply['error']}")
        return LayaDecision(reply["answers"], reply["infer_ms"], roundtrip_ms)

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
