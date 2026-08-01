# `src/foresight/perception/vlm/qwen_vl.py`

Compact VLM scene reasoning via [Qwen3-VL-2B-Instruct](https://github.com/QwenLM/Qwen3-VL), run zero-shot (no fine-tuning). Library code — no file I/O, no CLI; consumed by [`tools/describe_scene.py`](../../../../tools/describe_scene.md) (free-text scene description) and [`foresight.planning.fusion.ActionFusionController`](../../planning/fusion.md) (structured heading proposal, milestone #6).

Model choice writeup: [`docs/vlm_comparison.md`](../../../../vlm_comparison.md). The doc's stated primary pick was Gemma 4 E4B; the actual integration uses **Qwen3-VL-2B-Instruct** instead — Gemma 4's `gemma4` architecture needs `transformers>=5.x` (Python >=3.10), incompatible with the Python 3.9 `habitat-sim` 0.3.3 binary this project is pinned to. Qwen3-VL's `qwen3_vl` architecture works with the installed transformers 4.57.6.

## `DEFAULT_MODEL`

`"models/Qwen3-VL-2B-Instruct"` — a local path, not a HF hub ID. Downloaded via `scripts/download_qwen3_vl_2b.sh` into `models/` (gitignored; re-run the script to fetch it on a fresh checkout).

## `Qwen3VLSceneReasoner`

```python
reasoner = Qwen3VLSceneReasoner()                      # loads DEFAULT_MODEL, picks cuda if available
text = reasoner.describe(rgb)                           # free-text scene description
heading = reasoner.propose_heading(rgb)                 # radians rel. to straight-ahead, or None
```

- `__init__(model_name=DEFAULT_MODEL, device=None)` — loads `AutoProcessor` + `Qwen3VLForConditionalGeneration`, `bfloat16` on CUDA / `float32` on CPU, moves to `device` (`cuda` if available else `cpu`), eval mode.
- `describe(rgb, prompt=..., system_prompt=DEFAULT_SYSTEM_PROMPT, max_new_tokens=256, repetition_penalty=1.15)` — builds a chat-template message (system + user text + image), generates, decodes only the newly generated tokens. `repetition_penalty=1.15` is not the checkpoint's default (1.0) — added after the first run at `repetition_penalty=1.0` degenerated into a repeated phrase loop toward the 256-token cap; see the 2026-08-01 log.
- `propose_heading(rgb, max_new_tokens=16)` — the structured variant used by the action fusion controller. Calls `describe` with `HEADING_SYSTEM_PROMPT`/`HEADING_PROMPT` (ask for exactly one line: a heading in degrees from -90 to 90, or the word `STOP`), `repetition_penalty=1.0` (short numeric output doesn't need it), and parses the reply with a regex (`_HEADING_NUMBER_RE`). Returns `None` — not an exception — if the reply contains "stop" or doesn't contain a parseable number, so a bad VLM turn degrades gracefully: the caller (`ActionFusionController.step`) just falls back to the goal bearing for that cycle instead of crashing the control loop.

## Sign convention

`propose_heading`'s return value is radians relative to straight-ahead, **positive = right**. This matches `foresight.sim.pose.goal_bearing_rad` and `foresight.planning.fusion.ActionFusionController`'s convention — all three were designed together so a heading value means the same thing everywhere in the stack.

## Verified

Both `describe` and `propose_heading` were run against the two semantic-annotated sensor captures (`00800-TEEsavR23oF`, `00802-wcojb4TFT35`): `describe` produced coherent, grounded descriptions (obstacles, traversable space, reflective/hazard callouts); `propose_heading` parsed cleanly on both (45° and 90°, both plausible given the two scenes' layouts).
