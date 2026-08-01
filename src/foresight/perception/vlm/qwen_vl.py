"""Compact VLM scene reasoning via Qwen3-VL-2B-Instruct (zero-shot, no fine-tuning)."""
from __future__ import annotations

import re

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_MODEL = "models/Qwen3-VL-2B-Instruct"

DEFAULT_SYSTEM_PROMPT = (
    "You are the scene-reasoning module of a monocular robot navigator. Given a single "
    "first-person RGB frame, describe traversable free space, obstacles, and any "
    "reflective or textureless surfaces (glass, mirrors, blank walls) that could fool a "
    "depth estimator. Be concise and grounded only in what is visible."
)

HEADING_SYSTEM_PROMPT = (
    "You are the heading-proposal module of a monocular robot navigator, used at low "
    "frequency (roughly once every 0.5-2 seconds) to redirect a fast reactive controller "
    "that handles per-frame obstacle avoidance on its own. Propose a direction of travel "
    "toward open, traversable free space, steering away from clutter, glass, and mirrors."
)

HEADING_PROMPT = (
    "Respond with EXACTLY one line, nothing else: a heading in degrees relative to "
    "straight ahead, from -90 (hard left) to 90 (hard right), 0 = straight ahead. "
    "If the way ahead is entirely blocked, respond with the single word STOP instead."
)

_HEADING_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class Qwen3VLSceneReasoner:
    """Qwen3-VL-2B-Instruct run zero-shot for scene description / navigation-relevant
    reasoning over a single RGB frame, picked over Gemma 4 E4B because Gemma 4's
    `gemma4` architecture needs transformers>=5.x (Python>=3.10), incompatible with the
    Python 3.9 `habitat-sim` 0.3.3 binary this project is pinned to. Qwen3-VL's
    `qwen3_vl` architecture works with the installed transformers 4.57.6.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def describe(
        self,
        rgb: np.ndarray,
        prompt: str = "Describe this scene for a robot deciding where it is safe to move next.",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.15,
    ) -> str:
        """rgb: (H, W, 3) uint8. Returns the model's free-text scene description."""
        image = Image.fromarray(rgb)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty
        )
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

    def propose_heading(self, rgb: np.ndarray, max_new_tokens: int = 16) -> float | None:
        """Low-frequency heading proposal for `foresight.planning.fusion.ActionFusionController`.

        Returns a heading in radians relative to straight-ahead (positive = right), or None if
        the model reports the way is blocked, or its output doesn't parse as a heading — in
        either case the controller falls back to the goal bearing for this cycle.
        """
        text = self.describe(
            rgb,
            prompt=HEADING_PROMPT,
            system_prompt=HEADING_SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
        )
        if "stop" in text.lower():
            return None
        match = _HEADING_NUMBER_RE.search(text)
        if match is None:
            return None
        degrees = max(-90.0, min(90.0, float(match.group())))
        return float(np.deg2rad(degrees))
