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

# Generic (scene-agnostic) nudge to check for reflections before describing. With the plain describe prompt,
# Qwen3-VL-2B narrated a hallway mirror in 00808 as "an open doorway into another hallway" on most frames;
# asking it to rule reflections in/out first surfaced the mirror on 4/4 approach frames (1/6 false alarm on
# mirror-free controls) — results/runs/mirror_trap/test_reflection_prompt.py, a 10-frame sanity check only.
REFLECTION_AWARE_PROMPT = (
    "First, in one sentence, say whether any open space, doorway or room in view could actually be a reflection in a "
    "mirror or glass, and what visual evidence supports that. Then describe this scene for a robot deciding where it is "
    "safe to move next."
)

_HEADING_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Qwen3-VL's native grounding format: boxes as [x1, y1, x2, y2] normalised to 0-1000. One category per query:
# asking for several at once made the 2B model emit a box for every listed category (results/runs/object_search/
# probe_grounding.py: single-category recall 0.8-1.0 at ~0.3 s; an open "list the objects" query recalled <= 0.4
# and took ~3.4 s).
GROUNDING_PROMPT = (
    'Locate every {name} in the image and output its bbox_2d coordinates in JSON as '
    '[{{"label": "{label}", "bbox_2d": [x1, y1, x2, y2]}}]. If there is no {name}, output [].'
)
GROUNDING_NAMES = {"tv": "TV (television screen)", "plant": "potted plant"}
# Search-direction hint for object search. results/runs/object_search/probe_direction.py (50 poses, 00808/00800):
# argmax matches the shortest path's sector 52% of the time (chance 33%), mean P(right sector) 0.43; it almost
# never answers "center". A weak, soft prior — used to discount frontier costs, never to steer directly.
DIRECTION_PROMPT = (
    "You are a robot looking for a {name}. Which direction should you go to find it: left, center, or right? "
    "Answer with one word."
)
# Presence check gating grounding (P(yes) from one forward pass). Spot check on probe frames: P(yes) ~1.0 on 11/12
# frames with a bed/sofa, ~0 on 19/20 without; a black padded bench the grounding query boxed as a "bed" scored 0.00.
PRESENT_PROMPT = "Is there a {name} in this image? Answer yes or no."
_BOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')


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

    def locate(self, rgb: np.ndarray, label: str, max_new_tokens: int = 120) -> list:
        """Zero-shot grounding of one object category. Returns boxes [x1, y1, x2, y2] normalised to 0-1000
        (Qwen3-VL's convention), possibly empty. The 2B model over-reports (a box on ~15-35% of frames
        without the object in the probe), so callers should require repeated sightings."""
        name = GROUNDING_NAMES.get(label, label)
        text = self.describe(
            rgb,
            prompt=GROUNDING_PROMPT.format(name=name, label=label),
            system_prompt="You are a helpful assistant.",
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
        )
        boxes = []
        for m in _BOX_RE.finditer(text):
            x1, y1, x2, y2 = (int(v) for v in m.groups())
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 - x1 >= 5 and y2 - y1 >= 5:
                boxes.append([x1, y1, x2, y2])
        return boxes

    @torch.inference_mode()
    def option_probs(self, rgb: np.ndarray, prompt: str, options: list,
                     system_prompt: str = "You are a helpful assistant.") -> np.ndarray:
        """One forward pass: the model's next-token probabilities over `options` (each scored by its first
        token, renormalised over the options), as the answer to `prompt` about the frame."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": Image.fromarray(rgb)}, {"type": "text", "text": prompt}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()  # last position only: full-vocab logits for every image token cost ~0.2 GB
        ids = [self.processor.tokenizer.encode(o, add_special_tokens=False)[0] for o in options]
        return torch.softmax(logits[ids], dim=0).cpu().numpy()
