#!/usr/bin/env python
"""Inference script for Qwen3-VL-2B-Instruct.

Usage:
    /ml/HelaBERT_Analysis/.venv/bin/python infer.py --image path/to/image.jpg --prompt "Describe this image."
    /ml/HelaBERT_Analysis/.venv/bin/python infer.py --prompt "Hello, who are you?"   # text-only
"""

import argparse

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "/ml/vlm_memory/Qwen3-VL-2B-Instruct"


def build_messages(prompt: str, image_path: str | None):
    content = []
    if image_path:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen3-VL-2B-Instruct")
    parser.add_argument("--prompt", required=True, help="Text prompt / question")
    parser.add_argument("--image", default=None, help="Path or URL to an image (optional)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )

    messages = build_messages(args.prompt, args.image)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(output_text)


if __name__ == "__main__":
    main()
