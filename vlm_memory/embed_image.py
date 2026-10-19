#!/usr/bin/env python
"""Extract image embeddings from Qwen3-VL-2B-Instruct and dump them to JSON.

Usage:
    /ml/HelaBERT_Analysis/.venv/bin/python embed_image.py --image path/to/image.jpg --output embeddings.json
"""

import argparse
import json
import os

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "/ml/vlm_memory/Qwen3-VL-2B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="Extract image embeddings from Qwen3-VL-2B-Instruct")
    parser.add_argument("--image", required=True, help="Path or URL to an image")
    parser.add_argument("--output", default="embeddings.json", help="Path to output JSON file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    from PIL import Image

    image = Image.open(args.image).convert("RGB") if not args.image.startswith("http") else args.image
    inputs = processor.image_processor(images=image, return_tensors="pt").to(model.device, model.dtype)

    with torch.inference_mode():
        image_embeds, _deepstack_image_embeds = model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )

    # image_embeds is a tuple of tensors, one per image, each (num_tokens, hidden_size)
    embeds = image_embeds[0].float().cpu()

    image_path = args.image if args.image.startswith("http") else os.path.abspath(args.image)

    result = {
        "image": image_path,
        "num_tokens": embeds.shape[0],
        "embedding_size": embeds.shape[1],
        "grid_thw": inputs["image_grid_thw"][0].tolist(),
        "embeddings": embeds.tolist(),
    }

    with open(args.output, "w") as f:
        json.dump(result, f)

    print(f"Saved {result['num_tokens']} embeddings of size {result['embedding_size']} to {args.output}")


if __name__ == "__main__":
    main()
