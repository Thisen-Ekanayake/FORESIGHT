"""Shared helpers for loading Qwen3-VL-2B-Instruct and extracting image embeddings."""

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "/ml/vlm_memory/Qwen3-VL-2B-Instruct"


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()
    return model, processor


def get_image_embeds(model, processor, image_path: str) -> torch.Tensor:
    """Return per-token image embeddings, shape (num_tokens, hidden_size)."""
    image = image_path if image_path.startswith("http") else Image.open(image_path).convert("RGB")
    inputs = processor.image_processor(images=image, return_tensors="pt").to(model.device, model.dtype)
    with torch.inference_mode():
        image_embeds, _deepstack_image_embeds = model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
    return image_embeds[0].float().cpu()


def mean_pool(embeds: torch.Tensor) -> list[float]:
    return embeds.mean(dim=0).tolist()
