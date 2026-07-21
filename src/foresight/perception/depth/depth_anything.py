"""Zero-shot monocular metric depth via Depth Anything V2 (indoor-metric checkpoint)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class ZeroShotDepthEstimator:
    """Pretrained Depth Anything V2 metric checkpoint, run as-is with no fine-tuning or
    scale/shift fitting against HM3D ground truth — this is the zero-shot condition the
    depth-error-propagation study (PROGRESS.md deliverable A) measures against sim GT depth.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: (H, W, 3) uint8. Returns (H, W) float32 metric depth in meters, at input resolution."""
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        depth = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]["predicted_depth"]
        return depth.cpu().numpy().astype(np.float32)
