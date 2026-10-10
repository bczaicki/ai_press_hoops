"""Zero-shot rim finder (OWLv2, Apache-2.0) for detectors without a rim class (e.g. COCO).

Suggestions feed the UI's "tap the hoop being played at" step, and the pipeline picks one
automatically (the hoop the ball actually approaches) when the user didn't mark a rim.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_ID = "google/owlv2-base-patch16-ensemble"
QUERIES = ["a basketball hoop", "a basketball rim with net", "an orange basketball rim"]

Box = tuple[float, float, float, float]


@dataclass
class RimSuggestion:
    box: Box  # rim (not net) in the image's pixels
    score: float
    hoop_box: Box  # raw detection, may include the net


@functools.lru_cache(maxsize=1)
def _model():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).eval().to(device)
    return processor, model, device


def rim_from_hoop(box: Box) -> Box:
    """Detections often include the net; keep the rim band (top of the box) when the box is tall."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w > 0 and h / w > 0.6:
        y2 = y1 + max(h * 0.5, w * 0.35)
    return (x1, y1, x2, y2)


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def suggest_rims(image_rgb: np.ndarray, max_n: int = 3, threshold: float = 0.15) -> list[RimSuggestion]:
    """Up to ``max_n`` likely rims in one RGB frame, best first (merged across queries)."""
    import torch
    from PIL import Image

    processor, model, device = _model()
    img = Image.fromarray(image_rgb)
    side = max(img.size)  # OWLv2 pads to a square; boxes come back in padded coordinates
    inputs = processor(text=[QUERIES], images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        out, threshold=threshold, target_sizes=torch.tensor([[side, side]])
    )[0]
    w, h = img.size
    raw = []
    for score, box in zip(res["scores"].tolist(), res["boxes"].tolist(), strict=True):
        x1, y1, x2, y2 = (max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(h), box[3]))
        if x2 - x1 >= 6 and y2 - y1 >= 4:
            raw.append((float(score), (x1, y1, x2, y2)))
    kept: list[RimSuggestion] = []
    for score, box in sorted(raw, reverse=True):
        if all(_iou(box, k.hoop_box) < 0.3 for k in kept):
            kept.append(RimSuggestion(box=rim_from_hoop(box), score=round(score, 3), hoop_box=box))
        if len(kept) >= max_n:
            break
    return kept


def pick_active_rim(suggestions: list[RimSuggestion], detections: pd.DataFrame) -> RimSuggestion | None:
    """The suggestion the ball approaches most: ball points within ~4 rim-widths and above rim level."""
    if not suggestions:
        return None
    balls = detections[detections["cls"] == "ball"] if len(detections) else detections
    if balls is None or not len(balls):
        return suggestions[0]
    bx = ((balls["x1"] + balls["x2"]) / 2).to_numpy()
    by = ((balls["y1"] + balls["y2"]) / 2).to_numpy()

    def activity(s: RimSuggestion) -> float:
        x1, y1, x2, y2 = s.box
        cx, rw = (x1 + x2) / 2, max(x2 - x1, 1.0)
        near = (np.abs(bx - cx) <= 4 * rw) & (by <= y2 + 2 * (y2 - y1)) & (by >= y1 - 6 * (y2 - y1))
        return float(near.sum()) + s.score  # score breaks ties

    return max(suggestions, key=activity)
