"""Perception contracts (T1).

Conventions every implementation follows:
- Frames are RGB uint8 numpy arrays (H, W, 3) from the 720p proxy. Convert at library boundaries
  (supervision/OpenCV annotators draw in BGR order).
- Detections are ``supervision.Detections`` with ``data["class_name"]`` normalized to CLASS_VOCAB.
- Time is seconds on the master timeline (``t_s``), never a bare frame index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import supervision as sv

# Normalized class names. Detector adapters map their model's labels onto these.
CLASS_VOCAB = (
    "player",
    "referee",
    "ball",
    "rim",
    "person",  # generic COCO person when player/referee can't be told apart
    "number",  # jersey number box
    "ball_in_basket",
    "jump_shot",
    "layup_dunk",
    "shot_block",
    "in_possession",
)

# Common label spellings -> CLASS_VOCAB (extend in adapters as needed).
LABEL_MAP = {
    "sports ball": "ball",
    "basketball": "ball",
    "ball": "ball",
    "rim": "rim",
    "hoop": "rim",
    "basket": "rim",
    "player": "player",
    "referee": "referee",
    "person": "person",
    "number": "number",
    "ball-in-basket": "ball_in_basket",
    "player-jump-shot": "jump_shot",
    "player-layup-dunk": "layup_dunk",
    "player-shot-block": "shot_block",
    "player-in-possession": "in_possession",
}


@runtime_checkable
class Detector(Protocol):
    key: str

    def detect(self, frame_rgb: np.ndarray) -> sv.Detections:
        """Detect objects in one RGB frame. data["class_name"] uses CLASS_VOCAB."""
        ...

    def info(self) -> dict[str, Any]:
        """{'impl', 'version', 'license', 'device', 'classes'} for the run manifest."""
        ...

    # Optional: ``set_clock(frame_index: int, t_s: float)`` is called before each detect() when present
    # (scripted/fake detectors use it to replay detections by time).


@runtime_checkable
class Tracker(Protocol):
    key: str

    def update(self, dets: sv.Detections, frame_rgb: np.ndarray, t_s: float) -> sv.Detections:
        """Assign tracker_id to people (players/referees). Ball handling is separate (top-1 + gating)."""
        ...

    def reset(self) -> None:
        """Called at scene cuts: IDs never survive a broadcast cut."""
        ...
