"""Detector adapters, registered in ``hoops.registry.detectors``.

Every adapter returns ``supervision.Detections`` with ``data["class_name"]`` normalized to CLASS_VOCAB
(``class_id`` is the index into CLASS_VOCAB) and implements ``info()`` for the run manifest.
Heavy imports (torch, rfdetr, inference_models, ultralytics) stay inside constructors.

Optional hook: a detector may define ``set_clock(frame_index, t_s)``; ``run_perception`` calls it before
each ``detect()`` (the scripted test detector uses it).
"""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from hoops.config import get_settings
from hoops.perception.base import CLASS_VOCAB, LABEL_MAP
from hoops.registry import detectors

if TYPE_CHECKING:
    import supervision as sv

Row = tuple[str, float, float, float, float, float]  # (class_name, score, x1, y1, x2, y2)
BallPass = Literal["off", "hires", "sliced"]

COCO_KEEP = ("person", "ball")
YOLOE_PROMPTS = {
    "basketball player": "player",
    "basketball": "ball",
    "basketball hoop": "rim",
    "referee": "referee",
}


# ---------------------------------------------------------------- helpers


def pick_device(prefer: str | None = None) -> str:
    """'mps' on Apple Silicon (with CPU fallback for unsupported ops), else 'cuda', else 'cpu'."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before torch is imported
    if prefer:
        return prefer
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def empty_detections() -> sv.Detections:
    import supervision as sv

    dets = sv.Detections.empty()
    dets.class_id = np.array([], dtype=int)
    dets.data["class_name"] = np.array([], dtype=object)
    return dets


def dets_from_rows(rows: Iterable[Row]) -> sv.Detections:
    """Build Detections from ``(class_name, score, x1, y1, x2, y2)`` tuples (tests, fakes, scripted runs)."""
    import supervision as sv

    rows = list(rows)
    if not rows:
        return empty_detections()
    names = np.array([r[0] for r in rows], dtype=object)
    return sv.Detections(
        xyxy=np.array([r[2:6] for r in rows], dtype=float).reshape(-1, 4),
        confidence=np.array([r[1] for r in rows], dtype=float),
        class_id=np.array([CLASS_VOCAB.index(n) if n in CLASS_VOCAB else -1 for n in names], dtype=int),
        data={"class_name": names},
    )


def normalize_labels(
    dets: sv.Detections,
    keep: Collection[str] | None = None,
    label_map: Mapping[str, str] = LABEL_MAP,
    id_names: Sequence[str] | None = None,
) -> sv.Detections:
    """Map model labels onto CLASS_VOCAB by NAME (never by numeric id) and drop everything else.

    ``id_names`` is a fallback when the model gives class ids but no ``class_name``.
    """
    names = dets.data.get("class_name")
    if names is None:
        ids = dets.class_id if dets.class_id is not None else np.full(len(dets), -1)
        names = [id_names[i] if id_names is not None and 0 <= i < len(id_names) else "" for i in ids]
    mapped = []
    for n in names:
        s = str(n).strip()
        mapped.append(label_map.get(s.lower(), label_map.get(s, s if s in CLASS_VOCAB else "")))
    mapped_arr = np.array(mapped, dtype=object)
    ok = np.array([m in CLASS_VOCAB and (keep is None or m in keep) for m in mapped], dtype=bool)
    out = dets[ok] if len(dets) else dets
    out.data = {"class_name": mapped_arr[ok] if len(dets) else np.array([], dtype=object)}
    out.class_id = np.array([CLASS_VOCAB.index(m) for m in mapped_arr[ok]], dtype=int)
    out.metadata = {}
    out.mask = None
    if out.confidence is None:
        out.confidence = np.ones(len(out), dtype=float)
    return out


def _split_by_score(dets: sv.Detections, conf: float, ball_conf: float) -> sv.Detections:
    """Keep balls at ``ball_conf`` (the ball filter gates them later) and everything else at ``conf``."""
    if not len(dets):
        return dets
    is_ball = dets.data["class_name"] == "ball"
    keep = np.where(is_ball, dets.confidence >= ball_conf, dets.confidence >= conf)
    return dets[keep]


def _top_ball(dets: sv.Detections) -> sv.Detections:
    """All non-ball detections plus the single highest-scoring ball."""
    if not len(dets):
        return dets
    is_ball = dets.data["class_name"] == "ball"
    if is_ball.sum() <= 1:
        return dets
    best = np.flatnonzero(is_ball)[np.argmax(dets.confidence[is_ball])]
    keep = ~is_ball
    keep[best] = True
    return dets[keep]


# ---------------------------------------------------------------- RF-DETR (COCO weights)


class RFDETRDetector:
    """RF-DETR with COCO weights: person + sports ball. COCO has no rim class (the user draws the rim).

    ``ball_pass`` adds a second, ball-only pass: "hires" re-runs at ``ball_resolution``; "sliced" tiles the
    frame with ``sv.InferenceSlicer``. The best ball of both passes is kept.
    """

    key = "rfdetr:coco"

    def __init__(
        self,
        size: Literal["nano", "small", "medium", "large"] = "small",
        conf: float = 0.35,
        ball_conf: float = 0.15,
        resolution: int | None = None,
        ball_pass: BallPass = "off",
        ball_resolution: int = 1280,
        slice_px: int = 640,
        device: str | None = None,
    ):
        self.device = pick_device(device)
        import rfdetr

        classes = {
            "nano": rfdetr.RFDETRNano,
            "small": rfdetr.RFDETRSmall,
            "medium": rfdetr.RFDETRMedium,
            "large": rfdetr.RFDETRLarge,
        }
        if size not in classes:
            raise ValueError(f"unknown RF-DETR size '{size}' (choose from {sorted(classes)})")
        self.size, self.conf, self.ball_conf = size, conf, ball_conf
        self.resolution, self.ball_pass, self.ball_resolution = resolution, ball_pass, ball_resolution
        self.slice_px = slice_px
        self.model = classes[size](device=self.device)
        self._slicer: Any = None

    def _predict(self, rgb: np.ndarray, threshold: float, resolution: int | None) -> sv.Detections:
        shape = (resolution, resolution) if resolution else None
        return self.model.predict(rgb, threshold=threshold, shape=shape, include_source_image=False)

    def detect(self, frame_rgb: np.ndarray) -> sv.Detections:
        import supervision as sv

        raw = self._predict(frame_rgb, min(self.conf, self.ball_conf), self.resolution)
        dets = _split_by_score(normalize_labels(raw, keep=COCO_KEEP), self.conf, self.ball_conf)
        if self.ball_pass == "off":
            return dets
        balls = self._ball_pass(frame_rgb)
        if not len(balls):
            return dets
        return _top_ball(sv.Detections.merge([dets, balls]))

    def _ball_pass(self, frame_rgb: np.ndarray) -> sv.Detections:
        if self.ball_pass == "hires":
            raw = self._predict(frame_rgb, self.ball_conf, self.ball_resolution)
        else:
            import supervision as sv

            if self._slicer is None:
                self._slicer = sv.InferenceSlicer(
                    callback=lambda tile: self._predict(tile, self.ball_conf, self.resolution),
                    slice_wh=(self.slice_px, self.slice_px),
                    overlap_wh=(self.slice_px // 8, self.slice_px // 8),
                )
            raw = self._slicer(frame_rgb)
        return _top_ball(normalize_labels(raw, keep=("ball",)))

    def info(self) -> dict[str, Any]:
        return {
            "impl": self.key,
            "version": f"rfdetr=={_version('rfdetr')}",
            "license": "Apache-2.0",
            "device": self.device,
            "classes": list(COCO_KEEP),
            "model": f"RFDETR{self.size.title()}",
            "resolution": self.resolution or getattr(self.model.model, "resolution", None),
            "ball_pass": self.ball_pass,
        }


# ---------------------------------------------------------------- Roboflow basketball weights (inference-models)


class RoboflowBasketballDetector:
    """Roboflow Universe ``basketball-player-detection-3-ycjdo`` (RF-DETR fine-tuned on NBA broadcast).

    Runs locally through ``inference_models`` (``uv sync --extra roboflow``) with a free ROBOFLOW_API_KEY.
    Untested here (the package is not installed by default); written defensively.
    """

    key = "rfdetr:basketball"
    project = "basketball-player-detection-3-ycjdo"
    classes = (
        "ball",
        "ball-in-basket",
        "number",
        "player",
        "player-in-possession",
        "player-jump-shot",
        "player-layup-dunk",
        "player-shot-block",
        "referee",
        "rim",
    )

    def __init__(
        self,
        version: int | str = 6,
        conf: float = 0.35,
        ball_conf: float = 0.15,
        api_key: str | None = None,
        device: str | None = None,
        **_: Any,
    ):
        try:
            from inference_models import AutoModel  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "rfdetr:basketball needs the 'inference-models' package: run `uv sync --extra roboflow`"
            ) from exc
        key = api_key or get_settings().key("roboflow_api_key")
        if not key:
            raise RuntimeError("rfdetr:basketball needs ROBOFLOW_API_KEY in .env (free at roboflow.com)")
        self.model_id = f"{self.project}/{version}"
        self.conf, self.ball_conf = conf, ball_conf
        self.device = device or "auto"
        try:
            self.model = AutoModel.from_pretrained(self.model_id, api_key=key)
        except Exception as exc:
            raise RuntimeError(f"could not load Roboflow model '{self.model_id}': {exc}") from exc
        self._names = list(getattr(self.model, "class_names", None) or self.classes)

    def detect(self, frame_rgb: np.ndarray) -> sv.Detections:
        bgr = np.ascontiguousarray(
            frame_rgb[..., ::-1]
        )  # inference_models treats numpy input as BGR (OpenCV)
        preds = self.model(bgr, confidence=min(self.conf, self.ball_conf))
        pred = preds[0] if isinstance(preds, (list, tuple)) else preds
        dets = pred.to_supervision() if hasattr(pred, "to_supervision") else pred
        dets = normalize_labels(dets, id_names=self._names)
        return _split_by_score(dets, self.conf, self.ball_conf)

    def info(self) -> dict[str, Any]:
        return {
            "impl": self.key,
            "version": f"{self.model_id} (inference-models=={_version('inference-models')})",
            "license": "Apache-2.0 runtime; Roboflow Universe weights (check the project page)",
            "device": self.device,
            "classes": sorted({LABEL_MAP.get(c, c) for c in self._names}),
        }


# ---------------------------------------------------------------- Ultralytics (AGPL-3.0, optional extra)


class UltralyticsDetector:
    """YOLO26 (COCO person + sports ball) or YOLOE-26 text prompts. AGPL-3.0; untested here (not installed)."""

    def __init__(
        self,
        weights: str = "yolo26s.pt",
        conf: float = 0.35,
        ball_conf: float = 0.15,
        resolution: int | None = None,
        prompts: Mapping[str, str] | None = None,
        device: str | None = None,
        key: str = "ultralytics:yolo26s",
    ):
        self.device = pick_device(device)
        try:
            from ultralytics import YOLO, YOLOE  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics detectors need the AGPL 'ultra' extra: `uv sync --extra ultra`"
            ) from exc
        self.key, self.weights, self.conf, self.ball_conf = key, weights, conf, ball_conf
        self.resolution, self.prompts = resolution, dict(prompts or {})
        if self.prompts:
            self.model = YOLOE(weights)
            names = list(self.prompts)
            try:
                self.model.set_classes(names, self.model.get_text_pe(names))
            except (AttributeError, TypeError):
                self.model.set_classes(names)
        else:
            self.model = YOLO(weights)

    def detect(self, frame_rgb: np.ndarray) -> sv.Detections:
        import supervision as sv

        kwargs: dict[str, Any] = {
            "conf": min(self.conf, self.ball_conf),
            "device": self.device,
            "verbose": False,
        }
        if self.resolution:
            kwargs["imgsz"] = self.resolution
        result = self.model.predict(np.ascontiguousarray(frame_rgb[..., ::-1]), **kwargs)[0]  # numpy in = BGR
        dets = sv.Detections.from_ultralytics(result)
        label_map = {**LABEL_MAP, **self.prompts}
        keep = None if self.prompts else COCO_KEEP
        return _split_by_score(
            normalize_labels(dets, keep=keep, label_map=label_map), self.conf, self.ball_conf
        )

    def info(self) -> dict[str, Any]:
        return {
            "impl": self.key,
            "version": f"ultralytics=={_version('ultralytics')}:{self.weights}",
            "license": "AGPL-3.0",
            "device": self.device,
            "classes": sorted(set(self.prompts.values())) if self.prompts else list(COCO_KEEP),
        }


# ---------------------------------------------------------------- scripted fake (tests, pipeline tests)


class ScriptedDetector:
    """Returns scripted detections per frame: ``script(frame_index, t_s)`` or ``script[frame_index]``.

    Values may be ``sv.Detections`` or ``(class_name, score, x1, y1, x2, y2)`` rows. ``run_perception`` calls
    ``set_clock`` before each ``detect``; without it, frames are counted at ``fps``.
    """

    key = "fake:scripted"

    def __init__(
        self,
        script: Callable[[int, float], Any] | Mapping[int, Any] | None = None,
        fps: float = 30.0,
        **_: Any,
    ):
        self.script = script if script is not None else {}
        self.fps = fps
        self._frame, self._t = -1, 0.0
        self._clocked = False
        self.calls = 0

    def set_clock(self, frame_index: int, t_s: float) -> None:
        self._frame, self._t, self._clocked = frame_index, t_s, True

    def detect(self, frame_rgb: np.ndarray) -> sv.Detections:
        if not self._clocked:
            self._frame += 1
            self._t = self._frame / self.fps
        self._clocked = False
        self.calls += 1
        if callable(self.script):
            out = self.script(self._frame, self._t)
        else:
            out = self.script.get(self._frame)
        if out is None:
            return empty_detections()
        if not hasattr(out, "xyxy"):
            out = dets_from_rows(out)  # type: ignore[arg-type]
        return normalize_labels(out)  # type: ignore[arg-type]

    def info(self) -> dict[str, Any]:
        return {
            "impl": self.key,
            "version": "test",
            "license": "",
            "device": "cpu",
            "classes": list(CLASS_VOCAB),
        }


# ---------------------------------------------------------------- registry


@detectors.register(
    "rfdetr:coco",
    label="RF-DETR Small · COCO",
    license="Apache-2.0",
    requires_modules=("rfdetr",),
    notes="person + ball only; draw the rim box once (COCO has no rim class)",
)
def _rfdetr_coco(**kwargs: Any) -> RFDETRDetector:
    return RFDETRDetector(**kwargs)


@detectors.register(
    "rfdetr:coco-medium",
    label="RF-DETR Medium · COCO",
    license="Apache-2.0",
    requires_modules=("rfdetr",),
    notes="slower, slightly more accurate than Small",
)
def _rfdetr_coco_medium(**kwargs: Any) -> RFDETRDetector:
    return RFDETRDetector(**{"size": "medium", **kwargs})


@detectors.register(
    "rfdetr:basketball",
    label="RF-DETR · Roboflow basketball",
    license="Apache-2.0 runtime; Roboflow Universe weights",
    requires_modules=("inference_models",),
    requires_keys=("roboflow_api_key",),
    notes="player/referee/ball/rim/number + shot classes; NBA-trained (domain gap on phone footage)",
)
def _rfdetr_basketball(**kwargs: Any) -> RoboflowBasketballDetector:
    kwargs.pop("resolution", None)
    return RoboflowBasketballDetector(**kwargs)


@detectors.register(
    "ultralytics:yolo26s",
    label="YOLO26s · COCO (AGPL)",
    license="AGPL-3.0",
    requires_modules=("ultralytics",),
    notes="faster detector option; AGPL applies to hosted use",
)
def _yolo26s(**kwargs: Any) -> UltralyticsDetector:
    return UltralyticsDetector(**{"weights": "yolo26s.pt", "key": "ultralytics:yolo26s", **kwargs})


@detectors.register(
    "ultralytics:yoloe",
    label="YOLOE-26 · type what to find (AGPL)",
    license="AGPL-3.0",
    requires_modules=("ultralytics",),
    notes="open-vocabulary prompts: basketball player, basketball, basketball hoop, referee",
)
def _yoloe(**kwargs: Any) -> UltralyticsDetector:
    defaults = {"weights": "yoloe-26s-seg.pt", "prompts": YOLOE_PROMPTS, "key": "ultralytics:yoloe"}
    return UltralyticsDetector(**{**defaults, **kwargs})


@detectors.register("fake:scripted", label="Scripted (tests)", notes="returns scripted detections")
def _scripted(**kwargs: Any) -> ScriptedDetector:
    kwargs.pop("conf", None)
    kwargs.pop("resolution", None)
    return ScriptedDetector(**kwargs)
