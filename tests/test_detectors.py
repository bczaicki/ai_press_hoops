from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from hoops import registry
from hoops.perception.base import CLASS_VOCAB
from hoops.perception.detectors import ScriptedDetector, dets_from_rows, normalize_labels


def test_normalize_maps_by_name_and_drops_unknown_labels():
    raw = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [5, 5, 9, 9], [1, 1, 3, 3], [2, 2, 4, 4]], dtype=float),
        confidence=np.array([0.9, 0.8, 0.7, 0.6]),
        class_id=np.array([1, 37, 3, 99]),  # numeric ids are ignored on purpose (they differ per framework)
        data={"class_name": np.array(["person", "sports ball", "car", "Player-Jump-Shot"], dtype=object)},
    )
    out = normalize_labels(raw)
    assert list(out.data["class_name"]) == ["person", "ball", "jump_shot"]
    assert list(out.class_id) == [CLASS_VOCAB.index(c) for c in ["person", "ball", "jump_shot"]]
    coco = normalize_labels(raw, keep=("person", "ball"))
    assert list(coco.data["class_name"]) == ["person", "ball"]
    empty = normalize_labels(sv.Detections.empty())
    assert len(empty) == 0 and "class_name" in empty.data


def test_scripted_detector_accepts_dicts_and_counts_frames():
    det = ScriptedDetector({1: [("ball", 0.5, 1, 1, 5, 5)], 2: dets_from_rows([("hoop", 0.9, 0, 0, 9, 3)])})
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert len(det.detect(frame)) == 0  # frame 0
    assert list(det.detect(frame).data["class_name"]) == ["ball"]
    assert list(det.detect(frame).data["class_name"]) == ["rim"]  # 'hoop' normalized via LABEL_MAP
    assert det.info()["impl"] == "fake:scripted"


def test_registry_reports_optional_detectors():
    available = registry.detectors.available()
    assert available["rfdetr:coco"] is None
    assert available["fake:scripted"] is None
    assert "inference_models" in (available["rfdetr:basketball"] or "inference_models")
    entry = registry.detectors.entry("ultralytics:yolo26s")
    assert entry.license == "AGPL-3.0"


@pytest.mark.slow
def test_rfdetr_coco_runs_on_one_frame():
    try:
        det = registry.detectors.create("rfdetr:coco", conf=0.3)
    except Exception as exc:  # noqa: BLE001 - weights download needs network
        pytest.skip(f"RF-DETR weights unavailable: {exc}")
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    frame[200:600, 300:420] = (200, 40, 40)  # a red block; COCO probably finds nothing, which is fine
    out = det.detect(frame)
    assert isinstance(out, sv.Detections)
    assert set(out.data["class_name"]) <= {"person", "ball"}
    info = det.info()
    assert info["impl"] == "rfdetr:coco" and info["license"] == "Apache-2.0"
    assert info["device"] in {"mps", "cpu", "cuda"}
    det.ball_pass = "hires"
    assert isinstance(det.detect(frame), sv.Detections)
    det.ball_pass = "sliced"
    assert isinstance(det.detect(frame), sv.Detections)
