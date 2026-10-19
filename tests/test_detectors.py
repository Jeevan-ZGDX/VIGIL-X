"""
Unit tests for independent RGB and Thermal pedestrian detectors.
"""

import numpy as np
import pytest

from src.core.config import DetectorConfig
from src.core.schemas import BoundingBox, FrameData, FrameMetadata, SensorType
from src.perception.rgb_detector import RGBPedestrianDetector, apply_nms
from src.perception.thermal_detector import ThermalPedestrianDetector


def create_test_frame(image: np.ndarray, sensor_type: SensorType) -> FrameData:
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    metadata = FrameMetadata(
        sensor_id=f"test_{sensor_type.value}",
        sensor_type=sensor_type,
        frame_id=1,
        width=w,
        height=h,
        channels=channels,
        fps=30.0,
        is_valid=True,
    )
    return FrameData(metadata=metadata, image=image)


def test_apply_nms():
    box1 = BoundingBox(x1=10, y1=10, x2=50, y2=90)
    box2 = BoundingBox(x1=12, y1=12, x2=52, y2=92)  # High IoU with box1
    box3 = BoundingBox(x1=100, y1=100, x2=140, y2=180)  # Non-overlapping
    scores = [0.90, 0.70, 0.85]

    keep = apply_nms([box1, box2, box3], scores, iou_threshold=0.3)
    assert 0 in keep
    assert 1 not in keep  # Suppressed due to overlap with 0
    assert 2 in keep


def test_thermal_detector_pedestrian_blob():
    config = DetectorConfig(
        backend="hog_contrast",
        confidence_threshold=0.35,
        min_box_area_px=100.0,
    )
    detector = ThermalPedestrianDetector(config, sensor_id="test_thermal")

    # Create cool background (ambient ~20C, grayscale 40)
    img = np.full((240, 320), fill_value=40, dtype=np.uint8)

    # Insert warm pedestrian-like upright rectangle (aspect ratio = 80/30 = 2.66)
    img[60:140, 140:170] = 220

    frame_data = create_test_frame(img, SensorType.THERMAL)
    detections = detector.detect(frame_data)

    assert len(detections) >= 1
    det = detections[0]
    assert det.sensor_type == SensorType.THERMAL
    assert det.class_name == "pedestrian"
    assert det.confidence >= 0.35
    # Bounding box should encompass the warm region
    assert det.bbox.x1 <= 145 and det.bbox.x2 >= 165
    assert det.bbox.y1 <= 65 and det.bbox.y2 >= 135


def test_thermal_detector_rejects_horizontal_heat():
    config = DetectorConfig(
        backend="hog_contrast",
        confidence_threshold=0.35,
        min_box_area_px=100.0,
    )
    detector = ThermalPedestrianDetector(config, sensor_id="test_thermal")

    # Cool background
    img = np.full((240, 320), fill_value=40, dtype=np.uint8)

    # Insert wide horizontal hot strip (e.g. exhaust pipe or heated equipment bed: aspect ratio = 20/120 = 0.16)
    img[100:120, 50:170] = 230

    frame_data = create_test_frame(img, SensorType.THERMAL)
    detections = detector.detect(frame_data)

    # Horizontal non-pedestrian heat source must be rejected by anthropometric filter
    assert len(detections) == 0


def test_rgb_detector_invalid_frame():
    config = DetectorConfig(backend="hog_contrast")
    detector = RGBPedestrianDetector(config, sensor_id="test_rgb")

    # Frame with is_valid=False
    metadata = FrameMetadata(
        sensor_id="test_rgb",
        sensor_type=SensorType.RGB,
        frame_id=1,
        width=320,
        height=240,
        channels=3,
        fps=30.0,
        is_valid=False,
    )
    frame_data = FrameData(metadata=metadata, image=None)
    assert detector.detect(frame_data) == []
