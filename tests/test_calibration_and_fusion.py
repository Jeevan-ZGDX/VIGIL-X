"""
Unit tests for sensor synchronization, spatial calibration, and sensor fusion.
"""

import time
import numpy as np
import pytest

from src.calibration.spatial import SpatialCalibrator
from src.calibration.synchronizer import SensorSynchronizer
from src.core.config import CalibrationConfig, FusionConfig
from src.core.schemas import BoundingBox, Detection, FrameData, FrameMetadata, SensorType
from src.fusion.engine import SensorFusionEngine
from src.fusion.uncertainty import UncertaintyModel


def make_dummy_frame(timestamp_sec: float, sensor_type: SensorType) -> FrameData:
    metadata = FrameMetadata(
        sensor_id=f"dummy_{sensor_type.value}",
        sensor_type=sensor_type,
        frame_id=1,
        timestamp_sec=timestamp_sec,
        width=640,
        height=480,
        channels=3,
        fps=30.0,
        is_valid=True,
    )
    return FrameData(metadata=metadata, image=np.zeros((480, 640, 3), dtype=np.uint8))


def test_sensor_synchronizer():
    config = CalibrationConfig(sync_tolerance_sec=0.05)
    sync = SensorSynchronizer(config)

    now = time.time()
    rgb = make_dummy_frame(now, SensorType.RGB)
    thermal = make_dummy_frame(now + 0.02, SensorType.THERMAL)  # 20ms skew

    bundle, f_rgb, f_th = sync.synchronize(rgb, thermal)
    assert bundle.is_synchronized is True
    assert bundle.time_skew_sec < 0.05
    assert bundle.rgb_valid is True
    assert bundle.thermal_valid is True

    # High skew test (>50ms)
    thermal_late = make_dummy_frame(now + 0.08, SensorType.THERMAL)  # 80ms skew
    bundle_late, _, _ = sync.synchronize(rgb, thermal_late)
    assert bundle_late.is_synchronized is False
    assert bundle_late.time_skew_sec >= 0.075

    # Single sensor test (thermal missing)
    bundle_single, _, _ = sync.synchronize(rgb, None)
    assert bundle_single.is_synchronized is False
    assert bundle_single.rgb_valid is True
    assert bundle_single.thermal_valid is False


def test_spatial_calibrator_ground_projection():
    config = CalibrationConfig(
        camera_height_m=2.0,
        camera_pitch_deg=15.0,
        focal_length_px=600.0,
    )
    calibrator = SpatialCalibrator(config, image_width=640, image_height=480)

    # Center-bottom of image represents ground close to machine
    box_near = BoundingBox(x1=280, y1=300, x2=360, y2=460)
    x_near, y_near = calibrator.estimate_pedestrian_ground_position(box_near)
    assert y_near > 0.0
    assert abs(x_near) < 1.0  # Center lateral position

    # Higher up in image represents further away
    box_far = BoundingBox(x1=300, y1=150, x2=340, y2=230)
    x_far, y_far = calibrator.estimate_pedestrian_ground_position(box_far)
    assert y_far > y_near  # Further away on ground


def test_fusion_dual_sensor_agreement():
    fusion_cfg = FusionConfig(spatial_iou_threshold=0.30)
    calib_cfg = CalibrationConfig()
    calibrator = SpatialCalibrator(calib_cfg, 640, 480)
    engine = SensorFusionEngine(fusion_cfg, calibrator)

    t = time.time()
    # Matching pedestrian in both sensors
    rgb_det = Detection(
        detection_id="rgb_01",
        frame_id=1,
        timestamp_sec=t,
        sensor_type=SensorType.RGB,
        sensor_id="cam_rgb",
        confidence=0.75,
        bbox=BoundingBox(x1=200, y1=150, x2=280, y2=350),
        uncertainty=0.20,
    )
    th_det = Detection(
        detection_id="th_01",
        frame_id=1,
        timestamp_sec=t,
        sensor_type=SensorType.THERMAL,
        sensor_id="cam_thermal",
        confidence=0.80,
        bbox=BoundingBox(x1=205, y1=148, x2=285, y2=352),  # Very high IoU with rgb_det
        uncertainty=0.15,
    )

    fused = engine.fuse([rgb_det], [th_det], mode="fused")
    assert len(fused) == 1
    assert fused[0].sensor_type == SensorType.FUSED
    assert fused[0].confidence > 0.80  # Agreement increases fused confidence
    assert fused[0].uncertainty < 0.15  # Fusion reduces uncertainty
    assert "rgb_01" in fused[0].source_detections
    assert "th_01" in fused[0].source_detections
    assert fused[0].calibrated_ground_pos_m is not None


def test_fusion_single_sensor_preservation():
    fusion_cfg = FusionConfig(spatial_iou_threshold=0.30)
    calibrator = SpatialCalibrator(CalibrationConfig(), 640, 480)
    engine = SensorFusionEngine(fusion_cfg, calibrator)

    t = time.time()
    # Thermal detects pedestrian in darkness, RGB detects nothing
    th_det = Detection(
        detection_id="th_dark_01",
        frame_id=1,
        timestamp_sec=t,
        sensor_type=SensorType.THERMAL,
        sensor_id="cam_thermal",
        confidence=0.85,
        bbox=BoundingBox(x1=100, y1=100, x2=150, y2=250),
        uncertainty=0.20,
    )

    fused = engine.fuse([], [th_det], mode="fused")
    assert len(fused) == 1
    assert fused[0].sensor_type == SensorType.THERMAL
    # Penalized single-sensor confidence
    assert fused[0].confidence < 0.85
    assert fused[0].uncertainty > 0.20
