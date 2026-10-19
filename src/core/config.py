"""
Configuration management and validation for the Pedestrian Safety System.
Loads YAML files, supports environment overrides, and validates with Pydantic v2.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
from pydantic import BaseModel, Field, ConfigDict


class SensorConfig(BaseModel):
    sensor_id: str
    sensor_type: str  # RGB, THERMAL
    mode: str = "replay"  # "live", "replay", "synthetic"
    device_index: Optional[int] = 0
    replay_path: Optional[str] = None
    width: int = 640
    height: int = 480
    target_fps: float = 30.0
    stale_timeout_sec: float = 0.5
    thermal_format: str = "GRAY_8BIT"  # GRAY_8BIT, RADIOMETRIC_16BIT, COLORMAP_BGR


class CalibrationConfig(BaseModel):
    sync_tolerance_sec: float = 0.050  # 50ms maximum allowable timestamp delta
    enable_spatial_alignment: bool = True
    # Ground plane projection parameters (homography / perspective)
    camera_height_m: float = 2.0
    camera_pitch_deg: float = 15.0
    focal_length_px: float = 600.0
    ground_homography_matrix: Optional[List[List[float]]] = None


class DetectorConfig(BaseModel):
    backend: str = "hog_contrast"  # "onnx", "hog_contrast", "mock"
    model_path: Optional[str] = None
    confidence_threshold: float = 0.40
    iou_nms_threshold: float = 0.45
    min_box_area_px: float = 200.0


class FusionConfig(BaseModel):
    spatial_iou_threshold: float = 0.25
    rgb_weight: float = 0.50
    thermal_weight: float = 0.50
    min_fused_confidence: float = 0.35
    single_sensor_confidence_penalty: float = 0.15


class TrackingConfig(BaseModel):
    max_missed_frames: int = 5
    min_hits_to_confirm: int = 2
    distance_threshold_m: float = 2.0
    iou_association_threshold: float = 0.20


class SafetyZonesConfig(BaseModel):
    # Safety zones relative to machine front/center in meters
    warning_distance_m: float = 8.0
    danger_distance_m: float = 3.5
    blind_spot_left_m: float = 2.0
    blind_spot_right_m: float = 2.0
    blind_spot_rear_m: float = 4.0
    time_to_collision_caution_sec: float = 4.0
    time_to_collision_danger_sec: float = 2.0


class StateMachineConfig(BaseModel):
    # Hysteresis frames to prevent alert flickering
    caution_to_normal_frames: int = 5
    danger_to_caution_frames: int = 3
    degraded_recovery_frames: int = 5


class MachineConfig(BaseModel):
    machine_id: str = "FORKLIFT_01"
    width_m: float = 1.6
    length_m: float = 3.2
    max_speed_mps: float = 5.5
    telemetry_stale_timeout_sec: float = 1.0


class SystemConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    app_name: str = "Pedestrian Safety Fusion System"
    version: str = "0.1.0"
    log_level: str = "INFO"
    event_log_path: str = "data/events.jsonl"
    rgb_sensor: SensorConfig = Field(
        default_factory=lambda: SensorConfig(
            sensor_id="cam_rgb",
            sensor_type="RGB",
            mode="replay",
            device_index=0,
            width=640,
            height=480,
            target_fps=30.0,
        )
    )
    thermal_sensor: SensorConfig = Field(
        default_factory=lambda: SensorConfig(
            sensor_id="cam_thermal",
            sensor_type="THERMAL",
            mode="replay",
            device_index=1,
            width=640,
            height=480,
            target_fps=30.0,
        )
    )
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    rgb_detector: DetectorConfig = Field(
        default_factory=lambda: DetectorConfig(
            backend="hog_contrast",
            confidence_threshold=0.40
        )
    )
    thermal_detector: DetectorConfig = Field(
        default_factory=lambda: DetectorConfig(
            backend="hog_contrast",
            confidence_threshold=0.40
        )
    )
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    zones: SafetyZonesConfig = Field(default_factory=SafetyZonesConfig)
    state_machine: StateMachineConfig = Field(default_factory=StateMachineConfig)
    machine: MachineConfig = Field(default_factory=MachineConfig)

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> SystemConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def save_to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
