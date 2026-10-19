"""
Core data schemas and typed data structures for the Pedestrian Safety System.
Enforces Pydantic v2 validation, explicit units, coordinate systems, and serialization.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import time
import uuid
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator


class SensorType(str, Enum):
    RGB = "RGB"
    THERMAL = "THERMAL"
    FUSED = "FUSED"


class SensorHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    STALE = "STALE"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"
    RECOVERING = "RECOVERING"


class AlertState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    DANGER = "DANGER"
    DEGRADED = "DEGRADED"


class SafetyZoneType(str, Enum):
    WARNING_ZONE = "WARNING_ZONE"      # Caution zone (e.g., 5-10 meters)
    DANGER_ZONE = "DANGER_ZONE"        # Imminent hazard zone (e.g., 0-5 meters)
    BLIND_SPOT_ZONE = "BLIND_SPOT_ZONE" # Side/rear machine blind spots


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    DELETED = "DELETED"


class BoundingBox(BaseModel):
    """
    Axis-aligned 2D bounding box in pixel coordinates [x1, y1, x2, y2].
    Origin (0,0) is top-left corner of image.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    x1: float = Field(..., description="Top-left x pixel coordinate")
    y1: float = Field(..., description="Top-left y pixel coordinate")
    x2: float = Field(..., description="Bottom-right x pixel coordinate")
    y2: float = Field(..., description="Bottom-right y pixel coordinate")

    @field_validator("x2")
    @classmethod
    def validate_x(cls, v: float, info) -> float:
        x1 = info.data.get("x1", 0.0)
        if v < x1:
            raise ValueError(f"x2 ({v}) cannot be less than x1 ({x1})")
        return v

    @field_validator("y2")
    @classmethod
    def validate_y(cls, v: float, info) -> float:
        y1 = info.data.get("y1", 0.0)
        if v < y1:
            raise ValueError(f"y2 ({v}) cannot be less than y1 ({y1})")
        return v

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Ground contact point approximation in image coordinates (center of bottom edge)."""
        return (self.center_x, self.y2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xyxy(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xywh(self) -> List[float]:
        return [self.x1, self.y1, self.width, self.height]

    def iou(self, other: BoundingBox) -> float:
        """Compute Intersection over Union (IoU) with another bounding box."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area


class FrameMetadata(BaseModel):
    """Metadata describing a captured or replayed sensor frame."""
    sensor_id: str
    sensor_type: SensorType
    frame_id: int
    timestamp_ns: int = Field(default_factory=time.time_ns)
    timestamp_sec: float = Field(default_factory=time.time)
    width: int
    height: int
    channels: int
    fps: float = 30.0
    is_valid: bool = True
    error_msg: Optional[str] = None
    stale_duration_sec: float = 0.0


class FrameData:
    """Container holding frame array and its validated metadata."""
    def __init__(self, metadata: FrameMetadata, image: Optional[np.ndarray]):
        self.metadata = metadata
        self.image = image

    @property
    def is_valid(self) -> bool:
        return self.metadata.is_valid and self.image is not None and self.image.size > 0


class Detection(BaseModel):
    """Individual or fused object detection output."""
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    frame_id: int
    timestamp_sec: float
    sensor_type: SensorType
    sensor_id: str
    class_name: str = "pedestrian"
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    calibrated_ground_pos_m: Optional[Tuple[float, float]] = Field(
        default=None, description="Physical coordinates (X, Y) in meters relative to camera/machine"
    )
    uncertainty: float = Field(default=0.1, ge=0.0, le=1.0)
    source_detections: List[str] = Field(default_factory=list, description="IDs of component detections if fused")
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class Track(BaseModel):
    """Temporal pedestrian track state."""
    track_id: str
    class_name: str = "pedestrian"
    state: TrackState = TrackState.TENTATIVE
    first_seen_sec: float
    last_seen_sec: float
    age_frames: int = 1
    missed_frames: int = 0
    hit_streak: int = 1
    bbox: BoundingBox
    ground_pos_m: Tuple[float, float] = (0.0, 0.0)  # (x, y) in meters
    ground_velocity_mps: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in meters/sec
    estimated_speed_mps: float = 0.0
    confidence: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.2, ge=0.0, le=1.0)
    sensor_sources: List[SensorType] = Field(default_factory=list)
    recent_trajectory: List[Tuple[float, float]] = Field(default_factory=list)


class MachineTelemetry(BaseModel):
    """Industrial equipment telemetry and kinematic state."""
    timestamp_sec: float = Field(default_factory=time.time)
    machine_id: str = "EQUIP_01"
    is_valid: bool = True
    speed_mps: float = Field(default=0.0, description="Forward speed in m/s")
    direction_deg: float = Field(default=0.0, description="Heading relative to forward axis (0=straight)")
    steering_angle_deg: float = 0.0
    turning_rate_degps: float = 0.0
    gear: str = "FORWARD"  # FORWARD, REVERSE, NEUTRAL, PARK
    width_m: float = 2.5
    length_m: float = 5.0
    safety_margin_m: float = 1.0


class RiskAssessment(BaseModel):
    """Deterministic output from collision risk evaluation."""
    timestamp_sec: float = Field(default_factory=time.time)
    alert_state: AlertState
    reason: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    active_zone: Optional[SafetyZoneType] = None
    critical_track_id: Optional[str] = None
    time_to_collision_sec: Optional[float] = None
    min_distance_m: Optional[float] = None
    machine_telemetry_valid: bool = True
    sensor_health_summary: Dict[str, SensorHealthStatus] = Field(default_factory=dict)
    uncertainty_level: float = 0.0


class SystemHealthReport(BaseModel):
    """Overall system operational telemetry, latency, and fault states."""
    timestamp_sec: float = Field(default_factory=time.time)
    alert_state: AlertState
    overall_health: str = "OK"
    fps: float = 0.0
    latency_capture_ms: float = 0.0
    latency_inference_ms: float = 0.0
    latency_fusion_ms: float = 0.0
    latency_risk_ms: float = 0.0
    latency_total_ms: float = 0.0
    sensors: Dict[str, SensorHealthStatus] = Field(default_factory=dict)
    dropped_frames: Dict[str, int] = Field(default_factory=dict)
    active_tracks_count: int = 0
    confirmed_pedestrians_count: int = 0
