"""
Pedestrian track object representing state, trajectory history, and kinematic motion estimates.
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
from src.core.schemas import BoundingBox, Detection, SensorType, Track, TrackState


class PedestrianTrack:
    """
    Internal state representation for an individual pedestrian track.
    Tracks metric ground position, velocity vector (vx, vy), confidence, and age.
    """

    def __init__(self, track_id: str, initial_detection: Detection):
        self.track_id = track_id
        self.class_name = initial_detection.class_name
        self.state = TrackState.TENTATIVE
        self.first_seen_sec = initial_detection.timestamp_sec
        self.last_seen_sec = initial_detection.timestamp_sec
        self.age_frames = 1
        self.missed_frames = 0
        self.hit_streak = 1
        self.bbox = initial_detection.bbox

        # Ground position (X: lateral, Y: longitudinal distance in meters)
        self.ground_pos_m = initial_detection.calibrated_ground_pos_m or (0.0, 5.0)
        self.ground_velocity_mps = (0.0, 0.0)
        self.estimated_speed_mps = 0.0

        self.confidence = initial_detection.confidence
        self.uncertainty = initial_detection.uncertainty
        self.sensor_sources = [initial_detection.sensor_type]
        self.trajectory_history: List[Tuple[float, float, float]] = [
            (initial_detection.timestamp_sec, self.ground_pos_m[0], self.ground_pos_m[1])
        ]

    def update(self, detection: Detection):
        """Update track with an associated detection in current frame."""
        dt = max(0.001, detection.timestamp_sec - self.last_seen_sec)
        self.last_seen_sec = detection.timestamp_sec
        self.age_frames += 1
        self.missed_frames = 0
        self.hit_streak += 1
        self.bbox = detection.bbox

        new_ground_pos = detection.calibrated_ground_pos_m or self.ground_pos_m
        old_x, old_y = self.ground_pos_m
        new_x, new_y = new_ground_pos

        # Instantaneous velocity
        inst_vx = (new_x - old_x) / dt
        inst_vy = (new_y - old_y) / dt

        # Exponential moving average filter for velocity smoothing
        alpha = 0.40
        vx = alpha * inst_vx + (1.0 - alpha) * self.ground_velocity_mps[0]
        vy = alpha * inst_vy + (1.0 - alpha) * self.ground_velocity_mps[1]

        # Clamp physically unrealistic human sprint jumps (> 8 m/s = ~29 km/h)
        speed = math.sqrt(vx**2 + vy**2)
        if speed > 8.0:
            scale = 8.0 / speed
            vx *= scale
            vy *= scale
            speed = 8.0

        self.ground_pos_m = (round(new_x, 2), round(new_y, 2))
        self.ground_velocity_mps = (round(vx, 2), round(vy, 2))
        self.estimated_speed_mps = round(speed, 2)

        # Smooth confidence and uncertainty
        self.confidence = 0.70 * self.confidence + 0.30 * detection.confidence
        self.uncertainty = 0.70 * self.uncertainty + 0.30 * detection.uncertainty

        if detection.sensor_type not in self.sensor_sources:
            self.sensor_sources.append(detection.sensor_type)

        self.trajectory_history.append((detection.timestamp_sec, self.ground_pos_m[0], self.ground_pos_m[1]))
        if len(self.trajectory_history) > 30:
            self.trajectory_history.pop(0)

    def predict_coast(self, dt: float):
        """Coast track forward during temporary missed detection frames."""
        self.age_frames += 1
        self.missed_frames += 1
        self.hit_streak = 0
        self.last_seen_sec += dt

        # Coast position based on estimated velocity
        vx, vy = self.ground_velocity_mps
        px, py = self.ground_pos_m
        self.ground_pos_m = (round(px + vx * dt, 2), round(max(0.5, py + vy * dt), 2))

        # Increase uncertainty when coasting
        self.uncertainty = min(0.95, self.uncertainty + 0.15)
        self.confidence = max(0.10, self.confidence * 0.85)

    def to_schema(self) -> Track:
        """Convert to public Pydantic Track schema."""
        return Track(
            track_id=self.track_id,
            class_name=self.class_name,
            state=self.state,
            first_seen_sec=self.first_seen_sec,
            last_seen_sec=self.last_seen_sec,
            age_frames=self.age_frames,
            missed_frames=self.missed_frames,
            hit_streak=self.hit_streak,
            bbox=self.bbox,
            ground_pos_m=self.ground_pos_m,
            ground_velocity_mps=self.ground_velocity_mps,
            estimated_speed_mps=self.estimated_speed_mps,
            confidence=round(self.confidence, 3),
            uncertainty=round(self.uncertainty, 3),
            sensor_sources=self.sensor_sources,
            recent_trajectory=[(p[1], p[2]) for p in self.trajectory_history[-10:]],
        )
