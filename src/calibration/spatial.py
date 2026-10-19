"""
Spatial calibration and ground-plane perspective projection.
Accounts for camera mounting position, pitch angle, focal length,
and transforms image pixel coordinates into physical ground meters (X, Y).
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
import numpy as np

from src.core.config import CalibrationConfig
from src.core.schemas import BoundingBox


class SpatialCalibrator:
    """
    Handles coordinate transformations between camera image planes and
    projects ground-contact points into real-world metric coordinates.
    """

    def __init__(self, config: CalibrationConfig, image_width: int = 640, image_height: int = 480):
        self.config = config
        self.image_width = image_width
        self.image_height = image_height
        self.camera_height_m = config.camera_height_m
        self.camera_pitch_rad = math.radians(config.camera_pitch_deg)
        self.focal_length_px = config.focal_length_px
        self.principal_x = image_width / 2.0
        self.principal_y = image_height / 2.0

        # Optional homography matrix from Thermal plane to RGB plane
        self.thermal_to_rgb_homography = None
        if config.ground_homography_matrix:
            self.thermal_to_rgb_homography = np.array(config.ground_homography_matrix, dtype=np.float32)

    def transform_thermal_to_rgb_bbox(self, bbox: BoundingBox) -> BoundingBox:
        """
        Map a bounding box from thermal camera coordinates to RGB camera coordinates.
        Uses homography if configured, otherwise applies default optical axis alignment.
        """
        if self.thermal_to_rgb_homography is not None:
            pts = np.array([
                [[bbox.x1, bbox.y1]],
                [[bbox.x2, bbox.y1]],
                [[bbox.x2, bbox.y2]],
                [[bbox.x1, bbox.y2]],
            ], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pts, self.thermal_to_rgb_homography)
            xs = transformed[:, 0, 0]
            ys = transformed[:, 0, 1]
            return BoundingBox(
                x1=float(np.min(xs)),
                y1=float(np.min(ys)),
                x2=float(np.max(xs)),
                y2=float(np.max(ys)),
            )
        # Default baseline: co-axial / aligned mounting with small offset tolerance
        return bbox

    def project_image_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """
        Inverse perspective projection mapping image pixel (u, v) to ground coordinate (X, Y) in meters.
        Coordinate system:
        - X is lateral (positive = right, negative = left)
        - Y is longitudinal distance in front of machine (positive = forward)
        """
        # Vertical angle relative to optical axis
        delta_v = v - self.principal_y
        angle_from_optical_axis = math.atan2(delta_v, self.focal_length_px)

        # Total angle with respect to ground
        total_pitch = self.camera_pitch_rad + angle_from_optical_axis

        # If pixel is above horizon (total_pitch <= 0), point is not on ground plane
        if total_pitch <= 0.01:
            return (0.0, 50.0)  # Horizon / far boundary fallback

        # Longitudinal distance Y along ground plane
        y_m = self.camera_height_m / math.tan(total_pitch)
        y_m = max(0.5, min(60.0, y_m))  # Bound physical estimation range

        # Lateral distance X along ground plane
        delta_u = u - self.principal_x
        slant_distance = math.sqrt(self.camera_height_m**2 + y_m**2)
        x_m = (delta_u / self.focal_length_px) * y_m

        return (round(x_m, 2), round(y_m, 2))

    def estimate_pedestrian_ground_position(self, bbox: BoundingBox) -> Tuple[float, float]:
        """
        Uses the bottom center of the pedestrian bounding box as the ground contact point.
        """
        u, v = bbox.bottom_center
        return self.project_image_to_ground(u, v)
