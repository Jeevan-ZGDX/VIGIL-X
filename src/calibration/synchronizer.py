"""
Sensor stream synchronization and temporal alignment.
Monitors timestamp skew between RGB and Thermal streams, detects jitter and dropped frames.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel

from src.core.config import CalibrationConfig
from src.core.schemas import FrameData, SensorHealthStatus, SensorType

logger = logging.getLogger(__name__)


class SyncFrameBundle(BaseModel):
    """Container for time-synchronized sensor frame pair."""
    timestamp_sec: float
    time_skew_sec: float
    is_synchronized: bool
    rgb_valid: bool
    thermal_valid: bool
    rgb_frame_id: int
    thermal_frame_id: int
    diagnostics: Dict[str, Any] = {}


class SensorSynchronizer:
    """
    Pairs frames from RGB and Thermal sensors based on timestamp proximity.
    Surfaces timestamp skew and synchronization faults.
    """

    def __init__(self, config: CalibrationConfig):
        self.sync_tolerance_sec = config.sync_tolerance_sec
        self.skew_history = []
        self.max_skew_observed = 0.0
        self.out_of_sync_count = 0

    def synchronize(
        self,
        rgb_frame: Optional[FrameData],
        thermal_frame: Optional[FrameData],
    ) -> Tuple[SyncFrameBundle, Optional[FrameData], Optional[FrameData]]:
        """
        Evaluate temporal alignment between RGB and Thermal frames.
        Returns SyncFrameBundle metadata along with the validated frame objects.
        """
        rgb_valid = rgb_frame is not None and rgb_frame.is_valid
        thermal_valid = thermal_frame is not None and thermal_frame.is_valid

        # Case 1: Both frames present
        if rgb_valid and thermal_valid:
            t_rgb = rgb_frame.metadata.timestamp_sec
            t_thermal = thermal_frame.metadata.timestamp_sec
            skew = abs(t_rgb - t_thermal)

            self.max_skew_observed = max(self.max_skew_observed, skew)
            is_sync = skew <= self.sync_tolerance_sec
            if not is_sync:
                self.out_of_sync_count += 1
                logger.warning(
                    f"Timestamp skew ({skew*1000:.1f}ms) exceeds tolerance ({self.sync_tolerance_sec*1000:.1f}ms)"
                )

            bundle = SyncFrameBundle(
                timestamp_sec=max(t_rgb, t_thermal),
                time_skew_sec=skew,
                is_synchronized=is_sync,
                rgb_valid=True,
                thermal_valid=True,
                rgb_frame_id=rgb_frame.metadata.frame_id,
                thermal_frame_id=thermal_frame.metadata.frame_id,
                diagnostics={
                    "skew_ms": skew * 1000.0,
                    "max_skew_ms": self.max_skew_observed * 1000.0,
                },
            )
            return bundle, rgb_frame, thermal_frame

        # Case 2: Only RGB present
        elif rgb_valid:
            t_rgb = rgb_frame.metadata.timestamp_sec
            bundle = SyncFrameBundle(
                timestamp_sec=t_rgb,
                time_skew_sec=0.0,
                is_synchronized=False,  # degraded: single sensor
                rgb_valid=True,
                thermal_valid=False,
                rgb_frame_id=rgb_frame.metadata.frame_id,
                thermal_frame_id=-1,
                diagnostics={"missing_sensor": "cam_thermal"},
            )
            return bundle, rgb_frame, None

        # Case 3: Only Thermal present
        elif thermal_valid:
            t_thermal = thermal_frame.metadata.timestamp_sec
            bundle = SyncFrameBundle(
                timestamp_sec=t_thermal,
                time_skew_sec=0.0,
                is_synchronized=False,  # degraded: single sensor
                rgb_valid=False,
                thermal_valid=True,
                rgb_frame_id=-1,
                thermal_frame_id=thermal_frame.metadata.frame_id,
                diagnostics={"missing_sensor": "cam_rgb"},
            )
            return bundle, None, thermal_frame

        # Case 4: Neither sensor valid
        else:
            bundle = SyncFrameBundle(
                timestamp_sec=0.0,
                time_skew_sec=0.0,
                is_synchronized=False,
                rgb_valid=False,
                thermal_valid=False,
                rgb_frame_id=-1,
                thermal_frame_id=-1,
                diagnostics={"error": "both_sensors_unavailable"},
            )
            return bundle, None, None
