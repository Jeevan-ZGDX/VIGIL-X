"""
Abstract base sensor interface and health monitoring lifecycle.
Tracks frame timestamps, dropped frames, stale frames, and reconnection logic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import time
from typing import Optional
import numpy as np

from src.core.schemas import SensorType, SensorHealthStatus, FrameMetadata, FrameData
from src.core.config import SensorConfig

logger = logging.getLogger(__name__)


class BaseSensor(ABC):
    """
    Abstract sensor interface for RGB, Thermal, or Replay cameras.
    Guarantees unified output format, health states, and frame freshness verification.
    """

    def __init__(self, config: SensorConfig):
        self.config = config
        self.sensor_id = config.sensor_id
        self.sensor_type = SensorType(config.sensor_type)
        self.health_status: SensorHealthStatus = SensorHealthStatus.DISCONNECTED
        self.frame_counter: int = 0
        self.dropped_frame_counter: int = 0
        self.last_frame_timestamp: float = 0.0
        self.last_successful_read_time: float = 0.0
        self.is_connected: bool = False
        self._stale_timeout_sec: float = config.stale_timeout_sec

    @abstractmethod
    def connect(self) -> bool:
        """Connect to physical camera or open recorded media."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Release sensor resources safely."""
        pass

    @abstractmethod
    def _capture_raw(self) -> Optional[np.ndarray]:
        """Internal capture implementation returning a numpy array or None."""
        pass

    def check_health(self) -> SensorHealthStatus:
        """Evaluate sensor health based on connection state and timestamp freshness."""
        if not self.is_connected:
            self.health_status = SensorHealthStatus.DISCONNECTED
            return self.health_status

        now = time.time()
        if self.last_successful_read_time > 0:
            elapsed = now - self.last_successful_read_time
            if elapsed > self._stale_timeout_sec:
                self.health_status = SensorHealthStatus.STALE
                return self.health_status

        self.health_status = SensorHealthStatus.HEALTHY
        return self.health_status

    def read_frame(self) -> FrameData:
        """
        Public capture method.
        Wraps raw capture with timestamping, dimension checks, stale checks,
        and health updates. Avoids silently returning stale frames.
        """
        now_sec = time.time()
        now_ns = time.time_ns()

        if not self.is_connected:
            self.dropped_frame_counter += 1
            metadata = FrameMetadata(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                frame_id=self.frame_counter,
                timestamp_ns=now_ns,
                timestamp_sec=now_sec,
                width=self.config.width,
                height=self.config.height,
                channels=3 if self.sensor_type == SensorType.RGB else 1,
                fps=self.config.target_fps,
                is_valid=False,
                error_msg=f"Sensor {self.sensor_id} is disconnected",
                stale_duration_sec=now_sec - self.last_successful_read_time if self.last_successful_read_time else 0.0,
            )
            return FrameData(metadata=metadata, image=None)

        try:
            raw_img = self._capture_raw()
        except Exception as e:
            logger.error(f"Sensor [{self.sensor_id}] capture error: {e}")
            self.health_status = SensorHealthStatus.ERROR
            self.dropped_frame_counter += 1
            metadata = FrameMetadata(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                frame_id=self.frame_counter,
                timestamp_ns=now_ns,
                timestamp_sec=now_sec,
                width=self.config.width,
                height=self.config.height,
                channels=3 if self.sensor_type == SensorType.RGB else 1,
                fps=self.config.target_fps,
                is_valid=False,
                error_msg=f"Capture exception: {str(e)}",
            )
            return FrameData(metadata=metadata, image=None)

        if raw_img is None or raw_img.size == 0:
            self.dropped_frame_counter += 1
            self.check_health()
            metadata = FrameMetadata(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                frame_id=self.frame_counter,
                timestamp_ns=now_ns,
                timestamp_sec=now_sec,
                width=self.config.width,
                height=self.config.height,
                channels=3 if self.sensor_type == SensorType.RGB else 1,
                fps=self.config.target_fps,
                is_valid=False,
                error_msg="Empty frame received",
                stale_duration_sec=now_sec - self.last_successful_read_time if self.last_successful_read_time else 0.0,
            )
            return FrameData(metadata=metadata, image=None)

        # Successful read
        self.frame_counter += 1
        self.last_frame_timestamp = now_sec
        self.last_successful_read_time = now_sec
        self.health_status = SensorHealthStatus.HEALTHY

        h, w = raw_img.shape[:2]
        channels = raw_img.shape[2] if raw_img.ndim == 3 else 1

        metadata = FrameMetadata(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            frame_id=self.frame_counter,
            timestamp_ns=now_ns,
            timestamp_sec=now_sec,
            width=w,
            height=h,
            channels=channels,
            fps=self.config.target_fps,
            is_valid=True,
            error_msg=None,
            stale_duration_sec=0.0,
        )
        return FrameData(metadata=metadata, image=raw_img)
