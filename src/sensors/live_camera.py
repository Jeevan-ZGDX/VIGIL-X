"""
Live camera capture using OpenCV VideoCapture with hardware reconnection handling.
"""

from __future__ import annotations
import logging
from typing import Optional
import cv2
import numpy as np

from src.core.config import SensorConfig
from src.core.schemas import SensorHealthStatus
from src.sensors.base import BaseSensor

logger = logging.getLogger(__name__)


class LiveCameraSensor(BaseSensor):
    """V4L2 / UVC live hardware camera sensor adapter."""

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.cap: Optional[cv2.VideoCapture] = None

    def connect(self) -> bool:
        """Initialize and configure OpenCV VideoCapture."""
        self.disconnect()
        device = self.config.device_index if self.config.device_index is not None else 0
        logger.info(f"Connecting to live camera [{self.sensor_id}] on device {device}")

        try:
            self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                # Fallback to default backend if V4L2 direct fails
                self.cap = cv2.VideoCapture(device)

            if not self.cap.isOpened():
                logger.warning(f"Could not open camera device {device}")
                self.is_connected = False
                self.health_status = SensorHealthStatus.DISCONNECTED
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)

            # Test-read one frame to verify active pipeline
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning(f"Camera opened but failed initial frame read on device {device}")
                self.is_connected = False
                self.health_status = SensorHealthStatus.ERROR
                return False

            self.is_connected = True
            self.health_status = SensorHealthStatus.HEALTHY
            logger.info(f"Camera [{self.sensor_id}] connected successfully ({self.config.width}x{self.config.height})")
            return True
        except Exception as e:
            logger.error(f"Error opening camera {device}: {e}")
            self.is_connected = False
            self.health_status = SensorHealthStatus.ERROR
            return False

    def disconnect(self) -> None:
        """Release camera hardware."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_connected = False
        self.health_status = SensorHealthStatus.DISCONNECTED

    def _capture_raw(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            self.is_connected = False
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning(f"Failed to read frame from camera [{self.sensor_id}]")
            return None

        # Resize if dimensions don't match requested
        if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
            frame = cv2.resize(frame, (self.config.width, self.config.height))

        return frame
