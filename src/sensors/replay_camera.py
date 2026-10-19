"""
Replay sensor implementation for recorded videos, image sequences, or synthetic arrays.
Supports deterministic step replay, simulated timestamp progression, frame-rate pacing,
and fault injection (stale frames, drops, disconnections).
"""

from __future__ import annotations
import logging
from pathlib import Path
import time
from typing import List, Optional, Union
import cv2
import numpy as np

from src.core.config import SensorConfig
from src.core.schemas import SensorHealthStatus, SensorType
from src.sensors.base import BaseSensor

logger = logging.getLogger(__name__)


class ReplayCameraSensor(BaseSensor):
    """
    Replays recorded or synthetic sensor frames for reproducible testing and evaluation.
    """

    def __init__(
        self,
        config: SensorConfig,
        in_memory_frames: Optional[List[np.ndarray]] = None,
        loop: bool = True,
        pace_realtime: bool = False,
    ):
        super().__init__(config)
        self.in_memory_frames = in_memory_frames
        self.loop = loop
        self.pace_realtime = pace_realtime
        self.current_index: int = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.image_files: List[Path] = []
        self._fault_simulate_drop: bool = False
        self._fault_simulate_stale: bool = False
        self._fault_stale_duration: float = 0.0

    def connect(self) -> bool:
        """Connect to in-memory frames, video file, or image sequence directory."""
        self.disconnect()
        self.current_index = 0

        # Mode 1: In-memory array sequence provided
        if self.in_memory_frames is not None and len(self.in_memory_frames) > 0:
            self.is_connected = True
            self.health_status = SensorHealthStatus.HEALTHY
            logger.info(f"Replay [{self.sensor_id}] connected with {len(self.in_memory_frames)} in-memory frames")
            return True

        # Mode 2: Replay path provided in config
        if self.config.replay_path:
            path = Path(self.config.replay_path)
            if not path.exists():
                logger.error(f"Replay path does not exist: {path}")
                self.is_connected = False
                self.health_status = SensorHealthStatus.ERROR
                return False

            if path.is_dir():
                all_images = sorted(
                    [p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".npy", ".tiff")]
                )
                # Filter by sensor prefix if multiple sensor streams are in the same folder
                pref = "rgb" if self.sensor_type == SensorType.RGB else "th"
                matched_files = [p for p in all_images if pref in p.name.lower()]
                self.image_files = matched_files if matched_files else all_images

                if not self.image_files:
                    logger.error(f"No image files found in {path}")
                    self.is_connected = False
                    self.health_status = SensorHealthStatus.ERROR
                    return False
                self.is_connected = True
                self.health_status = SensorHealthStatus.HEALTHY
                return True
            else:
                self.cap = cv2.VideoCapture(str(path))
                if not self.cap.isOpened():
                    logger.error(f"Failed to open video replay file: {path}")
                    self.is_connected = False
                    self.health_status = SensorHealthStatus.ERROR
                    return False
                self.is_connected = True
                self.health_status = SensorHealthStatus.HEALTHY
                return True

        # Mode 3: No replay path or memory frames -> create black default fallback sequence
        logger.warning(f"Replay [{self.sensor_id}] initialized with synthetic blank sequence")
        blank = np.zeros(
            (self.config.height, self.config.width, 3 if self.config.sensor_type == "RGB" else 1),
            dtype=np.uint8,
        )
        self.in_memory_frames = [blank]
        self.is_connected = True
        self.health_status = SensorHealthStatus.HEALTHY
        return True

    def disconnect(self) -> None:
        """Release replay resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        self.health_status = SensorHealthStatus.DISCONNECTED

    def inject_drop(self, enable: bool = True) -> None:
        """Fault injection: force the sensor to drop subsequent frame(s)."""
        self._fault_simulate_drop = enable

    def inject_stale(self, enable: bool = True, stale_seconds: float = 1.0) -> None:
        """Fault injection: simulate stale frame timestamps."""
        self._fault_simulate_stale = enable
        self._fault_stale_duration = stale_seconds

    def _capture_raw(self) -> Optional[np.ndarray]:
        if not self.is_connected:
            return None

        if self._fault_simulate_drop:
            # Emulate dropped frame without progressing index
            return None

        if self.pace_realtime and self.config.target_fps > 0:
            frame_period = 1.0 / self.config.target_fps
            time.sleep(frame_period)

        frame = None

        # From in-memory
        if self.in_memory_frames:
            if self.current_index >= len(self.in_memory_frames):
                if self.loop:
                    self.current_index = 0
                else:
                    return None
            frame = self.in_memory_frames[self.current_index]
            self.current_index += 1

        # From image directory
        elif self.image_files:
            if self.current_index >= len(self.image_files):
                if self.loop:
                    self.current_index = 0
                else:
                    return None
            img_path = self.image_files[self.current_index]
            self.current_index += 1
            if img_path.suffix.lower() == ".npy":
                frame = np.load(str(img_path))
            else:
                frame = cv2.imread(str(img_path))

        # From video file
        elif self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret:
                    return None
            self.current_index += 1

        if frame is None:
            return None

        # Check / enforce dimensions
        if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
            frame = cv2.resize(frame, (self.config.width, self.config.height))

        return frame

    def read_frame(self):
        """Override to support stale timestamp fault injection when requested."""
        frame_data = super().read_frame()
        if self._fault_simulate_stale and frame_data.is_valid:
            # Artificially age the timestamp by subtract stale_seconds
            frame_data.metadata.timestamp_sec -= self._fault_stale_duration
            frame_data.metadata.timestamp_ns -= int(self._fault_stale_duration * 1e9)
            frame_data.metadata.stale_duration_sec = self._fault_stale_duration
            self.last_successful_read_time -= self._fault_stale_duration
            self.check_health()
        return frame_data
