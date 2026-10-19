"""
Abstract base detector interface for sensor-specific pedestrian perception.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from src.core.schemas import Detection, FrameData
from src.core.config import DetectorConfig


class BaseDetector(ABC):
    """Abstract interface for sensor-specific pedestrian detection."""

    def __init__(self, config: DetectorConfig, sensor_id: str):
        self.config = config
        self.sensor_id = sensor_id
        self.model_identifier = config.backend

    @abstractmethod
    def detect(self, frame_data: FrameData) -> List[Detection]:
        """Process a valid sensor frame and return candidate pedestrian detections."""
        pass
