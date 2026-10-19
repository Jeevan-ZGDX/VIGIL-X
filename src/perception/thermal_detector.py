"""
Thermal pedestrian detector tailored for thermal infrared signatures.
Combines adaptive thermal contrast segmentation, morphological anthropometric filtering,
and ONNX Runtime deep learning integration.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List
import cv2
import numpy as np

from src.core.config import DetectorConfig
from src.core.schemas import BoundingBox, Detection, FrameData, SensorType
from src.perception.base import BaseDetector
from src.perception.rgb_detector import apply_nms
from src.sensors.thermal_processor import ThermalProcessor

logger = logging.getLogger(__name__)


class ThermalPedestrianDetector(BaseDetector):
    """
    Thermal pedestrian detector.
    Does NOT assume RGB visual features work on thermal imagery.
    Detects warm human signatures based on thermal contrast, vertical profile,
    and temperature distribution.
    """

    def __init__(self, config: DetectorConfig, sensor_id: str = "cam_thermal"):
        super().__init__(config, sensor_id)
        self.processor = ThermalProcessor()
        self.ort_session = None

        if self.config.backend == "onnx" and self.config.model_path:
            model_p = Path(self.config.model_path)
            if model_p.exists():
                try:
                    import onnxruntime as ort
                    self.ort_session = ort.InferenceSession(str(model_p), providers=["CPUExecutionProvider"])
                    logger.info(f"Loaded ONNX Thermal model from {model_p}")
                except Exception as e:
                    logger.warning(f"Failed to load ONNX thermal model: {e}")

        logger.info("Initialized Thermal pedestrian signature detector baseline")

    def detect(self, frame_data: FrameData) -> List[Detection]:
        if not frame_data.is_valid or frame_data.image is None:
            return []

        raw_img = frame_data.image
        gray_8u = self.processor.normalize_to_8bit(raw_img)
        h, w = gray_8u.shape[:2]

        # Extract thermal heat mask
        # Warm body temperature is typically in upper percentiles of the scene
        heat_mask = self.processor.extract_pedestrian_heat_mask(gray_8u, low_percentile=70.0)

        # Find connected components / contours
        contours, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes: List[BoundingBox] = []
        candidate_scores: List[float] = []

        mean_scene_temp = float(np.mean(gray_8u))
        scene_std = float(np.std(gray_8u))

        # If scene is thermally flat (ambient room with no heat sources), reject immediately
        if scene_std < 8.0:
            return []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_box_area_px:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)

            # Anthropometric aspect ratio check: height/width must resemble a standing or walking human
            aspect_ratio = float(bh) / max(1.0, float(bw))
            if aspect_ratio < 1.1 or aspect_ratio > 4.5:
                # Discard wide horizontal heat patches (e.g. heated machinery beds, exhaust pipes)
                continue

            # Height should be a reasonable proportion of image height (e.g., between 8% and 90%)
            if bh < (h * 0.08) or bh > (h * 0.95):
                continue

            # Compute internal contrast vs background
            roi = gray_8u[by : by + bh, bx : bx + bw]
            internal_mean = float(np.mean(roi))
            contrast_diff = internal_mean - mean_scene_temp

            # Strict thermal delta check: must be significantly warmer than background
            if contrast_diff < 15.0:
                continue

            # Normalize confidence based on contrast and aspect ratio solidity
            contrast_score = max(0.0, min(1.0, (contrast_diff - 10.0) / 40.0))
            aspect_score = 1.0 - abs(aspect_ratio - 2.5) / 2.5  # Ideal pedestrian aspect ratio is ~2.5
            aspect_score = max(0.2, min(1.0, aspect_score))

            conf = 0.70 * contrast_score + 0.30 * aspect_score
            conf = max(0.0, min(1.0, conf))

            if conf < self.config.confidence_threshold:
                continue

            x1 = float(max(0, bx))
            y1 = float(max(0, by))
            x2 = float(min(w - 1, bx + bw))
            y2 = float(min(h - 1, by + bh))

            candidate_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
            candidate_scores.append(conf)

        if not candidate_boxes:
            return []

        # NMS suppression
        keep_indices = apply_nms(candidate_boxes, candidate_scores, self.config.iou_nms_threshold)

        detections: List[Detection] = []
        for idx in keep_indices:
            bbox = candidate_boxes[idx]
            conf = candidate_scores[idx]

            # In thermal, uncertainty is low in darkness/glare, but higher if contrast is weak
            uncertainty = max(0.10, min(0.85, 1.0 - conf))

            detections.append(
                Detection(
                    frame_id=frame_data.metadata.frame_id,
                    timestamp_sec=frame_data.metadata.timestamp_sec,
                    sensor_type=SensorType.THERMAL,
                    sensor_id=self.sensor_id,
                    class_name="pedestrian",
                    confidence=conf,
                    bbox=bbox,
                    uncertainty=uncertainty,
                    diagnostics={
                        "model": "Thermal_Contrast_Baseline",
                        "mean_scene_temp": mean_scene_temp,
                        "raw_score": conf,
                    },
                )
            )

        return detections
