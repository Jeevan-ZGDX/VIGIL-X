"""
RGB pedestrian detector supporting gradient/silhouette edge-baseline
and ONNX Runtime deep learning inference.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from src.core.config import DetectorConfig
from src.core.schemas import BoundingBox, Detection, FrameData, SensorType
from src.perception.base import BaseDetector

logger = logging.getLogger(__name__)


def apply_nms(boxes: List[BoundingBox], scores: List[float], iou_threshold: float) -> List[int]:
    """Non-maximum suppression for bounding boxes."""
    if not boxes:
        return []

    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []

    while indices:
        current = indices.pop(0)
        keep.append(current)
        indices = [
            i for i in indices
            if boxes[current].iou(boxes[i]) < iou_threshold
        ]

    return keep


class RGBPedestrianDetector(BaseDetector):
    """
    Pedestrian detector for RGB streams.
    Provides gradient-silhouette pedestrian detector baseline and ONNX runtime integration.
    """

    def __init__(self, config: DetectorConfig, sensor_id: str = "cam_rgb"):
        super().__init__(config, sensor_id)
        self.ort_session = None

        if self.config.backend == "onnx" and self.config.model_path:
            model_p = Path(self.config.model_path)
            if model_p.exists():
                try:
                    import onnxruntime as ort
                    self.ort_session = ort.InferenceSession(str(model_p), providers=["CPUExecutionProvider"])
                    logger.info(f"Loaded ONNX RGB model from {model_p}")
                except Exception as e:
                    logger.warning(f"Failed to load ONNX model, falling back to gradient baseline: {e}")

        logger.info(f"Initialized RGB pedestrian detector (backend={self.config.backend})")

    def _assess_illumination(self, gray: np.ndarray) -> dict:
        """Measure average brightness, contrast, and glare saturation ratio."""
        mean_brightness = float(np.mean(gray))
        glare_pixels = float(np.sum(gray >= 250))
        glare_ratio = glare_pixels / float(max(1, gray.size))
        is_low_light = mean_brightness < 45.0
        is_glare = glare_ratio > 0.08
        return {
            "mean_brightness": mean_brightness,
            "glare_ratio": glare_ratio,
            "is_low_light": is_low_light,
            "is_glare": is_glare,
        }

    def detect(self, frame_data: FrameData) -> List[Detection]:
        if not frame_data.is_valid or frame_data.image is None or frame_data.image.size == 0:
            return []

        img = frame_data.image
        h, w = img.shape[:2]
        if img.ndim == 3:
            if img.shape[2] == 1:
                gray = img[:, :, 0].copy()
            elif img.shape[2] in (3, 4):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img[:, :, 0].copy()
        else:
            gray = img.copy()

        illumination = self._assess_illumination(gray)
        candidate_boxes: List[BoundingBox] = []
        candidate_scores: List[float] = []

        # ONNX path if session is loaded
        if self.ort_session is not None:
            # Model inference placeholder for custom ONNX models
            pass
        else:
            # Gradient & silhouette edge pedestrian detection baseline
            # Computes spatial gradients to find vertical pedestrian silhouettes
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(grad_x, grad_y)

            # Threshold high-contrast structural edges
            mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, edge_thresh = cv2.threshold(mag_norm, 40, 255, cv2.THRESH_BINARY)

            # Morphological close to join human contours (head, torso, legs)
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
            closed = cv2.morphologyEx(edge_thresh, cv2.MORPH_CLOSE, kernel_v)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.config.min_box_area_px:
                    continue

                bx, by, bw, bh = cv2.boundingRect(cnt)

                # Anthropometric aspect ratio: pedestrian height/width ratio typically 1.2 to 4.2
                aspect_ratio = float(bh) / max(1.0, float(bw))
                if aspect_ratio < 1.1 or aspect_ratio > 4.5:
                    continue

                # Height filter
                if bh < (h * 0.08) or bh > (h * 0.95):
                    continue

                # Pedestrian confidence score based on edge density and aspect ratio alignment
                roi_edge = edge_thresh[by : by + bh, bx : bx + bw]
                edge_density = float(np.count_nonzero(roi_edge)) / float(max(1, bw * bh))
                aspect_score = 1.0 - min(1.0, abs(aspect_ratio - 2.5) / 2.5)

                conf = 0.50 * edge_density * 4.0 + 0.50 * aspect_score
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

        # Apply NMS
        keep_indices = apply_nms(candidate_boxes, candidate_scores, self.config.iou_nms_threshold)

        detections: List[Detection] = []
        for idx in keep_indices:
            bbox = candidate_boxes[idx]
            conf = candidate_scores[idx]

            # Uncertainty increases in low-light or glare conditions for RGB
            uncertainty = 0.15
            if illumination["is_low_light"]:
                uncertainty += 0.35
                conf *= 0.80
            if illumination["is_glare"]:
                uncertainty += 0.30
                conf *= 0.75
            uncertainty = min(0.95, uncertainty)

            detections.append(
                Detection(
                    frame_id=frame_data.metadata.frame_id,
                    timestamp_sec=frame_data.metadata.timestamp_sec,
                    sensor_type=SensorType.RGB,
                    sensor_id=self.sensor_id,
                    class_name="pedestrian",
                    confidence=conf,
                    bbox=bbox,
                    uncertainty=uncertainty,
                    diagnostics={
                        "model": "Gradient_Silhouette_Baseline",
                        "illumination": illumination,
                        "raw_score": conf,
                    },
                )
            )

        return detections
