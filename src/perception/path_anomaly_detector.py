"""
Path anomaly and multi-class obstacle detector.
Detects and classifies:
- Pedestrians / Intruders (human targets)
- Potholes / Ground surface cavities in the vehicle path corridor
- Paper / Flat debris on the travel path
- Obstacles / Cargo (boxes, backpacks, equipment, furniture)
"""

from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.core.config import DetectorConfig
from src.core.schemas import BoundingBox, Detection, FrameData, SensorType

logger = logging.getLogger(__name__)


class PathAnomalyDetector:
    """
    Combined anomaly perception engine.
    Integrates deep learning multi-class object detection and ground-plane surface
    defect analysis (potholes, paper/debris, obstacles).
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.ort_session = None
        self.input_name = "images"
        self.input_type = "tensor(float32)"

        model_p = Path("models/yolov5n.onnx")
        if model_p.exists():
            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(str(model_p), providers=["CPUExecutionProvider"])
                self.input_name = self.ort_session.get_inputs()[0].name
                self.input_type = self.ort_session.get_inputs()[0].type
                logger.info(f"PathAnomalyDetector loaded ONNX YOLOv5n model from {model_p}")
            except Exception as e:
                logger.warning(f"Could not load ONNX model in PathAnomalyDetector: {e}")

        # COCO class mapping for industrial path obstacles
        # 0: person
        # 24: backpack, 26: handbag, 28: suitcase
        # 39: bottle, 41: cup, 56: chair, 62: tv, 63: laptop
        # 73: book (paper/flat document)
        self.obstacle_classes = {
            24: "obstacle",
            26: "obstacle",
            28: "obstacle",
            39: "obstacle",
            41: "obstacle",
            56: "obstacle",
            62: "obstacle",
            63: "obstacle",
            73: "paper",
        }

    def detect_anomalies(
        self,
        frame_data: FrameData,
        calibrator=None,
    ) -> List[Detection]:
        """
        Detect all path anomalies, classifying pedestrians, potholes, paper, and obstacles.
        """
        if not frame_data.is_valid or frame_data.image is None or frame_data.image.size == 0:
            return []

        img = frame_data.image
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        detections: List[Detection] = []

        # 1. Multi-class Deep Learning Detections (YOLOv5n)
        if self.ort_session is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if (img.ndim == 3 and img.shape[2] == 3) else cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            target_dtype = np.float16 if "float16" in getattr(self, "input_type", "") else np.float32
            input_tensor = cv2.resize(rgb, (640, 640)).astype(target_dtype) / target_dtype(255.0)
            input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, ...]

            raw_out = self.ort_session.run(None, {self.input_name: input_tensor})[0][0].astype(np.float32)
            obj_conf = raw_out[:, 4]
            scale_x = w / 640.0
            scale_y = h / 640.0

            # A. Pedestrian (Class 0)
            person_scores = obj_conf * raw_out[:, 5]
            p_mask = person_scores >= self.config.confidence_threshold
            for row, score in zip(raw_out[p_mask], person_scores[p_mask]):
                cx, cy, bw, bh = row[0:4]
                x1 = max(0.0, float((cx - bw / 2.0) * scale_x))
                y1 = max(0.0, float((cy - bh / 2.0) * scale_y))
                x2 = min(float(w - 1), float((cx + bw / 2.0) * scale_x))
                y2 = min(float(h - 1), float((cy + bh / 2.0) * scale_y))
                if (x2 - x1) * (y2 - y1) >= self.config.min_box_area_px:
                    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    ground_pos = calibrator.estimate_pedestrian_ground_position(bbox) if calibrator else (0.0, 4.0)
                    detections.append(
                        Detection(
                            frame_id=frame_data.metadata.frame_id,
                            timestamp_sec=frame_data.metadata.timestamp_sec,
                            sensor_type=SensorType.RGB,
                            sensor_id=frame_data.metadata.sensor_id,
                            class_name="pedestrian",
                            confidence=float(score),
                            bbox=bbox,
                            calibrated_ground_pos_m=ground_pos,
                            uncertainty=max(0.1, 1.0 - float(score)),
                            diagnostics={"model": "YOLOv5n_ONNX", "class": "person"},
                        )
                    )

            # B. Other COCO Objects (Obstacles & Paper)
            for cls_idx, class_label in self.obstacle_classes.items():
                cls_scores = obj_conf * raw_out[:, 5 + cls_idx]
                obs_mask = cls_scores >= max(0.30, self.config.confidence_threshold - 0.10)
                for row, score in zip(raw_out[obs_mask], cls_scores[obs_mask]):
                    cx, cy, bw, bh = row[0:4]
                    x1 = max(0.0, float((cx - bw / 2.0) * scale_x))
                    y1 = max(0.0, float((cy - bh / 2.0) * scale_y))
                    x2 = min(float(w - 1), float((cx + bw / 2.0) * scale_x))
                    y2 = min(float(h - 1), float((cy + bh / 2.0) * scale_y))
                    if (x2 - x1) * (y2 - y1) >= 150.0:
                        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                        ground_pos = calibrator.estimate_pedestrian_ground_position(bbox) if calibrator else (0.0, 3.0)
                        detections.append(
                            Detection(
                                frame_id=frame_data.metadata.frame_id,
                                timestamp_sec=frame_data.metadata.timestamp_sec,
                                sensor_type=SensorType.RGB,
                                sensor_id=frame_data.metadata.sensor_id,
                                class_name=class_label,
                                confidence=float(score),
                                bbox=bbox,
                                calibrated_ground_pos_m=ground_pos,
                                uncertainty=max(0.15, 1.0 - float(score)),
                                diagnostics={"model": "YOLOv5n_COCO", "coco_id": cls_idx},
                            )
                        )

        # 2. Ground Surface Corridor Analysis (Potholes & Paper on Floor)
        # Vehicle corridor is the lower half trapezoid of the screen
        corridor_y_start = int(h * 0.55)
        ground_roi = gray[corridor_y_start:h, :]
        roi_h, roi_w = ground_roi.shape[:2]

        if roi_h > 20 and roi_w > 20:
            # Potholes: Dark sunken depressions on the ground plane
            # Adaptive threshold for dark recessed contours
            mean_ground = float(np.mean(ground_roi))
            dark_thresh = max(10, int(mean_ground - 38.0))
            _, pothole_mask = cv2.threshold(ground_roi, dark_thresh, 255, cv2.THRESH_BINARY_INV)

            # Remove noise
            kernel_pothole = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            pothole_clean = cv2.morphologyEx(pothole_mask, cv2.MORPH_OPEN, kernel_pothole)

            contours_pothole, _ = cv2.findContours(pothole_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_pothole:
                area = cv2.contourArea(cnt)
                if 400.0 < area < (roi_w * roi_h * 0.35):
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect = float(bw) / max(1.0, float(bh))
                    # Potholes on ground plane perspective typically have wider horizontal span (aspect 1.1 to 4.0)
                    if 0.8 < aspect < 4.5:
                        x1 = float(max(0, bx))
                        y1 = float(corridor_y_start + by)
                        x2 = float(min(w - 1, bx + bw))
                        y2 = float(min(h - 1, corridor_y_start + by + bh))
                        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                        ground_pos = calibrator.estimate_pedestrian_ground_position(bbox) if calibrator else (0.0, 2.0)
                        detections.append(
                            Detection(
                                frame_id=frame_data.metadata.frame_id,
                                timestamp_sec=frame_data.metadata.timestamp_sec,
                                sensor_type=SensorType.RGB,
                                sensor_id="path_analyzer",
                                class_name="pothole",
                                confidence=0.78,
                                bbox=bbox,
                                calibrated_ground_pos_m=ground_pos,
                                uncertainty=0.25,
                                diagnostics={"anomaly_type": "ground_depression", "area": area},
                            )
                        )

            # Paper / Flat surface debris: Bright flat rectangular planar regions on the floor
            bright_thresh = min(245, int(mean_ground + 50.0))
            _, paper_mask = cv2.threshold(ground_roi, bright_thresh, 255, cv2.THRESH_BINARY)
            kernel_paper = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            paper_clean = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel_paper)

            contours_paper, _ = cv2.findContours(paper_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_paper:
                area = cv2.contourArea(cnt)
                if 300.0 < area < (roi_w * roi_h * 0.20):
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    # Check rectangularity
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / max(1.0, hull_area)
                    if solidity > 0.70:
                        x1 = float(max(0, bx))
                        y1 = float(corridor_y_start + by)
                        x2 = float(min(w - 1, bx + bw))
                        y2 = float(min(h - 1, corridor_y_start + by + bh))
                        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                        ground_pos = calibrator.estimate_pedestrian_ground_position(bbox) if calibrator else (0.0, 2.5)
                        detections.append(
                            Detection(
                                frame_id=frame_data.metadata.frame_id,
                                timestamp_sec=frame_data.metadata.timestamp_sec,
                                sensor_type=SensorType.RGB,
                                sensor_id="path_analyzer",
                                class_name="paper",
                                confidence=0.74,
                                bbox=bbox,
                                calibrated_ground_pos_m=ground_pos,
                                uncertainty=0.28,
                                diagnostics={"anomaly_type": "surface_paper_debris", "solidity": solidity},
                            )
                        )

        return detections
