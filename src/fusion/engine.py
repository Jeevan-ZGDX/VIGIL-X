"""
Multimodal detection-level sensor fusion engine.
Associates RGB and Thermal detections, resolves conflicts, and balances uncertainty.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple
import uuid
import numpy as np

from src.calibration.spatial import SpatialCalibrator
from src.core.config import FusionConfig
from src.core.schemas import BoundingBox, Detection, SensorType
from src.fusion.uncertainty import UncertaintyModel

logger = logging.getLogger(__name__)


class SensorFusionEngine:
    """
    Detection-level sensor fusion engine.
    Supports FUSED, RGB_ONLY, and THERMAL_ONLY evaluation modes.
    """

    def __init__(self, config: FusionConfig, calibrator: Optional[SpatialCalibrator] = None):
        self.config = config
        self.calibrator = calibrator or SpatialCalibrator(config=None)
        self.uncertainty_model = UncertaintyModel()

    def fuse(
        self,
        rgb_detections: List[Detection],
        thermal_detections: List[Detection],
        mode: str = "fused",
    ) -> List[Detection]:
        """
        Fuse detections according to the specified operational mode.
        """
        # Mode 1: RGB-only evaluation
        if mode == "rgb_only":
            return self._process_single_sensor_list(rgb_detections, SensorType.RGB)

        # Mode 2: Thermal-only evaluation
        if mode == "thermal_only":
            return self._process_single_sensor_list(thermal_detections, SensorType.THERMAL)

        # Mode 3: Multimodal Fused operation
        return self._fuse_multimodal(rgb_detections, thermal_detections)

    def _process_single_sensor_list(
        self,
        detections: List[Detection],
        sensor_type: SensorType,
    ) -> List[Detection]:
        """Process detections from a single sensor with single-sensor uncertainty inflation."""
        results = []
        for det in detections:
            conf, uncert = self.uncertainty_model.penalize_single_sensor(
                det.confidence,
                det.uncertainty,
                penalty=self.config.single_sensor_confidence_penalty,
            )
            ground_pos = self.calibrator.estimate_pedestrian_ground_position(det.bbox)

            fused_det = Detection(
                detection_id=f"single_{det.detection_id}",
                frame_id=det.frame_id,
                timestamp_sec=det.timestamp_sec,
                sensor_type=sensor_type,
                sensor_id=det.sensor_id,
                class_name=det.class_name,
                confidence=conf,
                bbox=det.bbox,
                calibrated_ground_pos_m=ground_pos,
                uncertainty=uncert,
                source_detections=[det.detection_id],
                diagnostics={"mode": f"{sensor_type.value}_only", "raw_conf": det.confidence},
            )
            if fused_det.confidence >= self.config.min_fused_confidence:
                results.append(fused_det)
        return results

    def _fuse_multimodal(
        self,
        rgb_dets: List[Detection],
        th_dets: List[Detection],
    ) -> List[Detection]:
        """
        Associate detections across modalities using geometric IoU.
        Merges matching candidates and retains isolated candidates with uncertainty penalties.
        """
        fused_output: List[Detection] = []
        matched_rgb = set()
        matched_th = set()

        # Build IoU association matrix
        for i, r_det in enumerate(rgb_dets):
            best_iou = 0.0
            best_j = -1

            for j, t_det in enumerate(th_dets):
                if j in matched_th:
                    continue

                # Transform thermal bbox into RGB camera coordinates
                mapped_th_bbox = self.calibrator.transform_thermal_to_rgb_bbox(t_det.bbox)
                iou = r_det.bbox.iou(mapped_th_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            # Check if match satisfies threshold
            if best_iou >= self.config.spatial_iou_threshold and best_j >= 0:
                matched_rgb.add(i)
                matched_th.add(best_j)
                t_det = th_dets[best_j]

                # Fused confidence and uncertainty
                fused_conf, fused_uncert = self.uncertainty_model.fuse_confidence_and_uncertainty(
                    rgb_conf=r_det.confidence,
                    rgb_uncertainty=r_det.uncertainty,
                    thermal_conf=t_det.confidence,
                    thermal_uncertainty=t_det.uncertainty,
                    rgb_weight=self.config.rgb_weight,
                    thermal_weight=self.config.thermal_weight,
                )

                # Weighted bounding box interpolation
                w_r = r_det.confidence / (r_det.confidence + t_det.confidence)
                w_t = 1.0 - w_r
                mapped_th_bbox = self.calibrator.transform_thermal_to_rgb_bbox(t_det.bbox)

                fused_bbox = BoundingBox(
                    x1=w_r * r_det.bbox.x1 + w_t * mapped_th_bbox.x1,
                    y1=w_r * r_det.bbox.y1 + w_t * mapped_th_bbox.y1,
                    x2=w_r * r_det.bbox.x2 + w_t * mapped_th_bbox.x2,
                    y2=w_r * r_det.bbox.y2 + w_t * mapped_th_bbox.y2,
                )

                ground_pos = self.calibrator.estimate_pedestrian_ground_position(fused_bbox)

                fused_output.append(
                    Detection(
                        detection_id=f"fused_{uuid.uuid4().hex[:6]}",
                        frame_id=r_det.frame_id,
                        timestamp_sec=max(r_det.timestamp_sec, t_det.timestamp_sec),
                        sensor_type=SensorType.FUSED,
                        sensor_id="fused_rgb_thermal",
                        class_name="pedestrian",
                        confidence=fused_conf,
                        bbox=fused_bbox,
                        calibrated_ground_pos_m=ground_pos,
                        uncertainty=fused_uncert,
                        source_detections=[r_det.detection_id, t_det.detection_id],
                        diagnostics={
                            "association_iou": best_iou,
                            "rgb_conf": r_det.confidence,
                            "thermal_conf": t_det.confidence,
                            "source_modalities": ["RGB", "THERMAL"],
                        },
                    )
                )

        # Handle unmatched RGB detections (detected only by RGB, e.g. thermal occlusion or heat masking)
        for i, r_det in enumerate(rgb_dets):
            if i not in matched_rgb:
                conf, uncert = self.uncertainty_model.penalize_single_sensor(
                    r_det.confidence,
                    r_det.uncertainty,
                    penalty=self.config.single_sensor_confidence_penalty,
                )
                if conf >= self.config.min_fused_confidence:
                    ground_pos = self.calibrator.estimate_pedestrian_ground_position(r_det.bbox)
                    fused_output.append(
                        Detection(
                            detection_id=f"unmatched_rgb_{r_det.detection_id}",
                            frame_id=r_det.frame_id,
                            timestamp_sec=r_det.timestamp_sec,
                            sensor_type=SensorType.RGB,
                            sensor_id=r_det.sensor_id,
                            class_name=r_det.class_name,
                            confidence=conf,
                            bbox=r_det.bbox,
                            calibrated_ground_pos_m=ground_pos,
                            uncertainty=uncert,
                            source_detections=[r_det.detection_id],
                            diagnostics={"unmatched_source": "RGB"},
                        )
                    )

        # Handle unmatched Thermal detections (detected only by Thermal, e.g. darkness or glare)
        for j, t_det in enumerate(th_dets):
            if j not in matched_th:
                conf, uncert = self.uncertainty_model.penalize_single_sensor(
                    t_det.confidence,
                    t_det.uncertainty,
                    penalty=self.config.single_sensor_confidence_penalty,
                )
                if conf >= self.config.min_fused_confidence:
                    mapped_th_bbox = self.calibrator.transform_thermal_to_rgb_bbox(t_det.bbox)
                    ground_pos = self.calibrator.estimate_pedestrian_ground_position(mapped_th_bbox)
                    fused_output.append(
                        Detection(
                            detection_id=f"unmatched_thermal_{t_det.detection_id}",
                            frame_id=t_det.frame_id,
                            timestamp_sec=t_det.timestamp_sec,
                            sensor_type=SensorType.THERMAL,
                            sensor_id=t_det.sensor_id,
                            class_name=t_det.class_name,
                            confidence=conf,
                            bbox=mapped_th_bbox,
                            calibrated_ground_pos_m=ground_pos,
                            uncertainty=uncert,
                            source_detections=[t_det.detection_id],
                            diagnostics={"unmatched_source": "THERMAL"},
                        )
                    )

        return fused_output
