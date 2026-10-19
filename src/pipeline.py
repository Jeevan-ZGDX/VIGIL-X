"""
End-to-end Pedestrian Safety Fusion Pipeline Orchestrator.
Coordinates sensors, detectors, synchronization, fusion, tracking, risk, and integration.
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.calibration.spatial import SpatialCalibrator
from src.calibration.synchronizer import SensorSynchronizer, SyncFrameBundle
from src.core.config import SystemConfig
from src.core.schemas import (
    AlertState,
    Detection,
    FrameData,
    MachineTelemetry,
    RiskAssessment,
    SensorHealthStatus,
    SensorType,
    SystemHealthReport,
    Track,
)
from src.fusion.engine import SensorFusionEngine
from src.integration.simulator import MachineIntegrationSimulator
from src.perception.rgb_detector import RGBPedestrianDetector
from src.perception.thermal_detector import ThermalPedestrianDetector
from src.risk.engine import CollisionRiskEngine
from src.risk.state_machine import AlertStateMachine
from src.sensors.base import BaseSensor
from src.sensors.live_camera import LiveCameraSensor
from src.sensors.replay_camera import ReplayCameraSensor
from src.sensors.thermal_processor import ThermalProcessor
from src.storage.event_logger import StructuredEventLogger
from src.tracking.tracker import MultiPedestrianTracker

logger = logging.getLogger(__name__)


class SafetyPipeline:
    """
    Main orchestrator integrating all stages of the pedestrian safety system.
    Supports live camera capture and reproducible replay feeds.
    """

    def __init__(
        self,
        config: SystemConfig,
        rgb_sensor: Optional[BaseSensor] = None,
        thermal_sensor: Optional[BaseSensor] = None,
        machine_adapter: Optional[MachineIntegrationSimulator] = None,
    ):
        self.config = config

        # 1. Sensors
        if rgb_sensor is not None:
            self.rgb_sensor = rgb_sensor
        elif config.rgb_sensor.mode == "live":
            self.rgb_sensor = LiveCameraSensor(config.rgb_sensor)
        else:
            self.rgb_sensor = ReplayCameraSensor(config.rgb_sensor)

        if thermal_sensor is not None:
            self.thermal_sensor = thermal_sensor
        elif config.thermal_sensor.mode == "live":
            self.thermal_sensor = LiveCameraSensor(config.thermal_sensor)
        else:
            self.thermal_sensor = ReplayCameraSensor(config.thermal_sensor)

        self.thermal_processor = ThermalProcessor()

        # 2. Calibration & Synchronizer
        self.synchronizer = SensorSynchronizer(config.calibration)
        self.calibrator = SpatialCalibrator(
            config.calibration,
            image_width=config.rgb_sensor.width,
            image_height=config.rgb_sensor.height,
        )

        # 3. Detectors
        self.rgb_detector = RGBPedestrianDetector(config.rgb_detector, config.rgb_sensor.sensor_id)
        self.thermal_detector = ThermalPedestrianDetector(config.thermal_detector, config.thermal_sensor.sensor_id)

        # 4. Fusion Engine
        self.fusion_engine = SensorFusionEngine(config.fusion, self.calibrator)

        # 5. Tracker
        self.tracker = MultiPedestrianTracker(config.tracking)

        # 6. Risk Engine & State Machine
        self.risk_engine = CollisionRiskEngine(config.zones, config.machine)
        self.state_machine = AlertStateMachine(config.state_machine, self.risk_engine)

        # 7. Machine Adapter & Storage
        self.machine_adapter = machine_adapter or MachineIntegrationSimulator(config.machine)
        self.event_logger = StructuredEventLogger(config.event_log_path)

        # Operational states & snapshots
        self.is_running = False
        self.frame_index = 0
        self.last_sync_bundle: Optional[SyncFrameBundle] = None
        self.last_rgb_frame: Optional[FrameData] = None
        self.last_thermal_frame: Optional[FrameData] = None
        self.last_fused_detections: List[Detection] = []
        self.last_tracks: List[Track] = []
        self.last_assessment: Optional[RiskAssessment] = None
        self.last_health_report: Optional[SystemHealthReport] = None
        self.fps_tracker = 0.0
        self._last_process_time = time.time()

    def start(self) -> bool:
        """Start and connect all sensors."""
        rgb_ok = self.rgb_sensor.connect()
        th_ok = self.thermal_sensor.connect()
        self.is_running = True
        logger.info(f"Pipeline started (RGB: {rgb_ok}, Thermal: {th_ok})")
        return rgb_ok and th_ok

    def stop(self) -> None:
        """Gracefully release camera resources and halt pipeline."""
        self.is_running = False
        self.rgb_sensor.disconnect()
        self.thermal_sensor.disconnect()
        logger.info("Pipeline stopped and camera resources released")

    def process_frame(
        self,
        telemetry_override: Optional[MachineTelemetry] = None,
        fusion_mode: str = "fused",
    ) -> Tuple[RiskAssessment, SystemHealthReport]:
        """
        Execute one full capture-detection-fusion-tracking-risk cycle.
        Returns the RiskAssessment and SystemHealthReport.
        """
        t_start = time.perf_counter()
        now_sec = time.time()
        self.frame_index += 1

        # Step 1: Capture Frames
        t_cap_start = time.perf_counter()
        raw_rgb = self.rgb_sensor.read_frame()
        raw_thermal = self.thermal_sensor.read_frame()
        latency_cap = (time.perf_counter() - t_cap_start) * 1000.0

        # Step 2: Temporal Synchronization
        sync_bundle, sync_rgb, sync_th = self.synchronizer.synchronize(raw_rgb, raw_thermal)
        self.last_sync_bundle = sync_bundle
        self.last_rgb_frame = sync_rgb
        self.last_thermal_frame = sync_th

        # Sensor health status summary
        sensor_health = {
            self.rgb_sensor.sensor_id: self.rgb_sensor.check_health(),
            self.thermal_sensor.sensor_id: self.thermal_sensor.check_health(),
        }

        # Step 3: Independent Perception
        t_det_start = time.perf_counter()
        rgb_dets: List[Detection] = []
        th_dets: List[Detection] = []

        if sync_rgb is not None and sync_rgb.is_valid:
            rgb_dets = self.rgb_detector.detect(sync_rgb)

        if sync_th is not None and sync_th.is_valid:
            th_dets = self.thermal_detector.detect(sync_th)

        latency_inference = (time.perf_counter() - t_det_start) * 1000.0

        # Step 4: Multimodal Sensor Fusion
        t_fuse_start = time.perf_counter()
        fused_dets = self.fusion_engine.fuse(rgb_dets, th_dets, mode=fusion_mode)
        self.last_fused_detections = fused_dets
        latency_fusion = (time.perf_counter() - t_fuse_start) * 1000.0

        # Step 5: Pedestrian Tracking
        active_tracks = self.tracker.update(fused_dets, timestamp_sec=now_sec)
        self.last_tracks = active_tracks

        # Step 6: Kinematic Collision-Risk Engine & Alert State Machine
        t_risk_start = time.perf_counter()
        telemetry = telemetry_override or self.machine_adapter.get_telemetry()
        assessment = self.state_machine.evaluate(
            tracks=active_tracks,
            telemetry=telemetry,
            sensor_health=sensor_health,
            timestamp_sec=now_sec,
        )
        self.last_assessment = assessment
        latency_risk = (time.perf_counter() - t_risk_start) * 1000.0

        # Step 7: Transmit Warning to Machine Integration Adapter
        self.machine_adapter.send_warning_event(assessment)

        # Performance and Health Report
        t_total = (time.perf_counter() - t_start) * 1000.0
        dt_cycle = time.time() - self._last_process_time
        self._last_process_time = time.time()
        self.fps_tracker = 1.0 / dt_cycle if dt_cycle > 0 else 30.0

        health_report = SystemHealthReport(
            timestamp_sec=now_sec,
            alert_state=assessment.alert_state,
            overall_health="OK" if assessment.alert_state != AlertState.DEGRADED else "DEGRADED",
            fps=round(self.fps_tracker, 1),
            latency_capture_ms=round(latency_cap, 2),
            latency_inference_ms=round(latency_inference, 2),
            latency_fusion_ms=round(latency_fusion, 2),
            latency_risk_ms=round(latency_risk, 2),
            latency_total_ms=round(t_total, 2),
            sensors=sensor_health,
            dropped_frames={
                self.rgb_sensor.sensor_id: self.rgb_sensor.dropped_frame_counter,
                self.thermal_sensor.sensor_id: self.thermal_sensor.dropped_frame_counter,
            },
            active_tracks_count=len(active_tracks),
            confirmed_pedestrians_count=len([t for t in active_tracks if t.hit_streak >= 2]),
        )
        self.last_health_report = health_report

        # Step 8: Structured Event Logging
        self.event_logger.log_risk_event(assessment, health_report)

        return assessment, health_report

    def render_annotated_rgb_frame(self) -> Optional[np.ndarray]:
        """Draw bounding boxes, track IDs, ground distance, and safety zones on RGB frame."""
        if self.last_rgb_frame is None or not self.last_rgb_frame.is_valid:
            # Generate placeholder image
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "RGB FEED UNAVAILABLE", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return placeholder

        canvas = self.last_rgb_frame.image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # Draw active tracks
        for track in self.last_tracks:
            b = track.bbox
            color = (0, 0, 255) if self.last_assessment and self.last_assessment.alert_state == AlertState.DANGER else (0, 255, 255)
            cv2.rectangle(canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
            label = f"{track.track_id} {track.ground_pos_m[1]:.1f}m ({track.estimated_speed_mps:.1f}m/s)"
            cv2.putText(canvas, label, (int(b.x1), max(20, int(b.y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return canvas

    def render_annotated_thermal_frame(self) -> Optional[np.ndarray]:
        """Draw bounding boxes and thermal false-color on Thermal frame."""
        if self.last_thermal_frame is None or not self.last_thermal_frame.is_valid:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "THERMAL FEED UNAVAILABLE", (130, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return placeholder

        norm_8u = self.thermal_processor.normalize_to_8bit(self.last_thermal_frame.image)
        canvas = self.thermal_processor.to_pseudocolor(norm_8u)

        # Draw tracks
        for track in self.last_tracks:
            b = track.bbox
            color = (0, 255, 255)
            cv2.rectangle(canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
            cv2.putText(canvas, track.track_id, (int(b.x1), max(20, int(b.y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return canvas
