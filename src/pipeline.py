"""
End-to-end Pedestrian Safety Fusion Pipeline Orchestrator.
Supports real-time background execution loops, live camera ingestion,
ISO 3691-4 safety zone projections, radiometric thermal spot metering, and standard measures.
"""

from __future__ import annotations
import collections
import logging
import math
import threading
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import psutil

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
from src.sensors.live_thermal_adapter import LiveThermalAdapter
from src.sensors.replay_camera import ReplayCameraSensor
from src.sensors.thermal_processor import ThermalProcessor
from src.storage.event_logger import StructuredEventLogger
from src.tracking.tracker import MultiPedestrianTracker

logger = logging.getLogger(__name__)


class SafetyPipeline:
    """
    Main orchestrator integrating all stages of the pedestrian safety system.
    Supports continuous real-time execution, live cameras, and standard industrial measures.
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
            self.thermal_sensor = LiveThermalAdapter(config.thermal_sensor, companion_live_sensor=self.rgb_sensor)
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

        # Threading and concurrency
        self.is_running = False
        self._realtime_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Operational states & snapshots
        self.frame_index = 0
        self.last_sync_bundle: Optional[SyncFrameBundle] = None
        self.last_rgb_frame: Optional[FrameData] = None
        self.last_thermal_frame: Optional[FrameData] = None
        self.last_fused_detections: List[Detection] = []
        self.last_tracks: List[Track] = []
        self.last_assessment: Optional[RiskAssessment] = None
        self.last_health_report: Optional[SystemHealthReport] = None

        # Cached annotated frames for zero-latency MJPEG streaming
        self._cached_rgb_rendered: Optional[np.ndarray] = None
        self._cached_thermal_rendered: Optional[np.ndarray] = None

        # Performance rolling metrics
        self.fps_tracker = 0.0
        self._last_process_time = time.time()
        self._latency_history = collections.deque(maxlen=60)
        self._last_cpu_percent = 0.0
        self._last_mem_mb = 0.0

    def start(self) -> bool:
        """Start and connect all sensors."""
        rgb_ok = self.rgb_sensor.connect()
        th_ok = self.thermal_sensor.connect()
        self.is_running = True
        logger.info(f"Pipeline started (RGB: {rgb_ok}, Thermal: {th_ok})")
        return rgb_ok and th_ok

    def stop(self) -> None:
        """Gracefully release camera resources and halt pipeline."""
        self.stop_realtime_loop()
        self.is_running = False
        self.rgb_sensor.disconnect()
        self.thermal_sensor.disconnect()
        logger.info("Pipeline stopped and camera resources released")

    def start_realtime_loop(self, target_fps: float = 25.0) -> None:
        """Launch background daemon thread executing real-time capture and perception."""
        if self._realtime_thread is not None and self._realtime_thread.is_alive():
            return

        self.start()
        self.is_running = True
        self._realtime_thread = threading.Thread(
            target=self._realtime_worker,
            args=(target_fps,),
            daemon=True,
            name="RealtimeSafetyWorker",
        )
        self._realtime_thread.start()
        logger.info(f"Launched continuous real-time safety pipeline worker ({target_fps} FPS)")

    def stop_realtime_loop(self) -> None:
        """Halt continuous background worker."""
        self.is_running = False
        if self._realtime_thread is not None and self._realtime_thread.is_alive():
            self._realtime_thread.join(timeout=1.5)
            self._realtime_thread = None

    def _realtime_worker(self, target_fps: float):
        frame_interval = 1.0 / max(1.0, target_fps)
        while self.is_running:
            t0 = time.perf_counter()
            try:
                self.process_frame()
            except Exception as e:
                logger.error(f"Real-time cycle error: {e}", exc_info=True)

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)

    def process_frame(
        self,
        telemetry_override: Optional[MachineTelemetry] = None,
        fusion_mode: str = "fused",
    ) -> Tuple[RiskAssessment, SystemHealthReport]:
        """
        Execute one full capture-detection-fusion-tracking-risk cycle with standard measures.
        Thread-safe method updating internal state and cached visual overlays.
        """
        t_start = time.perf_counter()
        now_sec = time.time()

        # Step 1: Capture Frames
        t_cap_start = time.perf_counter()
        raw_rgb = self.rgb_sensor.read_frame()
        raw_thermal = self.thermal_sensor.read_frame()
        latency_cap = (time.perf_counter() - t_cap_start) * 1000.0

        # Step 2: Temporal Synchronization
        sync_bundle, sync_rgb, sync_th = self.synchronizer.synchronize(raw_rgb, raw_thermal)

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
        latency_fusion = (time.perf_counter() - t_fuse_start) * 1000.0

        # Step 5: Pedestrian Tracking
        active_tracks = self.tracker.update(fused_dets, timestamp_sec=now_sec)

        # Step 6: Kinematic Collision-Risk Engine & Alert State Machine
        t_risk_start = time.perf_counter()
        telemetry = telemetry_override or self.machine_adapter.get_telemetry()
        assessment = self.state_machine.evaluate(
            tracks=active_tracks,
            telemetry=telemetry,
            sensor_health=sensor_health,
            timestamp_sec=now_sec,
        )

        # Attach thermal spot radiometry if available
        if hasattr(self.thermal_sensor, "spot_temperature_celsius"):
            assessment.thermal_spot_celsius = self.thermal_sensor.spot_temperature_celsius

        latency_risk = (time.perf_counter() - t_risk_start) * 1000.0

        # Step 7: Transmit Warning to Machine Integration Adapter
        self.machine_adapter.send_warning_event(assessment)

        # Performance and Health Report with standard measures
        t_total = (time.perf_counter() - t_start) * 1000.0
        self._latency_history.append(t_total)
        p50_lat = float(np.percentile(list(self._latency_history), 50))
        p95_lat = float(np.percentile(list(self._latency_history), 95))

        dt_cycle = time.time() - self._last_process_time
        self._last_process_time = time.time()
        self.fps_tracker = 1.0 / dt_cycle if dt_cycle > 0 else 30.0

        # Periodic resource query
        if self.frame_index % 10 == 0:
            try:
                self._last_cpu_percent = psutil.cpu_percent()
                self._last_mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            except Exception:
                pass

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
            latency_p50_ms=round(p50_lat, 2),
            latency_p95_ms=round(p95_lat, 2),
            cpu_percent=round(self._last_cpu_percent, 1),
            memory_mb=round(self._last_mem_mb, 1),
            sensors=sensor_health,
            dropped_frames={
                self.rgb_sensor.sensor_id: self.rgb_sensor.dropped_frame_counter,
                self.thermal_sensor.sensor_id: self.thermal_sensor.dropped_frame_counter,
            },
            active_tracks_count=len(active_tracks),
            confirmed_pedestrians_count=len([t for t in active_tracks if t.hit_streak >= 2]),
        )

        # Step 8: Structured Event Logging
        self.event_logger.log_risk_event(assessment, health_report)

        # Thread-safe state update and immediate overlay generation
        with self._lock:
            self.frame_index += 1
            self.last_sync_bundle = sync_bundle
            self.last_rgb_frame = sync_rgb
            self.last_thermal_frame = sync_th
            self.last_fused_detections = fused_dets
            self.last_tracks = active_tracks
            self.last_assessment = assessment
            self.last_health_report = health_report
            self._render_overlays_locked()

        return assessment, health_report

    def _render_overlays_locked(self) -> None:
        """Internal overlay renderer generating HUD banners, metric units, and ISO safety zones."""
        # 1. RGB Render
        if self.last_rgb_frame is not None and self.last_rgb_frame.is_valid and self.last_rgb_frame.image is not None:
            canvas_rgb = self.last_rgb_frame.image.copy()
            if canvas_rgb.ndim == 2:
                canvas_rgb = cv2.cvtColor(canvas_rgb, cv2.COLOR_GRAY2BGR)

            h, w = canvas_rgb.shape[:2]
            stop_dist = self.last_assessment.stopping_distance_m if self.last_assessment else 3.5

            # Project Ground Safety Zones (Corridor lines)
            # Danger field: red ground trapezoid
            pt_danger_left = (int(w * 0.25), int(h * 0.95))
            pt_danger_right = (int(w * 0.75), int(h * 0.95))
            pt_danger_top_left = (int(w * 0.35), int(h * 0.70))
            pt_danger_top_right = (int(w * 0.65), int(h * 0.70))
            pts_danger = np.array([pt_danger_left, pt_danger_right, pt_danger_top_right, pt_danger_top_left], np.int32)

            overlay = canvas_rgb.copy()
            state_color = (0, 0, 255) if self.last_assessment and self.last_assessment.alert_state == AlertState.DANGER else (0, 165, 255) if self.last_assessment and self.last_assessment.alert_state == AlertState.CAUTION else (0, 255, 0)
            cv2.polylines(overlay, [pts_danger], isClosed=True, color=state_color, thickness=2)
            cv2.addWeighted(overlay, 0.4, canvas_rgb, 0.6, 0, canvas_rgb)

            # Draw Pedestrian Track Annotations
            for track in self.last_tracks:
                b = track.bbox
                tx, ty = track.ground_pos_m
                dist_m = math.hypot(tx, ty)
                dist_cm = int(dist_m * 100)
                speed_kmh = track.estimated_speed_mps * 3.6

                box_color = (0, 0, 255) if dist_m <= stop_dist else (0, 255, 255)
                cv2.rectangle(canvas_rgb, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), box_color, 2)

                # Standard Measures Label
                lbl1 = f"{track.track_id} | {dist_m:.2f}m ({dist_cm}cm)"
                lbl2 = f"{speed_kmh:.1f} km/h | Conf: {int(track.confidence * 100)}%"
                cv2.rectangle(canvas_rgb, (int(b.x1), max(0, int(b.y1) - 34)), (int(b.x1) + 210, int(b.y1)), (15, 23, 42), -1)
                cv2.putText(canvas_rgb, lbl1, (int(b.x1) + 4, max(12, int(b.y1) - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                cv2.putText(canvas_rgb, lbl2, (int(b.x1) + 4, max(24, int(b.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1)

            # Top HUD Header Bar
            cv2.rectangle(canvas_rgb, (0, 0), (w, 30), (15, 23, 42), -1)
            state_text = self.last_assessment.alert_state.value if self.last_assessment else "NORMAL"
            hud_str = f"ISO STATE: {state_text} | FPS: {self.fps_tracker:.1f} | P95: {self.last_health_report.latency_p95_ms if self.last_health_report else 0:.1f}ms | STOP DIST: {stop_dist:.1f}m"
            cv2.putText(canvas_rgb, hud_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color, 1)

            self._cached_rgb_rendered = canvas_rgb
        else:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "RGB FEED CONNECTING...", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            self._cached_rgb_rendered = blank

        # 2. Thermal Render
        if self.last_thermal_frame is not None and self.last_thermal_frame.is_valid and self.last_thermal_frame.image is not None:
            norm_8u = self.thermal_processor.normalize_to_8bit(self.last_thermal_frame.image)
            canvas_th = self.thermal_processor.to_pseudocolor(norm_8u)
            th_h, th_w = canvas_th.shape[:2]

            # Crosshair and Radiometric Spot Temperature
            cx, cy = th_w // 2, th_h // 2
            cv2.drawMarker(canvas_th, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)
            spot_temp = getattr(self.thermal_sensor, "spot_temperature_celsius", 24.0)
            max_temp = getattr(self.thermal_sensor, "max_temperature_celsius", 35.0)

            # Draw Tracks on Thermal
            for track in self.last_tracks:
                b = track.bbox
                cv2.rectangle(canvas_th, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 255, 255), 2)
                cv2.putText(canvas_th, f"{track.track_id} (IR)", (int(b.x1), max(20, int(b.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Thermal Telemetry HUD
            cv2.rectangle(canvas_th, (0, 0), (th_w, 30), (15, 23, 42), -1)
            th_hud = f"THERMAL RADIOMETRY | Spot: {spot_temp:.1f} C | Peak: {max_temp:.1f} C | Range: 20-38 C"
            cv2.putText(canvas_th, th_hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

            self._cached_thermal_rendered = canvas_th
        else:
            blank_th = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_th, "THERMAL IR CONNECTING...", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            self._cached_thermal_rendered = blank_th

    def render_annotated_rgb_frame(self) -> Optional[np.ndarray]:
        """Return latest annotated RGB frame from thread-safe cache."""
        with self._lock:
            if self._cached_rgb_rendered is not None:
                return self._cached_rgb_rendered.copy()
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def render_annotated_thermal_frame(self) -> Optional[np.ndarray]:
        """Return latest false-color Thermal frame with radiometric spot readouts."""
        with self._lock:
            if self._cached_thermal_rendered is not None:
                return self._cached_thermal_rendered.copy()
            return np.zeros((480, 640, 3), dtype=np.uint8)
