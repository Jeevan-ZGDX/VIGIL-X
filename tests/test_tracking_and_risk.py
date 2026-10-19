"""
Unit tests for pedestrian tracking, collision risk engine, and alert state machine.
"""

import time
import pytest

from src.core.config import MachineConfig, SafetyZonesConfig, StateMachineConfig, TrackingConfig
from src.core.schemas import (
    AlertState,
    BoundingBox,
    Detection,
    MachineTelemetry,
    SafetyZoneType,
    SensorHealthStatus,
    SensorType,
    TrackState,
)
from src.risk.engine import CollisionRiskEngine
from src.risk.state_machine import AlertStateMachine
from src.risk.zones import SafetyZoneManager
from src.tracking.tracker import MultiPedestrianTracker


def test_tracker_lifecycle_and_motion():
    cfg = TrackingConfig(min_hits_to_confirm=2, max_missed_frames=3, distance_threshold_m=2.0)
    tracker = MultiPedestrianTracker(cfg)

    t0 = 100.0
    det1 = Detection(
        frame_id=1,
        timestamp_sec=t0,
        sensor_type=SensorType.FUSED,
        sensor_id="fused",
        confidence=0.85,
        bbox=BoundingBox(x1=100, y1=100, x2=150, y2=250),
        calibrated_ground_pos_m=(0.0, 10.0),
    )

    # Frame 1: Tentative track created
    tracks1 = tracker.update([det1], t0)
    assert len(tracks1) == 1
    assert tracks1[0].state == TrackState.TENTATIVE

    # Frame 2: Pedestrian moved closer to machine (from y=10.0 to y=9.0 at dt=1.0s -> vy = -1.0 m/s)
    t1 = t0 + 1.0
    det2 = Detection(
        frame_id=2,
        timestamp_sec=t1,
        sensor_type=SensorType.FUSED,
        sensor_id="fused",
        confidence=0.88,
        bbox=BoundingBox(x1=100, y1=100, x2=150, y2=250),
        calibrated_ground_pos_m=(0.0, 9.0),
    )
    tracks2 = tracker.update([det2], t1)
    assert len(tracks2) == 1
    assert tracks2[0].state == TrackState.CONFIRMED
    assert tracks2[0].ground_velocity_mps[1] < 0.0  # Approaching (negative vy)

    # Coasting: missing detections for 2 frames
    tracker.update([], t1 + 0.1)
    tracks_coast = tracker.update([], t1 + 0.2)
    assert len(tracks_coast) == 1
    assert tracks_coast[0].missed_frames == 2

    # Miss 2 more frames -> total missed = 4 > max_missed_frames (3) -> Deleted
    tracker.update([], t1 + 0.3)
    tracks_final = tracker.update([], t1 + 0.4)
    assert len(tracks_final) == 0


def test_safety_zones():
    zone_cfg = SafetyZonesConfig(warning_distance_m=8.0, danger_distance_m=3.5)
    mach_cfg = MachineConfig(width_m=2.0, length_m=4.0)
    manager = SafetyZoneManager(zone_cfg, mach_cfg)
    telemetry = MachineTelemetry()

    # In front at 2.0m -> DANGER_ZONE
    assert manager.classify_point(0.0, 2.0, telemetry) == SafetyZoneType.DANGER_ZONE

    # In front at 5.0m -> WARNING_ZONE
    assert manager.classify_point(0.0, 5.0, telemetry) == SafetyZoneType.WARNING_ZONE

    # To the side (flank blind spot) at x=2.0m, y=-1.0m
    assert manager.classify_point(2.0, -1.0, telemetry) == SafetyZoneType.BLIND_SPOT_ZONE

    # Rear blind spot at x=0.0m, y=-2.0m
    assert manager.classify_point(0.0, -2.0, telemetry) == SafetyZoneType.BLIND_SPOT_ZONE

    # Far away (15m) -> None
    assert manager.classify_point(0.0, 15.0, telemetry) is None


def test_collision_risk_ttc_and_reversing():
    zone_cfg = SafetyZonesConfig(warning_distance_m=8.0, danger_distance_m=3.5)
    mach_cfg = MachineConfig(width_m=2.0, length_m=4.0)
    engine = CollisionRiskEngine(zone_cfg, mach_cfg)

    # Case 1: Machine moving forward at 3.0 m/s, pedestrian at y=5.0m walking toward machine at 1.0 m/s
    # Closing velocity = 4.0 m/s -> TTC = 5.0 / 4.0 = 1.25s <= 2.0s -> DANGER
    telemetry = MachineTelemetry(speed_mps=3.0, gear="FORWARD")
    track = tracker_dummy_track(x=0.0, y=5.0, vx=0.0, vy=-1.0)

    state, reason, zone, track_id, ttc, dist, valid = engine.evaluate_risk([track], telemetry, time.time())
    assert state == AlertState.DANGER
    assert ttc is not None and ttc < 2.0
    assert "Imminent collision" in reason or "DANGER" in reason

    # Case 2: Machine in REVERSE with pedestrian behind machine at y=-2.5m
    telemetry_rev = MachineTelemetry(speed_mps=1.5, gear="REVERSE")
    track_rear = tracker_dummy_track(x=0.0, y=-2.5, vx=0.0, vy=0.0)
    state_r, reason_r, _, _, _, _, _ = engine.evaluate_risk([track_rear], telemetry_rev, time.time())
    assert state_r == AlertState.DANGER
    assert "Rear collision risk" in reason_r


def test_alert_state_machine_hysteresis_and_sensor_fault():
    sm_cfg = StateMachineConfig(
        caution_to_normal_frames=4,
        danger_to_caution_frames=3,
        degraded_recovery_frames=3,
    )
    risk_engine = CollisionRiskEngine(SafetyZonesConfig(), MachineConfig())
    sm = AlertStateMachine(sm_cfg, risk_engine)

    healthy_sensors = {
        "cam_rgb": SensorHealthStatus.HEALTHY,
        "cam_thermal": SensorHealthStatus.HEALTHY,
    }
    telem = MachineTelemetry()

    # Step 1: Initial state is NORMAL
    res = sm.evaluate([], telem, healthy_sensors, 1.0)
    assert res.alert_state == AlertState.NORMAL

    # Step 2: Pedestrian in Danger zone -> Immediate DANGER
    danger_track = tracker_dummy_track(x=0.0, y=1.5, vx=0.0, vy=0.0)
    res_danger = sm.evaluate([danger_track], telem, healthy_sensors, 2.0)
    assert res_danger.alert_state == AlertState.DANGER

    # Step 3: Pedestrian moves away or leaves -> Hysteresis de-escalation
    # Needs 3 frames to transition from DANGER to CAUTION / NORMAL
    r1 = sm.evaluate([], telem, healthy_sensors, 3.0)
    assert r1.alert_state == AlertState.DANGER  # Frame 1: still DANGER
    r2 = sm.evaluate([], telem, healthy_sensors, 4.0)
    assert r2.alert_state == AlertState.DANGER  # Frame 2: still DANGER
    r3 = sm.evaluate([], telem, healthy_sensors, 5.0)
    assert r3.alert_state in (AlertState.CAUTION, AlertState.NORMAL)  # Frame 3: stepped down

    # Step 4: Sensor disconnection fault -> Immediate DEGRADED
    faulty_sensors = {
        "cam_rgb": SensorHealthStatus.DISCONNECTED,
        "cam_thermal": SensorHealthStatus.HEALTHY,
    }
    r_deg = sm.evaluate([], telem, faulty_sensors, 6.0)
    assert r_deg.alert_state == AlertState.DEGRADED
    assert "Sensor cam_rgb is DISCONNECTED" in r_deg.reason


def tracker_dummy_track(x: float, y: float, vx: float, vy: float):
    from src.core.schemas import Track
    return Track(
        track_id="PED_TEST",
        class_name="pedestrian",
        state=TrackState.CONFIRMED,
        first_seen_sec=100.0,
        last_seen_sec=101.0,
        age_frames=10,
        missed_frames=0,
        hit_streak=10,
        bbox=BoundingBox(x1=100, y1=100, x2=150, y2=250),
        ground_pos_m=(x, y),
        ground_velocity_mps=(vx, vy),
        estimated_speed_mps=(vx**2 + vy**2)**0.5,
        confidence=0.85,
        uncertainty=0.15,
        sensor_sources=[SensorType.FUSED],
    )
