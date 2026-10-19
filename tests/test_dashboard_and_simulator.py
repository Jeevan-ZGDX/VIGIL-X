"""
Unit and API integration tests for machine integration simulator and FastAPI dashboard.
"""

import pytest
from starlette.testclient import TestClient

from src.core.config import MachineConfig
from src.core.schemas import AlertState, MachineTelemetry, RiskAssessment, SafetyZoneType
from src.dashboard.app import app, get_pipeline
from src.integration.simulator import MachineIntegrationSimulator


def test_machine_integration_simulator_warnings():
    cfg = MachineConfig(machine_id="TEST_CRANE_01")
    sim = MachineIntegrationSimulator(cfg)

    # 1. Normal state -> no alarms
    norm_assessment = RiskAssessment(
        alert_state=AlertState.NORMAL,
        reason="Normal operation",
    )
    sim.send_warning_event(norm_assessment)
    assert sim.visual_strobe_active is False
    assert sim.audio_buzzer_hz == 0
    assert sim.advisory_slowdown_requested is False

    # 2. Caution state -> strobe active, chime active, no automatic slowdown
    caution_assessment = RiskAssessment(
        alert_state=AlertState.CAUTION,
        reason="Pedestrian in warning zone",
        active_zone=SafetyZoneType.WARNING_ZONE,
    )
    sim.send_warning_event(caution_assessment)
    assert sim.visual_strobe_active is True
    assert sim.audio_buzzer_hz == 800
    assert sim.advisory_slowdown_requested is False

    # 3. Danger state -> continuous siren, strobe, advisory slowdown
    danger_assessment = RiskAssessment(
        alert_state=AlertState.DANGER,
        reason="Pedestrian in danger zone",
        active_zone=SafetyZoneType.DANGER_ZONE,
    )
    sim.send_warning_event(danger_assessment)
    assert sim.visual_strobe_active is True
    assert sim.audio_buzzer_hz == 2500
    assert sim.advisory_slowdown_requested is True
    assert "CRITICAL" in sim.cabin_display_text


def test_dashboard_api_endpoints():
    client = TestClient(app)

    # Test /health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert "overall_health" in data_health
    assert "alert_state" in data_health

    # Test /api/step
    res_step = client.post("/api/step")
    assert res_step.status_code == 200
    data_step = res_step.json()
    assert "alert_state" in data_step

    # Test /api/status
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    data_status = res_status.json()
    assert "assessment" in data_status
    assert "health" in data_status
    assert "simulator_outputs" in data_status

    # Test /api/telemetry
    res_telem = client.post("/api/telemetry", json={"speed_mps": 2.5, "gear": "FORWARD", "direction_deg": 10.0})
    assert res_telem.status_code == 200
    assert res_telem.json()["telemetry"]["speed_mps"] == 2.5

    # Test /api/fault_injection
    res_fault = client.post("/api/fault_injection", json={"sensor": "thermal", "action": "stale", "stale_duration_sec": 1.0})
    assert res_fault.status_code == 200
    assert res_fault.json()["status"] == "fault_applied"

    # Recover
    res_rec = client.post("/api/fault_injection", json={"sensor": "thermal", "action": "recover"})
    assert res_rec.status_code == 200
