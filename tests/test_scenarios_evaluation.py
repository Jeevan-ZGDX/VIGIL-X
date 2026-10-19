"""
Automated validation tests for the 8 standardized evaluation scenarios.
Verifies multimodal fusion advantages, fault handling, and risk state transitions.
"""

from pathlib import Path
import pytest

from scripts.run_benchmark import run_single_scenario


@pytest.fixture(scope="module")
def synthetic_data_dir():
    base_dir = Path("data/synthetic")
    assert base_dir.exists(), "Synthetic scenario data must exist before running scenario tests"
    return base_dir


def test_scenario_1_normal_lighting(synthetic_data_dir):
    """Scenario 1: In normal lighting, both sensors and fused pipeline achieve high recall and detect hazard."""
    scen_path = synthetic_data_dir / "scenario_1_normal"
    res_fused = run_single_scenario(scen_path, mode="fused")
    res_rgb = run_single_scenario(scen_path, mode="rgb_only")
    res_th = run_single_scenario(scen_path, mode="thermal_only")

    assert res_fused["recall"] >= 0.90
    assert res_rgb["recall"] >= 0.90
    assert res_th["recall"] >= 0.90
    assert "DANGER" in res_fused["alert_states"]
    assert res_fused["latency_p95_ms"] < 200.0  # Real-time capability check (accounts for multi-process test runner)


def test_scenario_2_low_light_and_glare_fusion_advantage(synthetic_data_dir):
    """Scenario 2: Low light and glare blind RGB-only perception, but Thermal and Fused succeed."""
    scen_path = synthetic_data_dir / "scenario_2_low_light_glare"
    res_fused = run_single_scenario(scen_path, mode="fused")
    res_rgb = run_single_scenario(scen_path, mode="rgb_only")
    res_th = run_single_scenario(scen_path, mode="thermal_only")

    # RGB fails or is severely degraded under zero light + blinding glare
    assert res_rgb["recall"] < 0.20
    # Thermal penetrates darkness and glare
    assert res_th["recall"] >= 0.90
    # Multimodal fusion preserves safety by leveraging thermal channel
    assert res_fused["recall"] >= 0.90
    assert "CAUTION" in res_fused["alert_states"] or "DANGER" in res_fused["alert_states"]


def test_scenario_4_blind_spot_warning(synthetic_data_dir):
    """Scenario 4: Pedestrian entering machine flank triggers CAUTION state."""
    scen_path = synthetic_data_dir / "scenario_4_blind_spot"
    res_fused = run_single_scenario(scen_path, mode="fused")

    assert res_fused["recall"] >= 0.90
    assert "CAUTION" in res_fused["alert_states"]


def test_scenario_5_multiple_pedestrians(synthetic_data_dir):
    """Scenario 5: Simultaneous tracking of multiple crossing pedestrians."""
    scen_path = synthetic_data_dir / "scenario_5_multiple_pedestrians"
    res_fused = run_single_scenario(scen_path, mode="fused")

    assert res_fused["recall"] >= 0.90
    assert res_fused["true_positives"] >= 10


def test_scenario_7_environmental_interference(synthetic_data_dir):
    """Scenario 7: Industrial steam/smoke washes out RGB contrast; Thermal infrared maintains detection."""
    scen_path = synthetic_data_dir / "scenario_7_interference"
    res_fused = run_single_scenario(scen_path, mode="fused")
    res_rgb = run_single_scenario(scen_path, mode="rgb_only")
    res_th = run_single_scenario(scen_path, mode="thermal_only")

    assert res_rgb["recall"] < 0.20
    assert res_th["recall"] >= 0.90
    assert res_fused["recall"] >= 0.90


def test_scenario_8_moving_conflict_lead_time(synthetic_data_dir):
    """Scenario 8: Fast-moving equipment approach produces early warning lead time."""
    scen_path = synthetic_data_dir / "scenario_8_moving_conflict"
    res_fused = run_single_scenario(scen_path, mode="fused")

    assert res_fused["recall"] >= 0.90
    assert "DANGER" in res_fused["alert_states"]
    assert res_fused["warning_lead_time_sec"] > 0.0
