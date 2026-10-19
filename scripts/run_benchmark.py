"""
Comprehensive benchmarking and scenario evaluation runner.
Evaluates all 8 scenarios comparing FUSED, RGB_ONLY, and THERMAL_ONLY configurations.
Computes recall, precision, missed detections, capture-to-warning latency, and lead time.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from src.core.config import SystemConfig
from src.core.schemas import AlertState, BoundingBox, MachineTelemetry
from src.pipeline import SafetyPipeline
from src.sensors.replay_camera import ReplayCameraSensor

logging.basicConfig(level=logging.WARNING)


def run_single_scenario(
    scen_dir: Path,
    mode: str = "fused",
) -> Dict[str, Any]:
    gt_path = scen_dir / "ground_truth.json"
    if not gt_path.exists():
        return {}

    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    # Initialize configuration
    cfg = SystemConfig.load_from_yaml("config/default_config.yaml")
    cfg.rgb_sensor.mode = "replay"
    cfg.rgb_sensor.replay_path = str(scen_dir)
    cfg.thermal_sensor.mode = "replay"
    cfg.thermal_sensor.replay_path = str(scen_dir)

    # Replay sensors with loop=False
    rgb_sensor = ReplayCameraSensor(cfg.rgb_sensor, loop=False)
    thermal_sensor = ReplayCameraSensor(cfg.thermal_sensor, loop=False)

    pipeline = SafetyPipeline(cfg, rgb_sensor=rgb_sensor, thermal_sensor=thermal_sensor)
    pipeline.start()

    total_gt_instances = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    latencies_ms: List[float] = []
    alert_states_seen: List[str] = []
    warning_lead_time_sec: float = 0.0
    first_warning_frame: int = -1

    for frame_idx, item in enumerate(ground_truth):
        telem_data = item.get("machine_telemetry", {})
        telem = MachineTelemetry(
            speed_mps=telem_data.get("speed_mps", 0.0),
            gear=telem_data.get("gear", "FORWARD"),
            direction_deg=telem_data.get("direction_deg", 0.0),
        )

        t0 = time.perf_counter()
        assessment, health = pipeline.process_frame(telemetry_override=telem, fusion_mode=mode)
        total_lat = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(total_lat)

        alert_states_seen.append(assessment.alert_state.value)

        # Track warning lead time (time between first CAUTION/DANGER and proximity boundary)
        if assessment.alert_state in (AlertState.CAUTION, AlertState.DANGER) and first_warning_frame == -1:
            first_warning_frame = frame_idx
            # Time elapsed before reaching danger distance (frame duration ~ 0.1s)
            warning_lead_time_sec = (len(ground_truth) - frame_idx) * 0.10

        gt_peds = item.get("pedestrians", [])
        total_gt_instances += len(gt_peds)

        # Matched detections
        detected_tracks = pipeline.last_tracks
        matched_gt = set()

        for det in detected_tracks:
            matched = False
            for g_idx, g_ped in enumerate(gt_peds):
                if g_idx in matched_gt:
                    continue
                gx1, gy1, gx2, gy2 = g_ped["bbox"]
                gt_box = BoundingBox(x1=gx1, y1=gy1, x2=gx2, y2=gy2)
                if det.bbox.iou(gt_box) >= 0.15:
                    matched_gt.add(g_idx)
                    matched = True
                    break
            if matched:
                true_positives += 1
            else:
                false_positives += 1

        fn_in_frame = len(gt_peds) - len(matched_gt)
        false_negatives += fn_in_frame

    pipeline.stop()

    recall = true_positives / max(1, (true_positives + false_negatives))
    precision = true_positives / max(1, (true_positives + false_positives))
    missed_rate = false_negatives / max(1, total_gt_instances)

    lat_sorted = sorted(latencies_ms)
    p50_lat = np.percentile(lat_sorted, 50) if lat_sorted else 0.0
    p95_lat = np.percentile(lat_sorted, 95) if lat_sorted else 0.0

    return {
        "scenario": scen_dir.name,
        "mode": mode,
        "frames_evaluated": len(ground_truth),
        "gt_pedestrians": total_gt_instances,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "missed_detection_rate": round(missed_rate, 3),
        "latency_p50_ms": round(float(p50_lat), 2),
        "latency_p95_ms": round(float(p95_lat), 2),
        "warning_lead_time_sec": round(warning_lead_time_sec, 2),
        "alert_states": list(set(alert_states_seen)),
    }


def run_full_benchmark(synthetic_dir: str = "data/synthetic") -> Dict[str, Any]:
    base_dir = Path(synthetic_dir)
    scenarios = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("scenario_")])

    all_results = []
    print("\n==================== RUNNING PEDESTRIAN SAFETY BENCHMARK ====================")
    print(f"Discovered {len(scenarios)} evaluation scenarios.")

    for scen in scenarios:
        print(f"\n---> Evaluating Scenario: {scen.name}")
        for mode in ["fused", "rgb_only", "thermal_only"]:
            res = run_single_scenario(scen, mode=mode)
            if res:
                all_results.append(res)
                print(
                    f"  [{mode.upper():12s}] Recall: {res['recall']*100:5.1f}% | "
                    f"Precision: {res['precision']*100:5.1f}% | "
                    f"Missed: {res['missed_detection_rate']*100:5.1f}% | "
                    f"Lat(p95): {res['latency_p95_ms']}ms | States: {res['alert_states']}"
                )

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_trials": len(all_results),
        "results": all_results,
    }

    out_json = Path("data/benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved benchmark results to {out_json}")
    return report


if __name__ == "__main__":
    run_full_benchmark()
