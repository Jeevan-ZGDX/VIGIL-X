"""
Synthetic scenario generator for the 8 standardized evaluation scenarios.
Generates synchronized RGB and Thermal frame sequences with ground-truth metadata.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np


def draw_synthetic_pedestrian(
    rgb_img: np.ndarray,
    th_img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    rgb_contrast: float = 1.0,
    thermal_contrast: float = 1.0,
    occlusion_ratio: float = 0.0,
):
    """Draw a realistic synthetic pedestrian profile on RGB and Thermal image canvases."""
    img_h, img_w = rgb_img.shape[:2]
    vis_h = int(h * (1.0 - occlusion_ratio))
    top_y = y
    bot_y = min(img_h - 1, y + vis_h)

    # RGB pedestrian: head (circle) and torso/legs (rounded rect)
    head_radius = max(3, int(w * 0.35))
    head_center_x = x + w // 2
    head_center_y = top_y + head_radius

    # Draw on RGB
    cv2.circle(rgb_img, (head_center_x, head_center_y), head_radius, (int(160 * rgb_contrast), int(130 * rgb_contrast), int(110 * rgb_contrast)), -1)
    body_y1 = top_y + 2 * head_radius
    body_y2 = bot_y
    if body_y2 > body_y1:
        cv2.rectangle(rgb_img, (x, body_y1), (x + w, body_y2), (int(40 * rgb_contrast), int(60 * rgb_contrast), int(180 * rgb_contrast)), -1)

    # Thermal infrared pedestrian: warm human body (210-240 intensity vs cooler 40 ambient)
    th_intensity = int(40 + 190 * thermal_contrast)
    cv2.circle(th_img, (head_center_x, head_center_y), head_radius, th_intensity, -1)
    if body_y2 > body_y1:
        cv2.rectangle(th_img, (x, body_y1), (x + w, body_y2), th_intensity, -1)


def generate_scenario_1_normal(out_dir: Path, num_frames: int = 15):
    """Scenario 1: Normal lighting. Pedestrian approaching in forward path."""
    scen_dir = out_dir / "scenario_1_normal"
    scen_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for f in range(num_frames):
        # Image background: well-lit industrial yard (gray 130)
        rgb = np.full((480, 640, 3), 130, dtype=np.uint8)
        th = np.full((480, 640), 45, dtype=np.uint8)

        # Pedestrian walking closer (height increases, y increases)
        h = int(90 + f * 7)
        w = int(h / 2.5)
        px = 300
        py = int(220 + f * 8)

        draw_synthetic_pedestrian(rgb, th, px, py, w, h, rgb_contrast=1.0, thermal_contrast=1.0)

        rgb_path = scen_dir / f"rgb_{f:03d}.png"
        th_path = scen_dir / f"th_{f:03d}.png"
        cv2.imwrite(str(rgb_path), rgb)
        cv2.imwrite(str(th_path), th)

        metadata.append({
            "frame_id": f,
            "timestamp_sec": 100.0 + f * 0.1,
            "pedestrians": [{"id": "P1", "bbox": [px, py, px + w, py + h], "distance_m": round(12.0 - f * 0.6, 1)}],
            "machine_telemetry": {"speed_mps": 2.0, "gear": "FORWARD", "direction_deg": 0.0},
        })

    with open(scen_dir / "ground_truth.json", "w") as f:
        json.dump(metadata, f, indent=2)


def generate_scenario_2_low_light_glare(out_dir: Path, num_frames: int = 15):
    """Scenario 2: Low light (night) and intense glare. Thermal maintains heat contrast while RGB is blinded."""
    scen_dir = out_dir / "scenario_2_low_light_glare"
    scen_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for f in range(num_frames):
        # Dark ambient background (intensity 15)
        rgb = np.full((480, 640, 3), 18, dtype=np.uint8)
        # Add harsh glare spot on left (e.g. oncoming forklift work lamp)
        cv2.circle(rgb, (160, 240), 90, (255, 255, 255), -1)

        # Thermal image immune to glare, cool ambient (35)
        th = np.full((480, 640), 35, dtype=np.uint8)

        h = 120
        w = int(h / 2.5)
        px = 320
        py = 220

        # Pedestrian has poor visibility in RGB (< 0.15 contrast), but clear thermal heat signature
        draw_synthetic_pedestrian(rgb, th, px, py, w, h, rgb_contrast=0.15, thermal_contrast=1.0)

        cv2.imwrite(str(scen_dir / f"rgb_{f:03d}.png"), rgb)
        cv2.imwrite(str(scen_dir / f"th_{f:03d}.png"), th)

        metadata.append({
            "frame_id": f,
            "timestamp_sec": 100.0 + f * 0.1,
            "pedestrians": [{"id": "P1", "bbox": [px, py, px + w, py + h], "distance_m": 6.5}],
            "machine_telemetry": {"speed_mps": 1.5, "gear": "FORWARD", "direction_deg": 0.0},
        })

    with open(scen_dir / "ground_truth.json", "w") as f:
        json.dump(metadata, f, indent=2)


def generate_scenario_3_occlusion(out_dir: Path, num_frames: int = 10):
    """Scenario 3: Partial pedestrian occlusion (lower 40% occluded by pallet/barrier)."""
    scen_dir = out_dir / "scenario_3_occlusion"
    scen_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for f in range(num_frames):
        rgb = np.full((480, 640, 3), 110, dtype=np.uint8)
        th = np.full((480, 640), 40, dtype=np.uint8)

        h = 130
        w = int(h / 2.5)
        px = 280
        py = 200

        # 40% occluded
        draw_synthetic_pedestrian(rgb, th, px, py, w, h, rgb_contrast=0.8, thermal_contrast=0.9, occlusion_ratio=0.4)

        # Draw cold occlusion crate in front
        cv2.rectangle(rgb, (px - 20, py + int(h * 0.6)), (px + w + 20, py + h + 20), (80, 60, 40), -1)
        cv2.rectangle(th, (px - 20, py + int(h * 0.6)), (px + w + 20, py + h + 20), 30, -1)

        cv2.imwrite(str(scen_dir / f"rgb_{f:03d}.png"), rgb)
        cv2.imwrite(str(scen_dir / f"th_{f:03d}.png"), th)

        metadata.append({
            "frame_id": f,
            "timestamp_sec": 100.0 + f * 0.1,
            "pedestrians": [{"id": "P1", "bbox": [px, py, px + w, py + h], "distance_m": 5.0, "occluded": True}],
            "machine_telemetry": {"speed_mps": 1.0, "gear": "FORWARD", "direction_deg": 0.0},
        })

    with open(scen_dir / "ground_truth.json", "w") as f:
        json.dump(metadata, f, indent=2)


def generate_scenario_4_blind_spot(out_dir: Path, num_frames: int = 10):
    """Scenario 4: Blind-spot approach. Pedestrian approaching machine rear/flank."""
    scen_dir = out_dir / "scenario_4_blind_spot"
    scen_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for f in range(num_frames):
        rgb = np.full((480, 640, 3), 120, dtype=np.uint8)
        th = np.full((480, 640), 40, dtype=np.uint8)

        # Positioned in side flank blind spot (near right edge of frame)
        h = 140
        w = int(h / 2.5)
        px = int(500 - f * 6)
        py = 220

        draw_synthetic_pedestrian(rgb, th, px, py, w, h)

        cv2.imwrite(str(scen_dir / f"rgb_{f:03d}.png"), rgb)
        cv2.imwrite(str(scen_dir / f"th_{f:03d}.png"), th)

        metadata.append({
            "frame_id": f,
            "timestamp_sec": 100.0 + f * 0.1,
            "pedestrians": [{"id": "P1", "bbox": [px, py, px + w, py + h], "distance_m": 2.5, "zone": "BLIND_SPOT"}],
            "machine_telemetry": {"speed_mps": 0.5, "gear": "FORWARD", "direction_deg": 15.0},
        })

    with open(scen_dir / "ground_truth.json", "w") as f:
        json.dump(metadata, f, indent=2)


def generate_scenario_5_multiple_pedestrians(out_dir: Path, num_frames: int = 12):
    """Scenario 5: Multiple pedestrians crossing with distinct trajectories."""
    scen_dir = out_dir / "scenario_5_multiple_pedestrians"
    scen_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for f in range(num_frames):
        rgb = np.full((480, 640, 3), 120, dtype=np.uint8)
        th = np.full((480, 640), 40, dtype=np.uint8)

        # Pedestrian 1: moving left to right
        p1_h = 120
        p1_w = int(p1_h / 2.5)
        p1_x = int(120 + f * 12)
        p1_y = 220
        draw_synthetic_pedestrian(rgb, th, p1_x, p1_y, p1_w, p1_h)

        # Pedestrian 2: moving right to left
        p2_h = 130
        p2_w = int(p2_h / 2.5)
        p2_x = int(480 - f * 10)
        p2_y = 210
        draw_synthetic_pedestrian(rgb, th, p2_x, p2_y, p2_w, p2_h)

        cv2.imwrite(str(scen_dir / f"rgb_{f:03d}.png"), rgb)
        cv2.imwrite(str(scen_dir / f"th_{f:03d}.png"), th)

        metadata.append({
            "frame_id": f,
            "timestamp_sec": 100.0 + f * 0.1,
            "pedestrians": [
                {"id": "P1", "bbox": [p1_x, p1_y, p1_x + p1_w, p1_y + p1_h]},
                {"id": "P2", "bbox": [p2_x, p2_y, p2_x + p2_w, p2_y + p2_h]},
            ],
            "machine_telemetry": {"speed_mps": 1.0, "gear": "FORWARD", "direction_deg": 0.0},
        })

    with open(scen_dir / "ground_truth.json", "w") as f:
        json.dump(metadata, f, indent=2)


def generate_scenario_6_sensor_fault(out_dir: Path, num_frames: int = 10):
    """Scenario 6: Sensor fault injection (thermal stream drops frames midway)."""
    scen_dir = out_dir / "scenario_6_sensor_fault"
    scen_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for f in range(num_frames):
        rgb = np.full((480, 640, 3), 120, dtype=np.uint8)
        th = np.full((480, 640), 40, dtype=np.uint8)

        # In frames 4-7, thermal camera disconnects (empty/black/stale)
        is_thermal_fault = 4 <= f <= 7

        h = 120
        w = int(h / 2.5)
        px = 300
        py = 220
        draw_synthetic_pedestrian(rgb, th, px, py, w, h)

        if is_thermal_fault:
            th = np.zeros((480, 640), dtype=np.uint8)  # Disconnected/blank

        cv2.imwrite(str(scen_dir / f"rgb_{f:03d}.png"), rgb)
        cv2.imwrite(str(scen_dir / f"th_{f:03d}.png"), th)

        metadata.append({
            "frame_id": f,
            "timestamp_sec": 100.0 + f * 0.1,
            "thermal_fault": is_thermal_fault,
            "expected_state": "DEGRADED" if is_thermal_fault else "NORMAL",
        })

    with open(scen_dir / "ground_truth.json", "w") as f:
        json.dump(metadata, f, indent=2)


def generate_scenario_7_interference(out_dir: Path, num_frames: int = 10):
    """Scenario 7: Environmental smoke/steam interference obscuring visible RGB light."""
    scen_dir = out_dir / "scenario_7_interference"
    scen_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for f in range(num_frames):
        rgb = np.full((480, 640, 3), 120, dtype=np.uint8)
        th = np.full((480, 640), 40, dtype=np.uint8)

        h = 120
        w = int(h / 2.5)
        px = 300
        py = 220
        draw_synthetic_pedestrian(rgb, th, px, py, w, h)

        # Add heavy smoke/steam haze in RGB (washes out contrast)
        haze = np.random.normal(160, 20, (480, 640, 3)).astype(np.uint8)
        rgb = cv2.addWeighted(rgb, 0.25, haze, 0.75, 0)

        cv2.imwrite(str(scen_dir / f"rgb_{f:03d}.png"), rgb)
        cv2.imwrite(str(scen_dir / f"th_{f:03d}.png"), th)

        metadata.append({
            "frame_id": f,
            "timestamp_sec": 100.0 + f * 0.1,
            "pedestrians": [{"id": "P1", "bbox": [px, py, px + w, py + h], "distance_m": 5.0}],
            "condition": "steam_smoke_interference",
        })

    with open(scen_dir / "ground_truth.json", "w") as f:
        json.dump(metadata, f, indent=2)


def generate_scenario_8_moving_conflict(out_dir: Path, num_frames: int = 15):
    """Scenario 8: Moving-equipment conflict. High forward velocity with imminent collision."""
    scen_dir = out_dir / "scenario_8_moving_conflict"
    scen_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for f in range(num_frames):
        rgb = np.full((480, 640, 3), 120, dtype=np.uint8)
        th = np.full((480, 640), 40, dtype=np.uint8)

        # Rapidly closing distance (machine speed 4.5 m/s)
        # Distance: 14.0m down to 2.0m
        dist_m = max(1.5, 14.0 - f * 0.9)
        h = int(60 + (14.0 - dist_m) * 12)
        w = int(h / 2.5)
        px = 300
        py = int(180 + (14.0 - dist_m) * 15)

        draw_synthetic_pedestrian(rgb, th, px, py, w, h)

        cv2.imwrite(str(scen_dir / f"rgb_{f:03d}.png"), rgb)
        cv2.imwrite(str(scen_dir / f"th_{f:03d}.png"), th)

        metadata.append({
            "frame_id": f,
            "timestamp_sec": 100.0 + f * 0.1,
            "pedestrians": [{"id": "P1", "bbox": [px, py, px + w, py + h], "distance_m": round(dist_m, 2)}],
            "machine_telemetry": {"speed_mps": 4.5, "gear": "FORWARD", "direction_deg": 0.0},
        })

    with open(scen_dir / "ground_truth.json", "w") as f:
        json.dump(metadata, f, indent=2)


def generate_all_scenarios(base_dir: str = "data/synthetic"):
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_scenario_1_normal(out_dir)
    generate_scenario_2_low_light_glare(out_dir)
    generate_scenario_3_occlusion(out_dir)
    generate_scenario_4_blind_spot(out_dir)
    generate_scenario_5_multiple_pedestrians(out_dir)
    generate_scenario_6_sensor_fault(out_dir)
    generate_scenario_7_interference(out_dir)
    generate_scenario_8_moving_conflict(out_dir)
    print(f"Successfully generated all 8 evaluation datasets in {out_dir}")


if __name__ == "__main__":
    generate_all_scenarios()
