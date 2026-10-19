"""
Unit and integration tests for sensor ingestion, health monitoring,
timestamp handling, and thermal image processing.
"""

import time
import numpy as np
import pytest

from src.core.config import SensorConfig
from src.core.schemas import SensorType, SensorHealthStatus
from src.sensors.base import BaseSensor
from src.sensors.replay_camera import ReplayCameraSensor
from src.sensors.thermal_processor import ThermalProcessor


def test_replay_sensor_in_memory_connection():
    config = SensorConfig(
        sensor_id="rgb_test",
        sensor_type="RGB",
        mode="replay",
        width=320,
        height=240,
        target_fps=30.0,
        stale_timeout_sec=0.2,
    )
    frames = [np.ones((240, 320, 3), dtype=np.uint8) * 100 for _ in range(5)]
    sensor = ReplayCameraSensor(config=config, in_memory_frames=frames, loop=True)

    assert sensor.connect() is True
    assert sensor.is_connected is True
    assert sensor.health_status == SensorHealthStatus.HEALTHY

    # Read 3 frames
    for i in range(3):
        frame_data = sensor.read_frame()
        assert frame_data.is_valid is True
        assert frame_data.metadata.frame_id == i + 1
        assert frame_data.metadata.sensor_type == SensorType.RGB
        assert frame_data.metadata.width == 320
        assert frame_data.metadata.height == 240
        assert frame_data.image.shape == (240, 320, 3)

    sensor.disconnect()
    assert sensor.is_connected is False
    assert sensor.health_status == SensorHealthStatus.DISCONNECTED

    # Attempt to read while disconnected
    stale_data = sensor.read_frame()
    assert stale_data.is_valid is False
    assert stale_data.image is None
    assert sensor.dropped_frame_counter >= 1


def test_sensor_stale_detection():
    config = SensorConfig(
        sensor_id="thermal_test",
        sensor_type="THERMAL",
        mode="replay",
        width=320,
        height=240,
        target_fps=30.0,
        stale_timeout_sec=0.05,  # 50 ms timeout
    )
    frame = np.ones((240, 320), dtype=np.uint8) * 150
    sensor = ReplayCameraSensor(config=config, in_memory_frames=[frame], loop=True)
    sensor.connect()

    # Read frame
    data = sensor.read_frame()
    assert data.is_valid is True
    assert sensor.check_health() == SensorHealthStatus.HEALTHY

    # Sleep longer than stale timeout
    time.sleep(0.08)
    assert sensor.check_health() == SensorHealthStatus.STALE


def test_sensor_fault_injection():
    config = SensorConfig(
        sensor_id="fault_test",
        sensor_type="RGB",
        mode="replay",
        width=320,
        height=240,
        target_fps=30.0,
        stale_timeout_sec=0.5,
    )
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(3)]
    sensor = ReplayCameraSensor(config=config, in_memory_frames=frames, loop=True)
    sensor.connect()

    # Inject frame drop
    sensor.inject_drop(True)
    dropped = sensor.read_frame()
    assert dropped.is_valid is False
    assert sensor.dropped_frame_counter == 1

    # Restore
    sensor.inject_drop(False)
    recovered = sensor.read_frame()
    assert recovered.is_valid is True


def test_thermal_processor_radiometric_16bit():
    processor = ThermalProcessor(target_width=320, target_height=240)

    # Simulate 16-bit radiometric image: ambient ~20C (29315 cK), pedestrian ~35C (30815 cK)
    raw_16bit = np.full((240, 320), fill_value=29315, dtype=np.uint16)
    # Add a warm pedestrian blob
    raw_16bit[80:160, 140:180] = 30815

    norm_8u = processor.normalize_to_8bit(raw_16bit)
    assert norm_8u.dtype == np.uint8
    assert norm_8u.shape == (240, 320)
    # Pedestrian region should have higher intensity
    ped_mean = np.mean(norm_8u[80:160, 140:180])
    ambient_mean = np.mean(norm_8u[0:40, 0:40])
    assert ped_mean > ambient_mean

    # Pseudocolor conversion
    color = processor.to_pseudocolor(norm_8u)
    assert color.shape == (240, 320, 3)

    # Heat mask extraction
    heat_mask = processor.extract_pedestrian_heat_mask(norm_8u, low_percentile=75.0)
    assert heat_mask.shape == (240, 320)
    assert np.sum(heat_mask[80:160, 140:180]) > 0
