"""
Configurable safety zones for industrial equipment.
Defines Warning, Danger, and Blind-Spot spatial boundaries relative to the machine.
"""

from __future__ import annotations
from typing import Optional
from src.core.config import SafetyZonesConfig, MachineConfig
from src.core.schemas import MachineTelemetry, SafetyZoneType


class SafetyZoneManager:
    """
    Evaluates 2D ground coordinates (X, Y) relative to the machine
    and classifies them into safety zones.
    Coordinate reference frame:
    - Origin (0,0) is center of front bumper of the machine.
    - +Y is forward in front of machine.
    - -Y is rear / behind machine.
    - +X is right flank.
    - -X is left flank.
    """

    def __init__(self, zone_config: SafetyZonesConfig, machine_config: MachineConfig):
        self.zone_cfg = zone_config
        self.mach_cfg = machine_config

    def classify_point(self, x: float, y: float, telemetry: MachineTelemetry) -> Optional[SafetyZoneType]:
        """Classify ground point (x, y) into active safety zones."""
        half_width = self.mach_cfg.width_m / 2.0
        safety_margin = 0.5

        # 1. Check Rear Blind Spot (active especially when in reverse)
        if y < 0:
            if abs(y) <= self.zone_cfg.blind_spot_rear_m and abs(x) <= (half_width + 1.0):
                return SafetyZoneType.BLIND_SPOT_ZONE

        # 2. Check Lateral Flank Blind Spots (alongside machine body)
        if -self.mach_cfg.length_m <= y <= 1.0:
            if half_width < abs(x) <= (half_width + self.zone_cfg.blind_spot_right_m):
                return SafetyZoneType.BLIND_SPOT_ZONE

        # 3. Danger Zone (imminent hazard directly in front path)
        if 0.0 <= y <= self.zone_cfg.danger_distance_m:
            if abs(x) <= (half_width + safety_margin):
                return SafetyZoneType.DANGER_ZONE

        # 4. Warning Zone (caution approach zone ahead)
        if 0.0 <= y <= self.zone_cfg.warning_distance_m:
            if abs(x) <= (half_width + 1.8):
                return SafetyZoneType.WARNING_ZONE

        return None
