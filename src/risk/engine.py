"""
Collision-risk evaluation engine.
Analyzes pedestrian tracks, equipment kinematics, zone encroachment,
dynamic Time-to-Collision (TTC), and crossing path conflict trajectories.
"""

from __future__ import annotations
import math
import time
from typing import List, Optional, Tuple
from src.core.config import SafetyZonesConfig, MachineConfig
from src.core.schemas import (
    AlertState,
    MachineTelemetry,
    RiskAssessment,
    SafetyZoneType,
    Track,
    TrackState,
)
from src.risk.zones import SafetyZoneManager


class CollisionRiskEngine:
    """
    Kinematic collision-risk evaluation engine.
    Produces deterministic warning states with explicit explanations.
    """

    def __init__(self, zone_config: SafetyZonesConfig, machine_config: MachineConfig):
        self.zone_cfg = zone_config
        self.mach_cfg = machine_config
        self.zone_manager = SafetyZoneManager(zone_config, machine_config)

    def evaluate_risk(
        self,
        tracks: List[Track],
        telemetry: Optional[MachineTelemetry],
        timestamp_sec: float,
    ) -> Tuple[AlertState, str, Optional[SafetyZoneType], Optional[str], Optional[float], Optional[float], bool]:
        """
        Evaluate collision risk across all active pedestrian tracks.
        Returns:
            (candidate_state, reason, active_zone, critical_track_id, min_ttc_sec, min_distance_m, telemetry_valid)
        """
        # Validate telemetry
        telem_valid = True
        telemetry_warning = ""
        if telemetry is None:
            telem_valid = False
            telemetry_warning = " [Machine telemetry unavailable]"
            telemetry = MachineTelemetry(timestamp_sec=timestamp_sec, is_valid=False)
        elif not telemetry.is_valid:
            telem_valid = False
            telemetry_warning = " [Machine telemetry invalid]"
        elif (timestamp_sec - telemetry.timestamp_sec) > self.mach_cfg.telemetry_stale_timeout_sec:
            telem_valid = False
            telemetry_warning = f" [Telemetry stale: {timestamp_sec - telemetry.timestamp_sec:.2f}s]"

        v_mach = telemetry.speed_mps if telem_valid else 0.0
        if telemetry.gear == "REVERSE":
            v_mach = -abs(v_mach)

        # Filter relevant tracks (ignore tentative with very low hits)
        active_tracks = [t for t in tracks if t.state == TrackState.CONFIRMED or t.hit_streak >= 2]

        if not active_tracks:
            reason = "No active pedestrians detected in monitored zones." + telemetry_warning
            return (AlertState.NORMAL, reason, None, None, None, None, telem_valid)

        highest_state = AlertState.NORMAL
        primary_reason = "No immediate conflict detected."
        critical_zone = None
        critical_track_id = None
        min_ttc: Optional[float] = None
        min_dist: Optional[float] = None

        for track in active_tracks:
            tx, ty = track.ground_pos_m
            dist = math.sqrt(tx**2 + ty**2)
            if min_dist is None or dist < min_dist:
                min_dist = round(dist, 2)

            vx, vy = track.ground_velocity_mps

            # 1. Spatial Zone Encroachment
            zone = self.zone_manager.classify_point(tx, ty, telemetry)

            # 2. Dynamic Time-To-Collision (TTC)
            # Machine forward velocity is v_mach (+Y), pedestrian longitudinal velocity is vy.
            # Closing velocity: machine advancing towards pedestrian or pedestrian moving towards machine.
            v_closing = v_mach - vy
            ttc_sec = None

            if ty > 0 and v_closing > 0.2:
                ttc_sec = ty / v_closing
                if min_ttc is None or ttc_sec < min_ttc:
                    min_ttc = round(ttc_sec, 2)

            # 3. Crossing Trajectory Conflict Analysis
            # Project positions ahead in 0.5s increments up to 3.0s
            is_crossing_conflict = False
            half_w = self.mach_cfg.width_m / 2.0 + 0.5
            for t_step in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                proj_px = tx + vx * t_step
                proj_py = ty + vy * t_step
                proj_mach_y = v_mach * t_step
                # Check if projected pedestrian position intersects machine envelope
                if abs(proj_px) <= half_w and abs(proj_py - proj_mach_y) <= 1.5:
                    is_crossing_conflict = True
                    break

            # --- Evaluate state priorities ---

            cls_title = track.class_name.capitalize()

            # Priority 1: DANGER Conditions
            if zone == SafetyZoneType.DANGER_ZONE:
                highest_state = AlertState.DANGER
                primary_reason = (
                    f"{cls_title} {track.track_id} in DANGER_ZONE (distance: {dist:.1f}m, "
                    f"speed: {track.estimated_speed_mps:.1f}m/s)"
                )
                critical_zone = zone
                critical_track_id = track.track_id
                break  # Maximum priority reached

            if ttc_sec is not None and ttc_sec <= self.zone_cfg.time_to_collision_danger_sec and abs(tx) <= half_w + 1.0:
                highest_state = AlertState.DANGER
                primary_reason = (
                    f"Imminent collision: {cls_title} {track.track_id} TTC={ttc_sec:.1f}s "
                    f"(distance: {dist:.1f}m, closing velocity: {v_closing:.1f}m/s)"
                )
                critical_zone = SafetyZoneType.DANGER_ZONE
                critical_track_id = track.track_id
                break

            # Reverse conflict
            if telemetry.gear == "REVERSE" and ty < 0 and abs(ty) <= self.zone_cfg.blind_spot_rear_m:
                highest_state = AlertState.DANGER
                primary_reason = f"Rear collision risk: {cls_title} {track.track_id} behind reversing machine ({abs(ty):.1f}m)"
                critical_zone = SafetyZoneType.BLIND_SPOT_ZONE
                critical_track_id = track.track_id
                break

            # Priority 2: CAUTION Conditions
            if highest_state != AlertState.DANGER:
                if zone == SafetyZoneType.WARNING_ZONE:
                    highest_state = AlertState.CAUTION
                    primary_reason = (
                        f"{cls_title} {track.track_id} in WARNING_ZONE (distance: {dist:.1f}m)"
                    )
                    critical_zone = zone
                    critical_track_id = track.track_id

                elif zone == SafetyZoneType.BLIND_SPOT_ZONE:
                    highest_state = AlertState.CAUTION
                    primary_reason = (
                        f"{cls_title} {track.track_id} in BLIND_SPOT_ZONE (distance: {dist:.1f}m)"
                    )
                    critical_zone = zone
                    critical_track_id = track.track_id

                elif ttc_sec is not None and ttc_sec <= self.zone_cfg.time_to_collision_caution_sec:
                    highest_state = AlertState.CAUTION
                    primary_reason = (
                        f"Potential conflict: {cls_title} {track.track_id} TTC={ttc_sec:.1f}s (closing speed {v_closing:.1f}m/s)"
                    )
                    critical_zone = SafetyZoneType.WARNING_ZONE
                    critical_track_id = track.track_id

                elif is_crossing_conflict:
                    highest_state = AlertState.CAUTION
                    primary_reason = (
                        f"Crossing trajectory conflict: {cls_title} {track.track_id} moving across machine travel corridor"
                    )
                    critical_zone = SafetyZoneType.WARNING_ZONE
                    critical_track_id = track.track_id

        # Attach telemetry notice if applicable
        if telemetry_warning:
            primary_reason += telemetry_warning

        return (
            highest_state,
            primary_reason,
            critical_zone,
            critical_track_id,
            min_ttc,
            min_dist,
            telem_valid,
        )
