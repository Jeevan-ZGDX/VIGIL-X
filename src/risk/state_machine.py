"""
Deterministic Alert State Machine with fault handling and hysteresis.
Controls transitions between NORMAL, CAUTION, DANGER, and DEGRADED states.
"""

from __future__ import annotations
import logging
import time
import uuid
from typing import Dict, List, Optional
from src.core.config import StateMachineConfig
from src.core.schemas import (
    AlertState,
    MachineTelemetry,
    RiskAssessment,
    SafetyZoneType,
    SensorHealthStatus,
    Track,
)
from src.risk.engine import CollisionRiskEngine

logger = logging.getLogger(__name__)


class AlertStateMachine:
    """
    Finite state machine governing pedestrian safety alert levels.
    Enforces asymmetric hysteresis: instantaneous hazard escalation,
    damped de-escalation, and immediate fault degradation.
    """

    def __init__(self, config: StateMachineConfig, risk_engine: CollisionRiskEngine):
        self.config = config
        self.risk_engine = risk_engine

        self.current_state: AlertState = AlertState.NORMAL
        self.current_reason: str = "System initialized in NORMAL state."

        # Hysteresis counters
        self._deescalation_counter: int = 0
        self._pending_lower_state: Optional[AlertState] = None
        self._degraded_recovery_counter: int = 0

    def evaluate(
        self,
        tracks: List[Track],
        telemetry: Optional[MachineTelemetry],
        sensor_health: Dict[str, SensorHealthStatus],
        timestamp_sec: float,
    ) -> RiskAssessment:
        """
        Evaluate inputs and determine system alert state.
        Never converts sensor loss or invalid inputs into an automatic all-clear.
        """
        # Step 1: Check hardware sensor health
        is_sensor_fault = False
        fault_reasons = []
        for s_id, status in sensor_health.items():
            if status in (SensorHealthStatus.DISCONNECTED, SensorHealthStatus.ERROR, SensorHealthStatus.STALE):
                is_sensor_fault = True
                fault_reasons.append(f"Sensor {s_id} is {status.value}")

        if is_sensor_fault:
            self.current_state = AlertState.DEGRADED
            self.current_reason = "DEGRADED: Sensor fault detected (" + ", ".join(fault_reasons) + ")"
            self._degraded_recovery_counter = 0
            return RiskAssessment(
                timestamp_sec=timestamp_sec,
                alert_state=AlertState.DEGRADED,
                reason=self.current_reason,
                event_id=str(uuid.uuid4())[:8],
                machine_telemetry_valid=telemetry.is_valid if telemetry else False,
                sensor_health_summary=sensor_health,
                uncertainty_level=1.0,
            )

        # If recovering from DEGRADED state
        if self.current_state == AlertState.DEGRADED:
            self._degraded_recovery_counter += 1
            if self._degraded_recovery_counter < self.config.degraded_recovery_frames:
                return RiskAssessment(
                    timestamp_sec=timestamp_sec,
                    alert_state=AlertState.DEGRADED,
                    reason=f"DEGRADED: Recovering sensor health ({self._degraded_recovery_counter}/{self.config.degraded_recovery_frames})",
                    event_id=str(uuid.uuid4())[:8],
                    machine_telemetry_valid=telemetry.is_valid if telemetry else False,
                    sensor_health_summary=sensor_health,
                    uncertainty_level=0.5,
                )
            else:
                self.current_state = AlertState.NORMAL
                self.current_reason = "Sensors recovered to HEALTHY state."

        # Step 2: Compute kinematic collision risk
        (
            candidate_state,
            candidate_reason,
            active_zone,
            critical_track_id,
            min_ttc,
            min_dist,
            telem_valid,
        ) = self.risk_engine.evaluate_risk(tracks, telemetry, timestamp_sec)

        # Step 3: Apply Hysteresis
        state_severity = {
            AlertState.NORMAL: 0,
            AlertState.CAUTION: 1,
            AlertState.DANGER: 2,
            AlertState.DEGRADED: 3,
        }

        curr_sev = state_severity[self.current_state]
        cand_sev = state_severity[candidate_state]

        if cand_sev > curr_sev:
            # Escalation is immediate
            self.current_state = candidate_state
            self.current_reason = candidate_reason
            self._deescalation_counter = 0
            self._pending_lower_state = None
        elif cand_sev < curr_sev:
            # De-escalation requires sustained absence of hazard
            threshold_frames = (
                self.config.danger_to_caution_frames
                if self.current_state == AlertState.DANGER
                else self.config.caution_to_normal_frames
            )

            if self._pending_lower_state == candidate_state:
                self._deescalation_counter += 1
            else:
                self._pending_lower_state = candidate_state
                self._deescalation_counter = 1

            if self._deescalation_counter >= threshold_frames:
                self.current_state = candidate_state
                self.current_reason = candidate_reason
                self._deescalation_counter = 0
                self._pending_lower_state = None
            else:
                # Retain current higher alert while de-escalation countdown runs
                pass
        else:
            # Same severity: update reason and reset counter
            self.current_reason = candidate_reason
            self._deescalation_counter = 0
            self._pending_lower_state = None

        return RiskAssessment(
            timestamp_sec=timestamp_sec,
            alert_state=self.current_state,
            reason=self.current_reason,
            event_id=str(uuid.uuid4())[:8],
            active_zone=active_zone,
            critical_track_id=critical_track_id,
            time_to_collision_sec=min_ttc,
            min_distance_m=min_dist,
            machine_telemetry_valid=telem_valid,
            sensor_health_summary=sensor_health,
            uncertainty_level=0.1 if self.current_state == AlertState.NORMAL else 0.3,
        )
