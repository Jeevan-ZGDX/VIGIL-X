"""
Machine integration simulator for testing without live machinery.
Simulates telemetry feeds, audio/visual warning enunciators, and fault injection.
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional
from src.core.config import MachineConfig
from src.core.schemas import AlertState, MachineTelemetry, RiskAssessment
from src.integration.machine_adapter import BaseMachineAdapter

logger = logging.getLogger(__name__)


class MachineIntegrationSimulator(BaseMachineAdapter):
    """
    Simulated machine interface for development, unit testing, and demonstration.
    Tracks received warning events and simulates hardware warning device states.
    """

    def __init__(self, config: MachineConfig):
        self.config = config
        self._connected = True
        self._simulated_speed_mps = 0.0
        self._simulated_gear = "FORWARD"
        self._simulated_direction_deg = 0.0

        # Physical warning device actuator simulation
        self.visual_strobe_active = False
        self.audio_buzzer_hz = 0  # 0=off, 800=pulsing caution, 2500=continuous danger
        self.cabin_display_text = "READY - SYSTEM OK"
        self.advisory_slowdown_requested = False

        self.received_events: List[Dict[str, Any]] = []

    def set_machine_state(self, speed_mps: float, gear: str = "FORWARD", direction_deg: float = 0.0):
        """Update simulated vehicle kinematics."""
        self._simulated_speed_mps = float(speed_mps)
        self._simulated_gear = gear
        self._simulated_direction_deg = float(direction_deg)

    def set_connection_fault(self, is_fault: bool):
        """Inject or clear communication link failure."""
        self._connected = not is_fault

    def is_connected(self) -> bool:
        return self._connected

    def get_telemetry(self) -> MachineTelemetry:
        now = time.time()
        if not self._connected:
            return MachineTelemetry(
                timestamp_sec=now,
                machine_id=self.config.machine_id,
                is_valid=False,
                speed_mps=0.0,
                gear="NEUTRAL",
                width_m=self.config.width_m,
                length_m=self.config.length_m,
            )

        return MachineTelemetry(
            timestamp_sec=now,
            machine_id=self.config.machine_id,
            is_valid=True,
            speed_mps=self._simulated_speed_mps,
            direction_deg=self._simulated_direction_deg,
            gear=self._simulated_gear,
            width_m=self.config.width_m,
            length_m=self.config.length_m,
        )

    def send_warning_event(self, assessment: RiskAssessment) -> bool:
        if not self._connected:
            logger.warning("Failed to send warning: Machine interface disconnected")
            return False

        # Translate safety state into warning outputs
        if assessment.alert_state == AlertState.DANGER:
            self.visual_strobe_active = True
            self.audio_buzzer_hz = 2500  # High pitch continuous alarm
            self.cabin_display_text = f"CRITICAL: {assessment.reason}"
            self.advisory_slowdown_requested = True
        elif assessment.alert_state == AlertState.CAUTION:
            self.visual_strobe_active = True
            self.audio_buzzer_hz = 800   # Pulsing caution chime
            self.cabin_display_text = f"CAUTION: {assessment.reason}"
            self.advisory_slowdown_requested = False
        elif assessment.alert_state == AlertState.DEGRADED:
            self.visual_strobe_active = True
            self.audio_buzzer_hz = 400   # Intermittent fault beep
            self.cabin_display_text = f"SYSTEM DEGRADED: {assessment.reason}"
            self.advisory_slowdown_requested = False
        else:
            # NORMAL
            self.visual_strobe_active = False
            self.audio_buzzer_hz = 0
            self.cabin_display_text = "READY - NO HAZARD DETECTED"
            self.advisory_slowdown_requested = False

        record = {
            "timestamp_sec": assessment.timestamp_sec,
            "event_id": assessment.event_id,
            "alert_state": assessment.alert_state.value,
            "reason": assessment.reason,
            "strobe": self.visual_strobe_active,
            "buzzer_hz": self.audio_buzzer_hz,
            "display": self.cabin_display_text,
            "slowdown": self.advisory_slowdown_requested,
        }
        self.received_events.append(record)
        if len(self.received_events) > 100:
            self.received_events.pop(0)

        return True
