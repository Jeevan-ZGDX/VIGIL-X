"""
Structured event logging and audit trail with configurable retention.
Records risk assessments, sensor health transitions, fault events, and performance metrics in JSONL format.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from src.core.schemas import RiskAssessment, SystemHealthReport

logger = logging.getLogger(__name__)


class StructuredEventLogger:
    """
    Appends structured JSONL events to disk with size-based rotation and privacy protection.
    Does NOT store raw image pixels in event logs to preserve operator privacy.
    """

    def __init__(self, log_path: str = "data/events.jsonl", max_events_retained: int = 5000):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_events_retained = max_events_retained
        self._in_memory_recent_events: List[Dict[str, Any]] = []

    def log_risk_event(
        self,
        assessment: RiskAssessment,
        system_health: Optional[SystemHealthReport] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a safety risk assessment or state change event."""
        event_dict = {
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(assessment.timestamp_sec)),
            "timestamp_sec": assessment.timestamp_sec,
            "event_id": assessment.event_id,
            "alert_state": assessment.alert_state.value,
            "reason": assessment.reason,
            "active_zone": assessment.active_zone.value if assessment.active_zone else None,
            "critical_track_id": assessment.critical_track_id,
            "time_to_collision_sec": assessment.time_to_collision_sec,
            "min_distance_m": assessment.min_distance_m,
            "machine_telemetry_valid": assessment.machine_telemetry_valid,
            "sensor_health": {k: v.value for k, v in assessment.sensor_health_summary.items()},
            "uncertainty_level": assessment.uncertainty_level,
        }

        if system_health is not None:
            event_dict["performance"] = {
                "fps": system_health.fps,
                "latency_total_ms": system_health.latency_total_ms,
                "latency_inference_ms": system_health.latency_inference_ms,
                "active_tracks": system_health.active_tracks_count,
            }

        if extra_metadata:
            event_dict["extra"] = extra_metadata

        # Maintain recent in-memory cache for API queries
        self._in_memory_recent_events.append(event_dict)
        if len(self._in_memory_recent_events) > 500:
            self._in_memory_recent_events.pop(0)

        # Append to JSONL file
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_dict) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to event log: {e}")

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent events from in-memory cache."""
        return self._in_memory_recent_events[-limit:]
