"""
Abstract machine interface contract for external industrial equipment communication.
Defines explicit non-actuating warning signal exchanges and telemetry ingestion.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from src.core.schemas import MachineTelemetry, RiskAssessment


class BaseMachineAdapter(ABC):
    """
    Contract for equipment integration.
    Safety principle: This interface only issues informational and advisory warning signals.
    It does NOT directly actuate emergency brakes or steering.
    """

    @abstractmethod
    def get_telemetry(self) -> MachineTelemetry:
        """Fetch current telemetry from equipment CAN bus, serial, or telemetry stream."""
        pass

    @abstractmethod
    def send_warning_event(self, assessment: RiskAssessment) -> bool:
        """Transmit safety warning state to cabin display and warning devices."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Return connectivity status of the machine integration link."""
        pass
