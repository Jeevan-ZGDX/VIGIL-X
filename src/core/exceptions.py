"""
Custom exception definitions for sensor, calibration, detection, and pipeline faults.
"""

class PedestrianSafetyError(Exception):
    """Base exception for all domain errors."""
    pass


class ConfigurationError(PedestrianSafetyError):
    """Raised when configuration validation fails."""
    pass


class SensorError(PedestrianSafetyError):
    """Raised when camera capture, read, or hardware access fails."""
    pass


class SensorDisconnectedError(SensorError):
    """Raised when a camera disconnects unexpectedly."""
    pass


class StaleFrameError(SensorError):
    """Raised when frame timestamps exceed freshness thresholds."""
    pass


class CalibrationError(PedestrianSafetyError):
    """Raised when camera calibration parameters or transformations are invalid."""
    pass


class InferenceError(PedestrianSafetyError):
    """Raised when model inference or vision detection fails."""
    pass


class MachineIntegrationError(PedestrianSafetyError):
    """Raised when telemetry communication or warning interface fails."""
    pass
