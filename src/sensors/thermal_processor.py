"""
Thermal frame normalization and radiometric temperature extraction.
Supports 16-bit radiometric Kelvin/centikelvin input, 8-bit grayscale,
CLAHE dynamic range optimization, and pseudo-color visualization.
"""

from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np


class ThermalProcessor:
    """
    Normalizes thermal images from various industrial thermal sensors
    (FLIR Lepton, Seek Thermal, MLX90640, or pre-recorded radiometric data).
    """

    def __init__(
        self,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        enable_clahe: bool = True,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
        default_colormap: int = cv2.COLORMAP_INFERNO,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.enable_clahe = enable_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.colormap = default_colormap

    def normalize_to_8bit(self, raw_thermal: np.ndarray) -> np.ndarray:
        """
        Normalize input thermal frame to standard uint8 [0, 255].
        Handles:
        - 16-bit radiometric uint16
        - Floating-point Celsius / Kelvin
        - 8-bit grayscale
        - 3-channel false-color
        """
        if raw_thermal is None or raw_thermal.size == 0:
            return np.zeros((self.target_height, self.target_width), dtype=np.uint8)

        # Check channels
        if raw_thermal.ndim == 3:
            if raw_thermal.shape[2] == 1:
                gray = raw_thermal[:, :, 0].copy()
            elif raw_thermal.shape[2] in (3, 4):
                gray = cv2.cvtColor(raw_thermal, cv2.COLOR_BGR2GRAY)
            else:
                gray = raw_thermal[:, :, 0].copy()
        else:
            gray = raw_thermal.copy()

        # Handle 16-bit radiometric
        if gray.dtype == np.uint16 or gray.dtype == np.int32:
            min_val = float(np.min(gray))
            max_val = float(np.max(gray))
            # Check if radiometric centikelvin (values > 10000, e.g. 29415 = 21 C)
            if min_val > 10000:
                # Dynamic auto-ranging with calibrated floor:
                if max_val > 30115:  # Target > 28 C (human or hot object)
                    calib_min = min(min_val, 29215.0)  # ~19 C
                    calib_max = max(max_val, 31015.0)  # ~37 C
                else:
                    # Ambient scene: map ambient range with natural contrast
                    calib_min = min_val - 20.0
                    calib_max = max(calib_min + 300.0, max_val + 20.0)
                norm = ((gray.astype(np.float32) - calib_min) / (calib_max - calib_min) * 255.0).clip(0, 255)
                gray_8u = norm.astype(np.uint8)
            else:
                if max_val > min_val:
                    norm = ((gray.astype(np.float32) - min_val) / (max_val - min_val) * 255.0).clip(0, 255)
                    gray_8u = norm.astype(np.uint8)
                else:
                    gray_8u = np.zeros_like(gray, dtype=np.uint8)
        elif gray.dtype in (np.float32, np.float64):
            min_val = float(np.min(gray))
            max_val = float(np.max(gray))
            if max_val > min_val:
                norm = ((gray - min_val) / (max_val - min_val) * 255.0).clip(0, 255)
                gray_8u = norm.astype(np.uint8)
            else:
                gray_8u = np.zeros_like(gray, dtype=np.uint8)
        else:
            gray_8u = gray.astype(np.uint8)

        # Apply CLAHE only if scene exhibits significant temperature dynamic range
        if self.enable_clahe and float(np.std(gray_8u)) > 15.0:
            gray_8u = self.clahe.apply(gray_8u)

        # Resize to standard target dimensions only if specified
        if self.target_width is not None and self.target_height is not None:
            if gray_8u.shape[0] != self.target_height or gray_8u.shape[1] != self.target_width:
                gray_8u = cv2.resize(gray_8u, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

        return gray_8u

    def to_pseudocolor(self, gray_8u: np.ndarray) -> np.ndarray:
        """Convert 8-bit normalized thermal image to 3-channel pseudo-color for operator display."""
        if gray_8u.ndim == 3:
            return gray_8u
        return cv2.applyColorMap(gray_8u, self.colormap)

    def extract_pedestrian_heat_mask(
        self,
        gray_8u: np.ndarray,
        low_percentile: float = 70.0,
        high_percentile: float = 99.5,
    ) -> np.ndarray:
        """
        Segment elevated heat signatures (warm bodies) against cooler industrial floors/machinery.
        Uses adaptive percentile thresholding and morphological cleanup.
        """
        # If no pixels reach warm human body temperature (> 30 C, ~135 intensity on calibrated scale)
        # or if the image has negligible standard deviation, suppress false contours completely
        if float(np.max(gray_8u)) < 135 or float(np.std(gray_8u)) < 8.0:
            return np.zeros_like(gray_8u)

        p_low = np.percentile(gray_8u, low_percentile)
        p_high = np.percentile(gray_8u, high_percentile)

        # Mask pixels in the elevated body temperature range
        _, thresh = cv2.threshold(gray_8u, int(max(p_low, 135)), 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return cleaned
