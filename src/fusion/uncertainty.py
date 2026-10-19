"""
Uncertainty and confidence fusion models for multimodal sensor observations.
Handles sensor disagreement, confidence blending, and environmental degradation.
"""

from __future__ import annotations
import math
from typing import Tuple
from src.core.schemas import SensorType


class UncertaintyModel:
    """
    Computes blended confidence and composite uncertainty for sensor fusion.
    Explicitly prevents treating mere sensor agreement as absolute certainty.
    """

    @staticmethod
    def fuse_confidence_and_uncertainty(
        rgb_conf: float,
        rgb_uncertainty: float,
        thermal_conf: float,
        thermal_uncertainty: float,
        rgb_weight: float = 0.50,
        thermal_weight: float = 0.50,
    ) -> Tuple[float, float]:
        """
        Bayesian-weighted confidence fusion with uncertainty propagation.
        Agreement boosts confidence, but residual uncertainty is bounded.
        """
        # Normalized weights
        total_w = rgb_weight + thermal_weight
        w_rgb = rgb_weight / total_w
        w_th = thermal_weight / total_w

        # Agreement factor: how close are the two confidences
        agreement = 1.0 - abs(rgb_conf - thermal_conf)

        # Probabilistic disjunction: P(A or B) = 1 - (1 - P(A))*(1 - P(B))
        prob_union = 1.0 - (1.0 - rgb_conf) * (1.0 - thermal_conf)

        # Weighted combination with agreement enhancement
        fused_conf = (w_rgb * rgb_conf + w_th * thermal_conf) * 0.40 + prob_union * 0.60
        fused_conf = max(0.0, min(1.0, fused_conf))

        # Fused uncertainty: combining independent sensors reduces uncertainty,
        # but minimum bounded uncertainty floor is maintained (e.g., 0.05).
        combined_var = 1.0 / ( (1.0 / max(0.01, rgb_uncertainty**2)) + (1.0 / max(0.01, thermal_uncertainty**2)) )
        fused_uncertainty = math.sqrt(combined_var)
        # Apply slight penalty if sensors disagree significantly
        disagreement_penalty = (1.0 - agreement) * 0.20
        fused_uncertainty = max(0.05, min(0.95, fused_uncertainty + disagreement_penalty))

        return round(fused_conf, 3), round(fused_uncertainty, 3)

    @staticmethod
    def penalize_single_sensor(
        confidence: float,
        uncertainty: float,
        penalty: float = 0.15,
    ) -> Tuple[float, float]:
        """
        When an object is detected in only one sensor, penalize confidence
        and inflate uncertainty because cross-sensor confirmation is absent.
        """
        adj_conf = max(0.0, confidence * (1.0 - penalty))
        adj_uncertainty = min(0.95, uncertainty + penalty)
        return round(adj_conf, 3), round(adj_uncertainty, 3)
