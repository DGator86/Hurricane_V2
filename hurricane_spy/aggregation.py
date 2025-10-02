"""Aggregation utilities for Hurricane SPY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd


@dataclass
class StressWeightedGMV:
    """Stress-weighted Global Minimum Variance aggregator."""

    tikhonov: float
    stress_weight: float

    def __call__(
        self,
        forecasts: Mapping[str, Mapping[str, float]],
        covariance: pd.DataFrame,
        stress_level: float,
    ) -> Dict[str, object]:
        if covariance.empty:
            raise ValueError("Covariance matrix must not be empty")
        cov = covariance.copy()
        cov.values[np.diag_indices_from(cov.values)] += self.tikhonov
        inv_cov = np.linalg.pinv(cov.values)
        ones = np.ones(len(cov))
        weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
        # Apply stress weighting: increase emphasis on slower horizons when stress high
        stress_multiplier = 1 + self.stress_weight * max(stress_level, 0)
        stress_adjusted = weights / np.sum(weights)
        stress_adjusted[-1] *= stress_multiplier
        stress_adjusted /= np.sum(stress_adjusted)
        aggregated = {
            "support_resistance": float(
                sum(
                    stress_adjusted[i] * forecasts[name]["support_resistance"]
                    for i, name in enumerate(cov.index)
                )
            ),
            "direction_score": float(
                sum(
                    stress_adjusted[i] * forecasts[name]["direction_score"]
                    for i, name in enumerate(cov.index)
                )
            ),
            "speed": float(
                sum(stress_adjusted[i] * forecasts[name]["speed"] for i, name in enumerate(cov.index))
            ),
            "probability": float(
                sum(stress_adjusted[i] * forecasts[name]["probability"] for i, name in enumerate(cov.index))
            ),
            "hurricane_intensity": float(
                sum(
                    stress_adjusted[i] * forecasts[name]["hurricane_intensity"]
                    for i, name in enumerate(cov.index)
                )
            ),
            "weights": {
                name: float(stress_adjusted[i]) for i, name in enumerate(cov.index)
            },
        }
        return aggregated
