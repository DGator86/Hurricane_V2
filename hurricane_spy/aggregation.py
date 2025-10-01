"""Aggregation utilities for Hurricane SPY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

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
        weights = {name: float(stress_adjusted[i]) for i, name in enumerate(cov.index)}

        aggregated: Dict[str, Any] = {"weights": weights}
        keys_to_aggregate = (
            "support_resistance",
            "direction_score",
            "speed",
            "probability",
            "hurricane_intensity",
        )
        for key in keys_to_aggregate:
            if all(key in forecasts[name] for name in cov.index):
                aggregated[key] = float(
                    sum(stress_adjusted[i] * forecasts[name][key] for i, name in enumerate(cov.index))
                )

        return aggregated
