"""Configuration objects for the Hurricane SPY pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence


@dataclass
class TimeframeConfig:
    r"""Configuration for a single timeframe.

    Attributes
    ----------
    name:
        Human readable identifier of the timeframe (e.g. ``"1m"``).
    horizon_minutes:
        Forecast horizon expressed in minutes. Used for volatility scaling and
        horizon-aware modules such as finite-horizon barrier probabilities.
    abstention_threshold:
        Margin around 0.5 probability inside which the model abstains. The value
        corresponds to the ``\\delta`` term in the specification.
    lambda_level:
        Kernel half-width controlling the support/resistance potential.
    weights:
        Dictionary with entries ``gamma``, ``dark_pool`` and ``volume`` that
        weight the respective components of the level score ``S(L)``.
    speed_coefficients:
        Tuple ``(alpha_v, beta_v, chi_v)`` for the speed forecast amplifier.
    direction_threshold:
        Magnitude of the drift estimate above which the sign is emitted instead
        of abstaining.
    """

    name: str
    horizon_minutes: int
    abstention_threshold: float = 0.05
    lambda_level: float = 1.0
    weights: Mapping[str, float] = field(
        default_factory=lambda: {"gamma": 0.4, "dark_pool": 0.3, "volume": 0.3}
    )
    speed_coefficients: Sequence[float] = (0.25, 0.1, 0.05)
    direction_threshold: float = 0.0


@dataclass
class HurricaneConfig:
    """Configuration for the Hurricane SPY pipeline."""

    timeframes: Iterable[TimeframeConfig]
    hurricane_alpha: float = 0.6
    hurricane_beta: float = 0.4
    stress_weight: float = 0.35
    cusum_threshold: float = 3.0
    cusum_drift: float = 0.2
    hedging_pressure_limit: float = 0.7
    exogenous_flow_limit: float = 0.65
    cooling_window: int = 5
    gmvtikhonov: float = 1e-3

    def timeframe_by_name(self) -> Dict[str, TimeframeConfig]:
        """Return the timeframe configuration indexed by name."""

        return {tf.name: tf for tf in self.timeframes}
