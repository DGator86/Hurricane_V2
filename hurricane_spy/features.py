"""Feature engineering primitives for the Hurricane SPY pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class DirectionInputs:
    sign_gex: float
    delta_dix: float
    technical_alignment: float
    ofi: float
    theta: Tuple[float, float, float, float]
    threshold: float


def hurricane_intensity(
    realised_vol: float,
    base_vol: float,
    drift: float,
    avg_gamma: float,
    alpha: float,
    beta: float,
) -> int:
    r"""Compute the discrete hurricane intensity scale.

    Parameters
    ----------
    realised_vol:
        Observed volatility (``\\sigma_{t,T}``) for timeframe ``T``.
    base_vol:
        Baseline volatility (``\\sigma_{0,T}``) used for comparison.
    drift:
        Absolute value of the drift estimate ``|\mu_{t,T}|``.
    avg_gamma:
        Average gamma exposure normalised to ``[0, 1]``.
    alpha, beta:
        Scaling coefficients as per the whitepaper.
    """

    if realised_vol <= 0 or base_vol <= 0:
        raise ValueError("Volatility inputs must be positive")
    log_ratio = np.log(realised_vol / base_vol) / np.log(2)
    intensity = np.floor(
        np.clip(log_ratio + alpha * np.abs(drift) + beta * avg_gamma, 0.0, 5.0)
    )
    return int(intensity)


def level_score(level_row: pd.Series, weights: Mapping[str, float]) -> float:
    """Compute the composite score for a single price level."""

    return (
        weights.get("gamma", 0.0) * level_row.get("gamma_score", 0.0)
        + weights.get("dark_pool", 0.0) * level_row.get("dark_pool_score", 0.0)
        + weights.get("volume", 0.0) * level_row.get("volume_score", 0.0)
    )


def support_resistance_potential(
    price: float, levels: pd.DataFrame, weights: Mapping[str, float], lambda_level: float
) -> float:
    r"""Compute the support/resistance potential field ``\\Phi(p)``."""

    if lambda_level <= 0:
        raise ValueError("lambda_level must be positive")
    scores = levels.apply(level_score, axis=1, weights=weights)
    distances = (price - levels["level"]) / lambda_level
    kernel = 1.0 / (1.0 + distances**2)
    return float(np.sum(scores * kernel))


def direction_estimate(inputs: DirectionInputs) -> float:
    r"""Estimate the signed drift ``\\mu_{t,T}`` using the provided inputs."""

    theta_1, theta_2, theta_3, theta_4 = inputs.theta
    mu = (
        theta_1 * inputs.sign_gex
        + theta_2 * inputs.delta_dix
        + theta_3 * inputs.technical_alignment
        + theta_4 * inputs.ofi
    )
    if np.abs(mu) < inputs.threshold:
        return 0.0
    return float(mu)


def direction_signal(mu: float, abstention_threshold: float) -> str:
    """Convert a drift estimate into an actionable signal with abstention."""

    if np.abs(mu) <= abstention_threshold:
        return "abstain"
    return "up" if mu > 0 else "down"


def speed_forecast(
    realised_vol: float,
    alpha_v: float,
    beta_v: float,
    chi_v: float,
    upsilon: float,
    gap_to_level: float,
    hurricane_intensity: float,
) -> float:
    """Compute the expected magnitude per unit time (speed forecast)."""

    if realised_vol < 0:
        raise ValueError("Realised volatility must be non-negative")
    return float(
        realised_vol * (1 + alpha_v * upsilon + beta_v * gap_to_level + chi_v * hurricane_intensity)
    )


def finite_horizon_barrier_probability(
    price: float,
    drift: float,
    vol: float,
    barrier: Mapping[str, float],
    horizon: float,
) -> float:
    """Finite-horizon probability of hitting the lower barrier before expiry."""

    if vol <= 0 or horizon <= 0:
        raise ValueError("vol and horizon must be positive")
    lower = barrier["lower"]
    upper = barrier.get("upper", np.inf)
    if not np.isfinite(upper):
        return float(1.0)
    numerator = norm.cdf((lower - price - drift * horizon) / (vol * np.sqrt(horizon)))
    exp_term = np.exp(2 * drift * (lower - price) / (vol**2))
    second_term = exp_term * norm.cdf((lower - price + drift * horizon) / (vol * np.sqrt(horizon)))
    return float(1 - (numerator - second_term))


def expected_abs_normal(mu: float, sigma: float) -> float:
    """Expected absolute value of a normal random variable."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(
        sigma * np.sqrt(2 / np.pi) * np.exp(-(mu**2) / (2 * sigma**2))
        + mu * (1 - 2 * norm.cdf(-mu / sigma))
    )


def conformal_interval(residuals: Iterable[float], alpha: float) -> float:
    """Return the (1-alpha) conformal quantile width."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    residuals = np.abs(np.array(list(residuals)))
    if residuals.size == 0:
        return 0.0
    quantile_kwargs = {"method": "higher"}
    try:
        q = np.quantile(residuals, 1 - alpha, **quantile_kwargs)
    except TypeError:  # pragma: no cover - numpy<1.22 fallback
        q = np.quantile(residuals, 1 - alpha, interpolation="higher")
    return float(q)


def brier_score(probability: float, outcome: float) -> float:
    """Compute the Brier score for a binary outcome."""

    if not 0 <= probability <= 1:
        raise ValueError("probability must be within [0,1]")
    if outcome not in (0, 1):
        raise ValueError("outcome must be binary")
    return float((probability - outcome) ** 2)
