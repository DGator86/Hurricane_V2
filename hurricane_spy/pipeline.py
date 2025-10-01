"""Pipeline orchestration for the Hurricane SPY algorithm."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from .aggregation import StressWeightedGMV
from .config import HurricaneConfig, TimeframeConfig
from .data_structures import MarketDataBundle
from .diagnostics import GatingDiagnostics, ReliabilityTracker
from .features import (
    DirectionInputs,
    conformal_interval,
    direction_estimate,
    direction_signal,
    expected_abs_normal,
    finite_horizon_barrier_probability,
    hurricane_intensity,
    speed_forecast,
    support_resistance_potential,
)
from .gating import (
    CusumRegimeFlipDetector,
    EventAbstention,
    ExogenousFlowGate,
    HedgingPressureGate,
    apply_gates,
)


class HurricaneSPY:
    """End-to-end execution of the Hurricane SPY inference pipeline."""

    def __init__(self, config: HurricaneConfig) -> None:
        self.config = config
        self.aggregator = StressWeightedGMV(
            tikhonov=config.gmvtikhonov, stress_weight=config.stress_weight
        )
        self.reliability = ReliabilityTracker()
        self.gating_diagnostics = GatingDiagnostics()

    def _build_gates(self) -> Dict[str, object]:
        return {
            "event": EventAbstention(self.config.cooling_window),
            "regime": CusumRegimeFlipDetector(
                threshold=self.config.cusum_threshold, drift=self.config.cusum_drift
            ),
            "flow": ExogenousFlowGate(self.config.exogenous_flow_limit),
            "hedging": HedgingPressureGate(self.config.hedging_pressure_limit),
        }

    def run(self, data: MarketDataBundle) -> Dict[str, object]:
        """Execute the Hurricane SPY pipeline on the provided data bundle."""

        timeframe_names = [tf.name for tf in self.config.timeframes]
        data.validate(timeframe_names)
        latest = data.latest()
        price_series = data.price["close"]
        gates = self._build_gates()
        per_timeframe: Dict[str, Dict[str, object]] = {}
        covariance_entries = []

        for tf in self.config.timeframes:
            tf_result = self._run_timeframe(tf, data, latest, price_series, gates)
            per_timeframe[tf.name] = tf_result
            covariance_entries.append(tf_result["speed_vol_proxy"])

        covariance = pd.DataFrame(
            np.diag(covariance_entries), index=timeframe_names, columns=timeframe_names
        )
        stress_level = float(latest.get("stress", 0.0))
        aggregated = self.aggregator(
            {name: per_timeframe[name] for name in timeframe_names}, covariance, stress_level
        )
        diagnostics = {
            "gating": self.gating_diagnostics.to_frame(),
            "reliability": self.reliability.summary(),
        }
        return {
            "timeframes": per_timeframe,
            "aggregate": aggregated,
            "diagnostics": diagnostics,
        }

    def _run_timeframe(
        self,
        tf: TimeframeConfig,
        data: MarketDataBundle,
        latest: Mapping[str, pd.Series],
        price_series: pd.Series,
        gates: Mapping[str, object],
    ) -> Dict[str, object]:
        timestamp = price_series.index[-1]
        price = float(latest["price"]["close"])
        levels = data.levels[data.levels.index == data.levels.index.max()]
        sr_potential = support_resistance_potential(
            price, levels, tf.weights, tf.lambda_level
        )
        nearest_position = (levels["level"] - price).abs().values.argmin()
        nearest_level = float(levels.iloc[nearest_position]["level"])
        gap_to_level = float(abs(price - nearest_level) / max(price, 1e-6))

        greeks_row = latest["greeks"]
        ofi_row = latest["ofi"]
        sign_gex = np.sign(greeks_row.get("gamma", 0.0))
        dix_series = data.ofi["dark_pool_index"] if "dark_pool_index" in data.ofi else None
        delta_dix = float(dix_series.diff().iloc[-1]) if dix_series is not None else 0.0
        technical_alignment = float(data.technical[tf.name].iloc[-1])
        ofi_value = float(ofi_row["ofi"])
        theta = (0.35, 0.25, 0.2, 0.2)
        mu = direction_estimate(
            DirectionInputs(
                sign_gex=sign_gex,
                delta_dix=float(delta_dix) if not np.isnan(delta_dix) else 0.0,
                technical_alignment=technical_alignment,
                ofi=ofi_value,
                theta=theta,
                threshold=tf.direction_threshold,
            )
        )
        realised_vol = float(data.realised_vol[tf.name].iloc[-1])
        base_vol = float(data.base_vol[tf.name])
        avg_gamma = float(np.tanh(abs(greeks_row.get("gamma", 0.0))))
        intensity = hurricane_intensity(
            realised_vol, base_vol, mu, avg_gamma, self.config.hurricane_alpha, self.config.hurricane_beta
        )
        upsilon = float(abs(ofi_row.get("variance_amplifier", ofi_value)))
        alpha_v, beta_v, chi_v = tf.speed_coefficients
        speed = speed_forecast(
            realised_vol, alpha_v, beta_v, chi_v, upsilon, gap_to_level, intensity
        )
        signal = direction_signal(mu, tf.abstention_threshold)
        probability = float(0.5 + 0.5 * np.tanh(mu))

        barrier_prob = None
        if data.barrier_levels and tf.name in data.barrier_levels:
            barrier_prob = finite_horizon_barrier_probability(
                price=price,
                drift=mu,
                vol=realised_vol,
                barrier=data.barrier_levels[tf.name],
                horizon=tf.horizon_minutes / 60.0,
            )

        residuals = np.abs(data.price["close"].diff().dropna())
        conformal_width = conformal_interval(residuals.tail(250), alpha=0.1)

        expected_move = expected_abs_normal(mu, realised_vol + 1e-6)

        gate_results = apply_gates(
            timestamp=timestamp,
            events=latest.get("events", pd.Series(dtype=float)),
            price_history=price_series,
            ofi_row=ofi_row,
            greeks_row=greeks_row,
            gates=gates,
        )
        self.gating_diagnostics.log(timestamp, gate_results)
        blocked = any(bool(flag) for flag in gate_results.values())
        if blocked:
            signal = "abstain"
        self.reliability.update("storm" if intensity >= 4 else "trend", probability, 1 if mu > 0 else 0)

        return {
            "timestamp": timestamp,
            "support_resistance": sr_potential,
            "direction_score": mu,
            "direction_signal": signal,
            "probability": probability,
            "speed": speed,
            "speed_vol_proxy": realised_vol + intensity,
            "hurricane_intensity": intensity,
            "gap_to_level": gap_to_level,
            "expected_move": expected_move,
            "conformal_width": conformal_width,
            "barrier_hit_probability": barrier_prob,
            "gates": gate_results,
            "config": asdict(tf),
        }
