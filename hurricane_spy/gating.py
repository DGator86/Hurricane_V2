"""Stability and abstention modules for Hurricane SPY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping

import pandas as pd


@dataclass
class EventAbstention:
    """Event-aware abstention logic."""

    cooling_window: int

    def __call__(self, events: pd.Series, timestamp: pd.Timestamp) -> bool:
        if events is None or events.empty or "is_event" not in events.index:
            return False
        if events["is_event"]:
            return True
        if "minutes_to_event" in events.index:
            return events["minutes_to_event"] <= self.cooling_window
        return False


@dataclass
class CusumRegimeFlipDetector:
    """CUSUM-based regime flip detector."""

    threshold: float
    drift: float

    def update(self, series: pd.Series) -> float:
        returns = series.diff().dropna()
        if returns.empty:
            return 0.0
        g_pos = 0.0
        g_neg = 0.0
        flip_signal = 0.0
        for r in returns:
            g_pos = max(0.0, g_pos + r - self.drift)
            g_neg = min(0.0, g_neg + r + self.drift)
            if g_pos > self.threshold:
                flip_signal = max(flip_signal, g_pos)
                g_pos = 0.0
            if abs(g_neg) > self.threshold:
                flip_signal = min(flip_signal, g_neg)
                g_neg = 0.0
        return flip_signal


@dataclass
class ExogenousFlowGate:
    limit: float

    def __call__(self, ofi_row: pd.Series) -> bool:
        value = ofi_row.get("exogenous_flow", 0.0)
        return abs(value) > self.limit


@dataclass
class HedgingPressureGate:
    limit: float

    def __call__(self, greeks_row: pd.Series) -> bool:
        gamma = abs(greeks_row.get("gamma", 0.0))
        vanna = abs(greeks_row.get("vanna", 0.0))
        charm = abs(greeks_row.get("charm", 0.0))
        pressure = gamma + 0.5 * vanna + 0.25 * charm
        return pressure > self.limit


def apply_gates(
    timestamp: pd.Timestamp,
    events: pd.Series,
    price_history: pd.Series,
    ofi_row: pd.Series,
    greeks_row: pd.Series,
    gates: Mapping[str, object],
) -> Dict[str, bool]:
    """Evaluate all gating modules and return their boolean status."""

    results: MutableMapping[str, bool] = {}
    event_gate = gates.get("event")
    if isinstance(event_gate, EventAbstention):
        results["event"] = event_gate(events, timestamp)
    regime_gate = gates.get("regime")
    if isinstance(regime_gate, CusumRegimeFlipDetector):
        results["regime_flip"] = regime_gate.update(price_history)
    flow_gate = gates.get("flow")
    if isinstance(flow_gate, ExogenousFlowGate):
        results["exogenous_flow"] = flow_gate(ofi_row)
    hedging_gate = gates.get("hedging")
    if isinstance(hedging_gate, HedgingPressureGate):
        results["hedging_pressure"] = hedging_gate(greeks_row)
    return dict(results)
