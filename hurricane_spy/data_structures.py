"""Data structures used by the Hurricane SPY pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class MarketDataBundle:
    r"""Container holding the data required for running the pipeline.

    Attributes
    ----------
    price:
        OHLCV data indexed by timestamp with columns ``[open, high, low, close,
        volume]``.
    greeks:
        Options metrics indexed by timestamp with columns ``[gamma, vanna, charm]``.
    ofi:
        Order-flow imbalance data indexed by timestamp with at least an
        ``ofi`` column and optionally ``dark_pool_index`` and ``exogenous_flow``.
    technical:
        Technical alignment signals per timeframe. Expected format is a mapping
        from timeframe name to a time-indexed series of alignment scores in
        ``[-1, 1]``.
    levels:
        Support/resistance candidate levels indexed by timestamp with columns
        ``[level, gamma_score, dark_pool_score, volume_score]``.
    realised_vol:
        Realised volatility estimates per timeframe. Mapping from timeframe name
        to series of non-negative floats.
    base_vol:
        Baseline volatility estimates (``\\sigma_{0,T}``) per timeframe.
    barrier_levels:
        Optional mapping from timeframe name to dictionaries containing
        ``upper`` and ``lower`` barrier levels for barrier hitting probability
        calculations.
    events:
        Optional time-indexed frame with binary columns ``[is_event]`` indicating
        market events that trigger abstention.
    stress_index:
        Optional series representing the macro stress level used for
        stress-weighted aggregation.
    """

    price: pd.DataFrame
    greeks: pd.DataFrame
    ofi: pd.DataFrame
    technical: Mapping[str, pd.Series]
    levels: pd.DataFrame
    realised_vol: Mapping[str, pd.Series]
    base_vol: Mapping[str, float]
    barrier_levels: Optional[Mapping[str, Mapping[str, float]]] = None
    events: Optional[pd.DataFrame] = None
    stress_index: Optional[pd.Series] = None

    def latest(self) -> Dict[str, pd.Series]:
        """Return the latest observation for each dataframe-like input."""

        latest_data = {
            "price": self.price.iloc[-1],
            "greeks": self.greeks.iloc[-1],
            "ofi": self.ofi.iloc[-1],
            "levels": self.levels[self.levels.index == self.levels.index.max()],
        }
        if self.events is not None:
            latest_data["events"] = self.events.iloc[-1]
        if self.stress_index is not None:
            latest_data["stress"] = self.stress_index.iloc[-1]
        return latest_data

    def validate(self, timeframes: Iterable[str]) -> None:
        """Validate that the bundle contains the expected inputs."""

        required_price_cols = {"open", "high", "low", "close", "volume"}
        required_greeks_cols = {"gamma", "vanna", "charm"}
        required_levels_cols = {
            "level",
            "gamma_score",
            "dark_pool_score",
            "volume_score",
        }
        if not required_price_cols.issubset(self.price.columns):
            missing = required_price_cols - set(self.price.columns)
            raise ValueError(f"Missing price columns: {missing}")
        if not required_greeks_cols.issubset(self.greeks.columns):
            missing = required_greeks_cols - set(self.greeks.columns)
            raise ValueError(f"Missing greek columns: {missing}")
        if not required_levels_cols.issubset(self.levels.columns):
            missing = required_levels_cols - set(self.levels.columns)
            raise ValueError(f"Missing level columns: {missing}")
        if "ofi" not in self.ofi.columns:
            raise ValueError("Order-flow imbalance data must include an 'ofi' column")
        for tf in timeframes:
            if tf not in self.realised_vol:
                raise ValueError(f"Realised volatility missing for timeframe {tf}")
            if tf not in self.base_vol:
                raise ValueError(f"Base volatility missing for timeframe {tf}")
            if np.any(self.realised_vol[tf] < 0):
                raise ValueError(f"Realised volatility contains negatives for {tf}")
