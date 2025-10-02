"""Example entry-point for running the Hurricane SPY pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from hurricane_spy import HurricaneConfig, HurricaneSPY, TimeframeConfig
from hurricane_spy.data_structures import MarketDataBundle


def generate_dummy_data(index: pd.DatetimeIndex) -> MarketDataBundle:
    price = pd.DataFrame(
        {
            "open": 430 + np.cumsum(np.random.normal(0, 0.3, size=len(index))),
            "high": 431 + np.cumsum(np.random.normal(0, 0.35, size=len(index))),
            "low": 429 + np.cumsum(np.random.normal(0, 0.35, size=len(index))),
            "close": 430 + np.cumsum(np.random.normal(0, 0.3, size=len(index))),
            "volume": 1_000_000 + np.random.normal(0, 50_000, size=len(index)),
        },
        index=index,
    )
    greeks = pd.DataFrame(
        {
            "gamma": np.random.normal(0, 0.5, size=len(index)),
            "vanna": np.random.normal(0, 0.3, size=len(index)),
            "charm": np.random.normal(0, 0.2, size=len(index)),
        },
        index=index,
    )
    ofi = pd.DataFrame(
        {
            "ofi": np.random.normal(0, 1, size=len(index)),
            "dark_pool_index": np.random.normal(0, 0.2, size=len(index)).cumsum(),
            "exogenous_flow": np.random.normal(0, 0.5, size=len(index)),
            "variance_amplifier": np.random.uniform(0, 1, size=len(index)),
        },
        index=index,
    )
    technical = {
        "1m": pd.Series(np.tanh(np.random.normal(0, 0.8, size=len(index))), index=index),
        "5m": pd.Series(np.tanh(np.random.normal(0, 0.6, size=len(index))), index=index),
        "30m": pd.Series(np.tanh(np.random.normal(0, 0.4, size=len(index))), index=index),
    }
    levels_index = pd.Index([index[-1]] * 5)
    levels = pd.DataFrame(
        {
            "level": np.linspace(420, 440, 5),
            "gamma_score": np.random.uniform(-1, 1, 5),
            "dark_pool_score": np.random.uniform(-1, 1, 5),
            "volume_score": np.random.uniform(-1, 1, 5),
        },
        index=levels_index,
    )
    realised_vol = {
        "1m": pd.Series(np.random.uniform(0.5, 1.2, len(index)), index=index),
        "5m": pd.Series(np.random.uniform(0.4, 1.0, len(index)), index=index),
        "30m": pd.Series(np.random.uniform(0.3, 0.8, len(index)), index=index),
    }
    base_vol = {"1m": 0.4, "5m": 0.35, "30m": 0.25}
    barrier_levels = {
        "1m": {"lower": 420.0, "upper": 440.0},
        "5m": {"lower": 418.0, "upper": 442.0},
        "30m": {"lower": 410.0, "upper": 448.0},
    }
    events = pd.DataFrame({"is_event": [False] * len(index)}, index=index)
    stress_index = pd.Series(np.random.uniform(0, 1, len(index)), index=index)
    return MarketDataBundle(
        price=price,
        greeks=greeks,
        ofi=ofi,
        technical=technical,
        levels=levels,
        realised_vol=realised_vol,
        base_vol=base_vol,
        barrier_levels=barrier_levels,
        events=events,
        stress_index=stress_index,
    )


def main() -> None:
    index = pd.date_range(end=datetime.now(UTC), periods=500, freq="min")
    bundle = generate_dummy_data(index)
    config = HurricaneConfig(
        timeframes=[
            TimeframeConfig(name="1m", horizon_minutes=1, abstention_threshold=0.05, lambda_level=0.75),
            TimeframeConfig(name="5m", horizon_minutes=5, abstention_threshold=0.06, lambda_level=1.0),
            TimeframeConfig(name="30m", horizon_minutes=30, abstention_threshold=0.08, lambda_level=1.5),
        ]
    )
    pipeline = HurricaneSPY(config)
    result = pipeline.run(bundle)
    print("Aggregate Predictions:\n", result["aggregate"])
    print("\nPer Timeframe:")
    for name, payload in result["timeframes"].items():
        print(f"- {name}: signal={payload['direction_signal']}, speed={payload['speed']:.3f}")
    print("\nDiagnostics head:\n", result["diagnostics"]["gating"].head())


if __name__ == "__main__":
    main()
