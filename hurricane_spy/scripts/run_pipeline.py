"""Example entry-point for running the Hurricane SPY pipeline."""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from hurricane_spy import (
    ExecutionConfig,
    HurricaneConfig,
    HurricaneSPY,
    TimeframeConfig,
    TradingExecutor,
    get_alpaca_client,
)
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Hurricane SPY pipeline and optionally trade via Alpaca.")
    parser.add_argument("--dry-run", action="store_true", help="Use the synthetic trading client instead of Alpaca.")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol to trade (default: SPY)")
    parser.add_argument("--base-quantity", type=int, default=10, help="Base order quantity before intensity scaling.")
    parser.add_argument("--max-position", type=int, default=200, help="Maximum quantity allowed per order.")
    parser.add_argument("--buy-threshold", type=float, default=0.6, help="Probability threshold for long trades.")
    parser.add_argument("--sell-threshold", type=float, default=0.4, help="Probability threshold for short trades.")
    parser.add_argument(
        "--intensity-multiplier",
        type=float,
        default=0.3,
        help="Multiplier applied to hurricane intensity when scaling order size.",
    )
    parser.add_argument(
        "--alpaca-mode",
        choices=["paper", "live"],
        default=None,
        help="Alpaca environment to target when not running in dry-run mode.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args()

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

    aggregate = result["aggregate"]
    print("Aggregate Predictions:\n", aggregate)
    print("\nPer Timeframe:")
    for name, payload in result["timeframes"].items():
        print(
            "- {name}: signal={signal}, speed={speed:.3f}, probability={prob:.2f}".format(
                name=name,
                signal=payload["direction_signal"],
                speed=payload["speed"],
                prob=payload["probability"],
            )
        )
    print("\nDiagnostics head:\n", result["diagnostics"]["gating"].head())

    client = get_alpaca_client(dry_run=args.dry_run, mode=args.alpaca_mode)
    exec_config = ExecutionConfig(
        symbol=args.symbol,
        base_quantity=args.base_quantity,
        max_position=args.max_position,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        intensity_multiplier=args.intensity_multiplier,
    )
    executor = TradingExecutor(client, exec_config)
    decision = executor.decide(aggregate)
    print(f"\nTrading decision: {decision}")
    executor.execute(decision)


if __name__ == "__main__":
    main()
