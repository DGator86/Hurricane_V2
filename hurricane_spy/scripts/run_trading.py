"""Example script wiring Hurricane SPY predictions to Alpaca execution."""

from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from datetime import UTC, datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import requests

from hurricane_spy import (
    AlpacaClient,
    AlpacaCredentials,
    HurricaneConfig,
    HurricaneSPY,
    HurricaneTrader,
    TimeframeConfig,
    TradingConfig,
    UnusualWhalesClient,
    assemble_bundle,
    merge_unusual_whales_signals,
)
from hurricane_spy.data_sources import align_series
from hurricane_spy.data_structures import MarketDataBundle
from hurricane_spy.scripts.run_pipeline import generate_dummy_data

LOGGER = logging.getLogger(__name__)


def fetch_alpaca_bars(
    credentials: AlpacaCredentials,
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch 1-minute bars from the Alpaca market data API."""

    data_base_url = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")
    url = f"{data_base_url.rstrip('/')}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Min",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "adjustment": "raw",
        "limit": 10_000,
    }
    headers = {
        "APCA-API-KEY-ID": credentials.api_key,
        "APCA-API-SECRET-KEY": credentials.secret_key,
    }
    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    bars = response.json().get("bars", [])
    frame = pd.DataFrame(bars)
    if frame.empty:
        raise ValueError("Alpaca data API returned no bars")
    frame["timestamp"] = pd.to_datetime(frame["t"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return frame[["open", "high", "low", "close", "volume"]].astype(float)


def build_live_bundle(
    *,
    symbol: str,
    credentials: AlpacaCredentials,
    lookback: timedelta,
    seed: int | None,
) -> Tuple[MarketDataBundle, bool]:
    """Attempt to build a MarketDataBundle from live data sources."""

    token = os.getenv("UNUSUAL_WHALES_API_TOKEN")
    end = datetime.now(UTC)
    start = end - lookback
    if not token:
        return (
            generate_dummy_data(pd.date_range(end=end, periods=500, freq="min"), seed=seed),
            False,
        )

    client = UnusualWhalesClient(api_token=token)
    try:
        flow = client.fetch_options_flow(symbol, start=start, end=end, limit=5000)
        dark_pool = client.fetch_dark_pool_activity(symbol, start=start, end=end)
        gex = client.fetch_gamma_exposure(symbol, start=start, end=end)
        ofi = merge_unusual_whales_signals(flow=flow, dark_pool=dark_pool, gex=gex)
    except Exception as exc:  # pragma: no cover - network
        LOGGER.warning("Failed to load Unusual Whales data (%s); falling back to dummy bundle", exc)
        return (
            generate_dummy_data(pd.date_range(end=end, periods=500, freq="min"), seed=seed),
            False,
        )

    try:
        price = fetch_alpaca_bars(credentials, symbol, start=start, end=end)
    except Exception as exc:  # pragma: no cover - network
        LOGGER.warning("Failed to load Alpaca price history (%s); falling back to dummy bundle", exc)
        return (
            generate_dummy_data(pd.date_range(end=end, periods=500, freq="min"), seed=seed),
            False,
        )

    shared_index = align_series([price["close"], ofi["ofi"]], rule="1min")
    price = price.resample("1min").last().reindex(shared_index).interpolate().ffill()
    ofi = ofi.reindex(shared_index).ffill().bfill()

    greeks_raw = gex.copy()
    if "timestamp" in greeks_raw:
        greeks_raw["timestamp"] = pd.to_datetime(greeks_raw["timestamp"], utc=True)
        greeks_raw = greeks_raw.set_index("timestamp").sort_index()
    greeks_raw = greeks_raw.resample("1min").mean().reindex(shared_index).interpolate().fillna(0.0)
    greeks = pd.DataFrame(index=shared_index)
    greeks["gamma"] = greeks_raw.filter(regex="gamma|gex", axis=1).sum(axis=1)
    greeks["vanna"] = greeks_raw.filter(regex="vanna|delta", axis=1).sum(axis=1)
    greeks["charm"] = greeks_raw.filter(regex="charm|theta", axis=1).sum(axis=1)
    greeks = greeks.fillna(0.0)

    close = price["close"]
    returns = close.pct_change().fillna(0.0)
    realised_vol = {
        "1m": returns.rolling(30).std().fillna(method="bfill") * np.sqrt(390),
        "5m": returns.rolling(150).std().fillna(method="bfill") * np.sqrt(390 / 5),
        "30m": returns.rolling(900).std().fillna(method="bfill") * np.sqrt(390 / 30),
    }
    base_vol = {"1m": 0.35, "5m": 0.3, "30m": 0.25}

    def _technical(window: int) -> pd.Series:
        ma = close.rolling(window).mean()
        std = close.rolling(window).std().replace(0, np.nan)
        score = (close - ma) / std
        return score.clip(-1, 1).fillna(0.0)

    technical = {
        "1m": _technical(20),
        "5m": _technical(60),
        "30m": _technical(180),
    }

    last_close = float(close.iloc[-1])
    levels = pd.DataFrame(
        {
            "level": np.linspace(last_close * 0.95, last_close * 1.05, 5),
            "gamma_score": np.linspace(-1, 1, 5),
            "dark_pool_score": np.tanh(ofi["dark_pool_index"].iloc[-1] / 1_000 if "dark_pool_index" in ofi else 0.0),
            "volume_score": np.linspace(-0.5, 0.5, 5),
        },
        index=pd.Index([shared_index[-1]] * 5),
    )

    barrier_levels = {
        "1m": {"lower": last_close * 0.97, "upper": last_close * 1.03},
        "5m": {"lower": last_close * 0.95, "upper": last_close * 1.05},
        "30m": {"lower": last_close * 0.9, "upper": last_close * 1.1},
    }
    events = pd.DataFrame({"is_event": False}, index=shared_index)
    stress_index = (returns.abs().rolling(120).mean() * 100).clip(0, 10)

    bundle = assemble_bundle(
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
    return bundle, True


def build_argument_parser() -> ArgumentParser:
    default_symbol = os.getenv("HURRICANE_SYMBOL", "SPY")
    default_lookback = float(os.getenv("HURRICANE_LOOKBACK_HOURS", "6"))
    parser = ArgumentParser(description="Run the Hurricane SPY trading harness")
    parser.add_argument(
        "--symbol",
        default=default_symbol,
        help=f"Underlying symbol to trade (default: {default_symbol})",
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=default_lookback,
        help=f"Lookback window for live data pulls (default: {default_lookback} hours)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for synthetic fallbacks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip submitting orders to Alpaca while still producing trade decisions",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("HURRICANE_LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=log_level)
    credentials = AlpacaCredentials.from_env()

    lookback = timedelta(hours=args.lookback_hours)
    bundle, live = build_live_bundle(
        symbol=args.symbol,
        credentials=credentials,
        lookback=lookback,
        seed=args.seed,
    )

    config = HurricaneConfig(
        timeframes=[
            TimeframeConfig(name="1m", horizon_minutes=1, abstention_threshold=0.05, lambda_level=0.75),
            TimeframeConfig(name="5m", horizon_minutes=5, abstention_threshold=0.06, lambda_level=1.0),
            TimeframeConfig(name="30m", horizon_minutes=30, abstention_threshold=0.08, lambda_level=1.5),
        ]
    )
    pipeline = HurricaneSPY(config)

    alpaca = AlpacaClient(credentials)

    trading = TradingConfig(
        symbol=args.symbol,
        base_position_size=10,
        max_position=100,
        min_order_size=1,
        confidence_threshold=0.25,
        speed_position_scale=2.0,
        direction_score_threshold=0.05,
    )

    trader = HurricaneTrader(pipeline=pipeline, alpaca=alpaca, trading_config=trading)
    result, decision = trader.execute(bundle, dry_run=args.dry_run)

    aggregate = result["aggregate"]
    print("Aggregate forecasts:")
    for key, value in aggregate.items():
        if key == "weights":
            continue
        print(f"  {key}: {value}")

    print("\nTrade decision:")
    print(f"  action: {decision.action}")
    print(f"  target_position: {decision.target_position:.2f}")
    print(f"  order_quantity: {decision.order_quantity:.2f}")
    print(f"  confidence: {decision.confidence:.2f}")
    print(f"  reason: {decision.reason}")
    if decision.order_response:
        print("  order_response:", decision.order_response)
    source = "Unusual Whales + Alpaca live feeds" if live else "synthetic bundle"
    print(f"\nData source: {source}")
    if args.dry_run:
        print("Order submission skipped (--dry-run enabled)")


if __name__ == "__main__":
    main()
