"""Pytest fixtures for Hurricane SPY tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict

import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hurricane_spy import HurricaneConfig, HurricaneSPY, TimeframeConfig
from hurricane_spy.data_structures import MarketDataBundle
from hurricane_spy.scripts.run_pipeline import generate_dummy_data


@pytest.fixture(scope="session")
def timeframe_names() -> Dict[str, TimeframeConfig]:
    """Return the canonical timeframe configuration used across tests."""

    config = {
        "1m": TimeframeConfig(
            name="1m",
            horizon_minutes=1,
            abstention_threshold=0.05,
            lambda_level=0.75,
        ),
        "5m": TimeframeConfig(
            name="5m",
            horizon_minutes=5,
            abstention_threshold=0.06,
            lambda_level=1.0,
        ),
        "30m": TimeframeConfig(
            name="30m",
            horizon_minutes=30,
            abstention_threshold=0.08,
            lambda_level=1.5,
        ),
    }
    return config


@pytest.fixture(scope="session")
def hurricane_config(timeframe_names: Dict[str, TimeframeConfig]) -> HurricaneConfig:
    """Provide a Hurricane configuration shared across tests."""

    return HurricaneConfig(timeframes=list(timeframe_names.values()))


@pytest.fixture
def dummy_index() -> pd.DatetimeIndex:
    """Deterministic timestamp index for synthetic bundles."""

    return pd.date_range("2025-01-01", periods=512, freq="min", tz="UTC")


@pytest.fixture
def dummy_bundle(dummy_index: pd.DatetimeIndex) -> MarketDataBundle:
    """Generate reproducible synthetic data for pipeline runs."""

    np.random.seed(42)
    return generate_dummy_data(dummy_index)


@pytest.fixture
def pipeline(hurricane_config: HurricaneConfig) -> HurricaneSPY:
    """Instantiate the Hurricane pipeline under test."""

    return HurricaneSPY(hurricane_config)


@pytest.fixture
def bundle_without_close(dummy_bundle: MarketDataBundle) -> MarketDataBundle:
    """Bundle missing the closing price column for validation checks."""

    price = dummy_bundle.price.drop(columns=["close"])
    return replace(dummy_bundle, price=price)


@pytest.fixture
def bundle_with_negative_vol(dummy_bundle: MarketDataBundle) -> MarketDataBundle:
    """Bundle containing a negative realised volatility entry."""

    realised_vol = {
        name: series.copy()
        for name, series in dummy_bundle.realised_vol.items()
    }
    first_tf = next(iter(realised_vol))
    realised_vol[first_tf].iloc[-1] = -0.1
    return replace(dummy_bundle, realised_vol=realised_vol)
