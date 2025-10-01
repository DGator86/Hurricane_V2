"""Top-level package for the Hurricane SPY automated prediction system."""

from .config import HurricaneConfig, TimeframeConfig
from .execution import (
    AlpacaClient,
    AlpacaCredentials,
    HurricaneTrader,
    TradeDecision,
    TradingConfig,
)
from .data_sources import (
    UnusualWhalesClient,
    assemble_bundle,
    merge_unusual_whales_signals,
)
from .pipeline import HurricaneSPY

__all__ = [
    "HurricaneConfig",
    "TimeframeConfig",
    "HurricaneSPY",
    "AlpacaCredentials",
    "AlpacaClient",
    "TradingConfig",
    "TradeDecision",
    "HurricaneTrader",
    "UnusualWhalesClient",
    "merge_unusual_whales_signals",
    "assemble_bundle",
]
