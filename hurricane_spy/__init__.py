"""Top-level package for the Hurricane SPY automated prediction system."""

from .config import HurricaneConfig, TimeframeConfig
from .execution import ExecutionConfig, TradingDecision, TradingExecutor, get_alpaca_client
from .data_sources import UnusualWhalesClient, load_market_data_from_unusual_whales
from .pipeline import HurricaneSPY

__all__ = [
    "ExecutionConfig",
    "HurricaneConfig",
    "HurricaneSPY",
    "TimeframeConfig",
    "TradingDecision",
    "TradingExecutor",
    "get_alpaca_client",
    "UnusualWhalesClient",
    "load_market_data_from_unusual_whales",
]
