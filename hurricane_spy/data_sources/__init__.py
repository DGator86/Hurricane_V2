"""External data source utilities for Hurricane SPY."""

from .unusual_whales import UnusualWhalesClient, load_market_data_from_unusual_whales

__all__ = ["UnusualWhalesClient", "load_market_data_from_unusual_whales"]
