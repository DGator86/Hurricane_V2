"""Alpaca execution utilities for Hurricane SPY."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


class SyntheticTradingClient:
    """Fallback trading client that simulates orders locally."""

    def __init__(self) -> None:
        self.orders: list[Dict[str, Any]] = []

    def submit_order(self, symbol: str, qty: int, side: str, type: str = "market", **kwargs: Any) -> Dict[str, Any]:
        order = {
            "id": f"simulated-{len(self.orders) + 1}",
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "status": "filled",
            "kwargs": kwargs,
        }
        self.orders.append(order)
        logger.info("Simulated %s order for %s shares of %s", side, qty, symbol)
        return order

    def list_positions(self) -> list[Dict[str, Any]]:
        return []


def _resolve_base_url(mode: Optional[str]) -> str:
    if not mode:
        mode = os.getenv("ALPACA_MODE", "paper")
    mode = mode.lower()
    if mode == "paper":
        return "https://paper-api.alpaca.markets"
    if mode == "live":
        return "https://api.alpaca.markets"
    raise ValueError(f"Unsupported Alpaca mode '{mode}'")


def get_alpaca_client(*, dry_run: bool = False, mode: Optional[str] = None) -> Any:
    """Return an Alpaca REST client or a synthetic fallback."""

    if dry_run:
        logger.info("Dry-run requested, using synthetic trading client")
        return SyntheticTradingClient()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        logger.warning("Alpaca credentials missing, switching to synthetic client")
        return SyntheticTradingClient()

    base_url = _resolve_base_url(mode)

    try:
        from alpaca_trade_api import REST  # type: ignore
    except ImportError as exc:  # pragma: no cover - triggered only without dependency
        raise RuntimeError(
            "alpaca-trade-api is required for live trading. Install it via 'pip install alpaca-trade-api'."
        ) from exc

    logger.info("Connecting to Alpaca at %s (%s mode)", base_url, os.getenv("ALPACA_MODE", mode or "paper"))
    return REST(api_key, secret_key, base_url)


@dataclass
class ExecutionConfig:
    """Configuration parameters for turning forecasts into orders."""

    symbol: str = "SPY"
    base_quantity: int = 10
    max_position: int = 200
    buy_threshold: float = 0.6
    sell_threshold: float = 0.4
    intensity_multiplier: float = 0.3


@dataclass
class TradingDecision:
    """Represents the action derived from a forecast."""

    action: str
    quantity: int
    probability: float
    intensity: float
    reason: str

    def is_actionable(self) -> bool:
        return self.action in {"buy", "sell"} and self.quantity > 0


class TradingExecutor:
    """Map Hurricane SPY outputs to Alpaca orders."""

    def __init__(self, client: Any, config: ExecutionConfig) -> None:
        self.client = client
        self.config = config

    def decide(self, aggregate: Mapping[str, Any]) -> TradingDecision:
        probability = float(aggregate.get("probability", 0.5))
        direction_score = float(aggregate.get("direction_score", 0.0))
        intensity = float(aggregate.get("hurricane_intensity", 0.0))

        if probability >= self.config.buy_threshold and direction_score > 0:
            qty = self._scaled_quantity(intensity)
            reason = f"probability {probability:.2f} exceeds buy threshold"
            return TradingDecision("buy", qty, probability, intensity, reason)

        if probability <= self.config.sell_threshold and direction_score < 0:
            qty = self._scaled_quantity(intensity)
            reason = f"probability {probability:.2f} below sell threshold"
            return TradingDecision("sell", qty, probability, intensity, reason)

        return TradingDecision("hold", 0, probability, intensity, "No actionable signal")

    def execute(self, decision: TradingDecision) -> Optional[Dict[str, Any]]:
        if not decision.is_actionable():
            logger.info(
                "Holding position (probability=%.2f, intensity=%.2f): %s",
                decision.probability,
                decision.intensity,
                decision.reason,
            )
            return None

        side = decision.action
        qty = min(decision.quantity, self.config.max_position)

        try:
            logger.info("Submitting %s order for %s shares of %s", side, qty, self.config.symbol)
            order = self.client.submit_order(
                symbol=self.config.symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
        except Exception as exc:  # pragma: no cover - network failure path
            logger.exception("Order submission failed: %s", exc)
            raise

        logger.info("Order response: %s", order)
        return order

    def _scaled_quantity(self, intensity: float) -> int:
        scaled = self.config.base_quantity * (1 + max(intensity, 0.0) * self.config.intensity_multiplier)
        return max(1, int(round(scaled)))

