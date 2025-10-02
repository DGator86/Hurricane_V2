import logging

import pytest

from hurricane_spy.execution import (
    ExecutionConfig,
    SyntheticTradingClient,
    TradingExecutor,
    get_alpaca_client,
)


def test_get_alpaca_client_dry_run():
    client = get_alpaca_client(dry_run=True)
    assert isinstance(client, SyntheticTradingClient)


def test_get_alpaca_client_without_credentials(monkeypatch, caplog):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    with caplog.at_level(logging.WARNING):
        client = get_alpaca_client()

    assert isinstance(client, SyntheticTradingClient)
    assert any("credentials missing" in record.getMessage() for record in caplog.records)


def test_trading_executor_buy_flow():
    client = SyntheticTradingClient()
    executor = TradingExecutor(client, ExecutionConfig(base_quantity=5))

    aggregate = {"probability": 0.75, "direction_score": 0.9, "hurricane_intensity": 2.0}
    decision = executor.decide(aggregate)

    assert decision.action == "buy"
    assert decision.quantity == 8  # 5 * (1 + 2 * 0.3) rounded

    order = executor.execute(decision)
    assert order is not None
    assert order["side"] == "buy"
    assert order["qty"] == 8


def test_trading_executor_hold(monkeypatch):
    client = SyntheticTradingClient()
    executor = TradingExecutor(client, ExecutionConfig(base_quantity=5))

    aggregate = {"probability": 0.5, "direction_score": 0.0, "hurricane_intensity": 0.5}
    decision = executor.decide(aggregate)

    assert decision.action == "hold"
    assert not decision.is_actionable()
    assert executor.execute(decision) is None
    assert client.orders == []
