"""Volatility-scaled risk levels, replacing invented stop-loss numbers.

The live failure: a portfolio manager set a stop at 1255.0 because it was "just
below" a verified Bollinger band at 1268.33. The band was real; the stop was
prose. These assert that every level is a computed consequence of measured ATR
and the configured risk budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows import risk_sizing
from tradingagents.dataflows.risk_sizing import (
    ATR_WINDOW,
    INR_TICK_SIZE,
    _round_to_tick,
    compute_risk_levels,
    render_risk_levels,
)


def _ohlcv(rows: int = 60, *, close: float = 1278.0, spread: float = 20.0) -> pd.DataFrame:
    """Synthetic OHLCV with a stable, known true range."""
    dates = pd.bdate_range(end="2026-07-24", periods=rows)
    closes = np.full(rows, close)
    return pd.DataFrame({
        "Date": dates,
        "Open": closes,
        "High": closes + spread / 2,
        "Low": closes - spread / 2,
        "Close": closes,
        "Volume": np.full(rows, 1_000_000),
    })


@pytest.fixture()
def stub_ohlcv(monkeypatch):
    def _install(frame):
        monkeypatch.setattr(risk_sizing, "load_ohlcv", lambda symbol, curr_date: frame)
    return _install


@pytest.mark.unit
class TestTickRounding:
    def test_rounds_to_the_nse_five_paise_tick(self):
        assert _round_to_tick(1254.97, INR_TICK_SIZE) == 1254.95
        assert _round_to_tick(1254.99, INR_TICK_SIZE) == 1255.00

    def test_zero_tick_falls_back_to_two_decimals(self):
        assert _round_to_tick(1254.9876, 0) == 1254.99


@pytest.mark.unit
class TestComputeRiskLevels:
    def test_stop_is_atr_scaled_not_a_round_number(self, stub_ohlcv):
        stub_ohlcv(_ohlcv(spread=20.0))
        levels = compute_risk_levels("RELIANCE.NS", "2026-07-24", atr_multiple=2.0)

        assert levels is not None
        assert levels.atr == pytest.approx(20.0, abs=0.5)
        # close 1278 - 2 x ATR(20) = 1238, derived rather than chosen.
        assert levels.stop_loss == pytest.approx(1238.0, abs=1.0)
        assert levels.risk_per_share == pytest.approx(40.0, abs=1.0)

    def test_target_follows_the_reward_multiple(self, stub_ohlcv):
        stub_ohlcv(_ohlcv(spread=20.0))
        levels = compute_risk_levels(
            "RELIANCE.NS", "2026-07-24", atr_multiple=2.0, reward_multiple=2.0
        )
        assert levels.reward_to_risk == pytest.approx(2.0, abs=0.05)
        assert levels.target > levels.reference_close

    def test_wider_atr_multiple_widens_the_stop(self, stub_ohlcv):
        stub_ohlcv(_ohlcv(spread=20.0))
        tight = compute_risk_levels("X.NS", "2026-07-24", atr_multiple=1.0)
        wide = compute_risk_levels("X.NS", "2026-07-24", atr_multiple=3.0)
        assert wide.stop_loss < tight.stop_loss
        assert wide.risk_per_share > tight.risk_per_share

    def test_position_size_follows_from_the_risk_budget(self, stub_ohlcv):
        stub_ohlcv(_ohlcv(spread=20.0))
        levels = compute_risk_levels(
            "RELIANCE.NS", "2026-07-24",
            atr_multiple=2.0, risk_budget_pct=1.0, account_value=1_000_000,
        )
        # 1% of ₹10,00,000 = ₹10,000 risked; at ~₹40/share that is ~250 shares.
        assert levels.position_shares == pytest.approx(250, abs=5)
        # The loss at the stop is the budget, which is the whole point.
        risked = levels.position_shares * levels.risk_per_share
        assert risked == pytest.approx(10_000, rel=0.05)

    def test_higher_volatility_yields_a_smaller_position(self, stub_ohlcv):
        stub_ohlcv(_ohlcv(spread=20.0))
        calm = compute_risk_levels("X.NS", "2026-07-24", account_value=1_000_000)
        stub_ohlcv(_ohlcv(spread=60.0))
        volatile = compute_risk_levels("X.NS", "2026-07-24", account_value=1_000_000)
        assert volatile.position_shares < calm.position_shares

    def test_no_account_value_leaves_size_unquantified(self, stub_ohlcv):
        stub_ohlcv(_ohlcv())
        levels = compute_risk_levels("X.NS", "2026-07-24")
        assert levels.position_shares is None
        assert "per ₹1,00,000 of capital" in levels.as_markdown()

    def test_indian_tickers_use_the_five_paise_tick(self, stub_ohlcv):
        stub_ohlcv(_ohlcv())
        assert compute_risk_levels("RELIANCE.NS", "2026-07-24").tick_size == INR_TICK_SIZE
        assert compute_risk_levels("AAPL", "2026-07-24").tick_size == 0.01

    def test_rows_after_the_analysis_date_are_excluded(self, stub_ohlcv):
        frame = _ohlcv(rows=60)
        future = frame.copy().tail(1)
        future["Date"] = pd.Timestamp("2026-08-15")
        future["Close"] = 2000.0
        stub_ohlcv(pd.concat([frame, future], ignore_index=True))

        levels = compute_risk_levels("X.NS", "2026-07-24")
        assert levels.reference_close == pytest.approx(1278.0)


@pytest.mark.unit
class TestUnavailableRiskLevels:
    """With no measured volatility there is no defensible stop — say so."""

    def test_insufficient_history_returns_none(self, stub_ohlcv):
        stub_ohlcv(_ohlcv(rows=ATR_WINDOW - 2))
        assert compute_risk_levels("X.NS", "2026-07-24") is None

    def test_empty_frame_returns_none(self, stub_ohlcv):
        stub_ohlcv(pd.DataFrame())
        assert compute_risk_levels("X.NS", "2026-07-24") is None

    def test_vendor_error_returns_none(self, monkeypatch):
        def explode(symbol, curr_date):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(risk_sizing, "load_ohlcv", explode)
        assert compute_risk_levels("X.NS", "2026-07-24") is None

    def test_render_forbids_inventing_levels_when_unavailable(self, stub_ohlcv):
        stub_ohlcv(pd.DataFrame())
        text = render_risk_levels("X.NS", "2026-07-24")
        assert "Unavailable" in text
        assert "Do NOT invent a stop-loss" in text


@pytest.mark.unit
class TestRenderedBlock:
    def test_block_states_the_derivation_of_every_level(self, stub_ohlcv):
        stub_ohlcv(_ohlcv())
        text = render_risk_levels("RELIANCE.NS", "2026-07-24", account_value=1_000_000)
        for label in ("Reference close", "Stop loss", "Target", "Risk per share",
                      "Reward : risk", "Position size", "Derivation"):
            assert label in text

    def test_block_forbids_rounding_to_psychological_numbers(self, stub_ohlcv):
        stub_ohlcv(_ohlcv())
        text = render_risk_levels("RELIANCE.NS", "2026-07-24")
        assert "psychological numbers" in text
        assert "Use these levels verbatim" in text
