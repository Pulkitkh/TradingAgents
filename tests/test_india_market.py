"""India market primitives: exchange identity, IST sessions, freshness, INR."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from tradingagents.dataflows.india import (
    IST,
    assess_freshness,
    benchmark_for,
    exchange_name,
    format_inr,
    format_pct,
    format_ratio,
    indian_digit_group,
    is_indian_equity,
    is_trading_hours,
    session_state,
)


@pytest.mark.unit
class TestExchangeIdentity:
    def test_nse_and_bse_suffixes_recognised(self):
        assert is_indian_equity("RELIANCE.NS")
        assert is_indian_equity("reliance.bo")
        assert exchange_name("RELIANCE.NS") == "NSE"
        assert exchange_name("500325.BO") == "BSE"

    def test_non_indian_tickers_rejected(self):
        for symbol in ("AAPL", "0700.HK", "AZN.L", "BTC-USD", "7203.T"):
            assert not is_indian_equity(symbol)
            assert exchange_name(symbol) is None

    def test_non_string_input_is_not_indian(self):
        assert not is_indian_equity(None)
        assert exchange_name(None) is None

    def test_benchmark_resolves_per_exchange(self):
        assert benchmark_for("RELIANCE.NS") == "^NSEI"
        assert benchmark_for("500325.BO") == "^BSESN"
        assert benchmark_for("AAPL") is None


@pytest.mark.unit
class TestSessionState:
    """09:15-15:30 IST on weekdays; pre-open auction from 09:00."""

    @pytest.mark.parametrize(
        ("clock", "expected"),
        [
            ((8, 30), "closed"),
            ((9, 0), "pre_open"),
            ((9, 14), "pre_open"),
            ((9, 15), "open"),
            ((12, 0), "open"),
            ((15, 30), "open"),
            ((15, 31), "closed"),
            ((18, 0), "closed"),
        ],
    )
    def test_weekday_boundaries(self, clock, expected):
        # 2026-07-27 is a Monday.
        moment = datetime(2026, 7, 27, *clock, tzinfo=IST)
        assert session_state(moment) == expected

    def test_weekend_is_never_open(self):
        saturday = datetime(2026, 7, 25, 12, 0, tzinfo=IST)
        sunday = datetime(2026, 7, 26, 12, 0, tzinfo=IST)
        assert session_state(saturday) == "weekend"
        assert session_state(sunday) == "weekend"
        assert not is_trading_hours(saturday)

    def test_naive_datetime_is_read_as_ist(self):
        assert session_state(datetime(2026, 7, 27, 10, 0)) == "open"

    def test_aware_non_ist_datetime_is_converted(self):
        from datetime import timezone

        # 05:00 UTC on a Monday is 10:30 IST — inside the session.
        utc_moment = datetime(2026, 7, 27, 5, 0, tzinfo=timezone.utc)
        assert session_state(utc_moment) == "open"


@pytest.mark.unit
class TestFreshness:
    """Staleness is measured in trading sessions, not calendar days."""

    def test_same_day_bar_is_current(self):
        sessions = [date(2026, 7, 23), date(2026, 7, 24), date(2026, 7, 27)]
        result = assess_freshness(sessions, "2026-07-27")
        assert result.is_current
        assert result.sessions_behind == 0
        assert result.latest_bar == date(2026, 7, 27)

    def test_friday_bar_on_monday_is_one_session_behind(self):
        """The exact live failure: a Monday-evening run pricing off Friday."""
        sessions = [date(2026, 7, 23), date(2026, 7, 24)]
        result = assess_freshness(sessions, "2026-07-27")
        assert not result.is_current
        assert result.sessions_behind == 1
        # Three calendar days, but only one missed session — the distinction
        # that makes this usable as a live-trading gate.
        assert result.calendar_days_behind == 3
        assert "1 trading session(s) behind" in result.as_markdown()

    def test_weekend_gap_alone_is_not_stale(self):
        """Friday's bar read on Saturday has missed no session."""
        sessions = [date(2026, 7, 24)]
        result = assess_freshness(sessions, "2026-07-25")
        assert result.sessions_behind == 0
        assert result.is_current

    def test_rows_after_analysis_date_are_ignored(self):
        sessions = [date(2026, 7, 24), date(2026, 7, 28), date(2026, 7, 29)]
        result = assess_freshness(sessions, "2026-07-24")
        assert result.latest_bar == date(2026, 7, 24)
        assert result.is_current

    def test_empty_input_reports_unknown(self):
        result = assess_freshness([], "2026-07-27")
        assert result.latest_bar is None
        assert not result.is_current
        assert "UNKNOWN" in result.as_markdown()

    def test_accepts_a_dataframe_with_a_date_column(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2026-07-23", "2026-07-24"]),
            "Close": [1280.0, 1278.0],
        })
        result = assess_freshness(frame, "2026-07-27")
        assert result.latest_bar == date(2026, 7, 24)
        assert result.sessions_behind == 1

    def test_accepts_a_datetime_index(self):
        frame = pd.DataFrame(
            {"Close": [1280.0, 1278.0]},
            index=pd.to_datetime(["2026-07-23", "2026-07-24"]),
        )
        assert assess_freshness(frame, "2026-07-24").is_current

    def test_tolerance_can_be_widened(self):
        sessions = [date(2026, 7, 24)]
        assert assess_freshness(sessions, "2026-07-27", max_sessions_behind=1).is_current


@pytest.mark.unit
class TestInrFormatting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (100, "100"),
            (1000, "1,000"),
            (100000, "1,00,000"),
            (1732156, "17,32,156"),
            (10000000, "1,00,00,000"),
        ],
    )
    def test_indian_digit_grouping(self, value, expected):
        assert indian_digit_group(value) == expected

    def test_negative_grouping_keeps_sign(self):
        assert indian_digit_group(-100000) == "-1,00,000"

    def test_market_cap_renders_in_crore(self):
        # The unreadable figure from the live report: 17.32 trillion INR.
        assert format_inr(17_321_564_307_456) == "₹17,32,156.43 Cr"
        assert format_inr(17_321_564_307_456, decimals=0) == "₹17,32,156 Cr"

    def test_auto_unit_selection(self):
        assert format_inr(500) == "₹500.00"
        assert "L" in format_inr(250_000)
        assert "Cr" in format_inr(50_000_000)

    def test_explicit_unit_overrides_auto(self):
        assert format_inr(10_000_000, unit="lakh") == "₹100.00 L"

    def test_price_levels_keep_paise(self):
        """A stop of 1238.05 quoted as "₹1,238" is a different order."""
        assert format_inr(1238.05, unit="plain") == "₹1,238.05"
        assert format_inr(1278.0, unit="plain") == "₹1,278.00"

    def test_none_and_nan_render_as_na(self):
        assert format_inr(None) == "N/A"
        assert format_inr(float("nan")) == "N/A"
        assert format_inr("not a number") == "N/A"

    def test_percentage_and_ratio_helpers(self):
        assert format_pct(0.25) == "+25.0%"
        assert format_pct(-0.073) == "-7.3%"
        assert format_pct(25.0, already_pct=True) == "+25.0%"
        assert format_pct(None) == "N/A"
        assert format_ratio(0.8234) == "0.82"
        assert format_ratio(None) == "N/A"
