"""Computed fundamentals, and the growth-basis conflict that motivated them.

The live failure being guarded against: a single quarter's 25% revenue growth,
lifted from a news headline, propagated through five downstream reports as
though it were the company's growth rate, while the trailing-twelve-month figure
implied roughly 7%. Both were defensible; nothing put them side by side.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows import fundamentals_metrics as fm
from tradingagents.dataflows.fundamentals_metrics import (
    FundamentalMetrics,
    Metric,
    compute_fundamental_metrics,
)


def _quarterly_frame(rows: dict[str, list[float]], periods: list[str]) -> pd.DataFrame:
    """Statement frame shaped like yfinance: rows = line items, cols = periods."""
    return pd.DataFrame(rows, index=list(rows)).iloc[:0].pipe(
        lambda _: pd.DataFrame(
            {pd.Timestamp(p): [rows[label][i] for label in rows]
             for i, p in enumerate(periods)},
            index=list(rows),
        )
    )


@pytest.mark.unit
class TestStatementRowLookup:
    def test_alias_matching_is_case_insensitive(self):
        frame = pd.DataFrame(
            {pd.Timestamp("2026-06-30"): [100.0]}, index=["total revenue"]
        )
        series = fm._row(frame, fm.REVENUE_ROWS)
        assert series is not None and series.iloc[0] == 100.0

    def test_falls_through_alias_list(self):
        frame = pd.DataFrame(
            {pd.Timestamp("2026-06-30"): [100.0]}, index=["Operating Revenue"]
        )
        assert fm._row(frame, fm.REVENUE_ROWS) is not None

    def test_missing_row_returns_none(self):
        frame = pd.DataFrame({pd.Timestamp("2026-06-30"): [1.0]}, index=["Something Else"])
        assert fm._row(frame, fm.REVENUE_ROWS) is None

    def test_empty_or_missing_frame_returns_none(self):
        assert fm._row(None, fm.REVENUE_ROWS) is None
        assert fm._row(pd.DataFrame(), fm.REVENUE_ROWS) is None


@pytest.mark.unit
class TestGrowthArithmetic:
    def test_ordered_returns_newest_first(self):
        series = pd.Series(
            [10.0, 20.0, 30.0],
            index=[pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01"),
                   pd.Timestamp("2026-07-01")],
        )
        assert fm._ordered(series) == [30.0, 20.0, 10.0]

    def test_growth_is_a_ratio(self):
        assert fm._growth(125.0, 100.0) == pytest.approx(0.25)
        assert fm._growth(93.0, 100.0) == pytest.approx(-0.07)

    def test_growth_off_a_zero_or_negative_base_is_undefined(self):
        # A percentage off a negative base reads as a turnaround or a collapse
        # depending on unseen sign conventions, so it is not reported at all.
        assert fm._growth(100.0, 0.0) is None
        assert fm._growth(100.0, -50.0) is None

    def test_growth_needs_both_operands(self):
        assert fm._growth(None, 100.0) is None
        assert fm._growth(100.0, None) is None

    def test_sum_window_requires_a_full_window(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert fm._sum_window(values, 0, 4) == 10.0
        assert fm._sum_window(values, 4, 4) is None  # only one quarter left


@pytest.mark.unit
class TestGrowthDivergenceWarning:
    """The guard that makes the 25%-vs-7% substitution impossible to miss."""

    def _metrics(self, quarterly: float | None, trailing: float | None):
        result = FundamentalMetrics(ticker="X.NS", canonical="X.NS", curr_date="2026-07-27")
        result.metrics["revenue_growth_quarter"] = Metric(quarterly, "q", "pct")
        result.metrics["revenue_growth_ttm"] = Metric(trailing, "ttm", "pct")
        fm._flag_growth_divergence(result)
        return result

    def test_the_observed_case_fires(self):
        result = self._metrics(quarterly=0.25, trailing=0.07)
        assert result.growth_divergence_warning is not None
        warning = result.growth_divergence_warning
        assert "+25.0%" in warning and "+7.0%" in warning
        assert "18.0 percentage-point" in warning
        assert "quarterly" in warning
        # It must specifically forbid the downstream misuse that occurred.
        assert "PEG" in warning

    def test_aligned_bases_do_not_warn(self):
        assert self._metrics(quarterly=0.12, trailing=0.10).growth_divergence_warning is None

    def test_threshold_boundary(self):
        below = self._metrics(quarterly=0.199, trailing=0.10)
        above = self._metrics(quarterly=0.21, trailing=0.10)
        assert below.growth_divergence_warning is None
        assert above.growth_divergence_warning is not None

    def test_warning_names_the_faster_basis(self):
        trailing_faster = self._metrics(quarterly=0.05, trailing=0.30)
        assert "trailing-twelve-month basis higher" in trailing_faster.growth_divergence_warning

    def test_missing_either_basis_cannot_warn(self):
        assert self._metrics(quarterly=None, trailing=0.07).growth_divergence_warning is None
        assert self._metrics(quarterly=0.25, trailing=None).growth_divergence_warning is None


@pytest.mark.unit
class TestDividendYield:
    """The vendor field is ambiguous, so the yield is computed where possible.

    ``dividendYield`` has shipped as both a ratio (0.0047) and a percentage
    (0.47), and the two conventions overlap in the range real equities occupy.
    A magnitude heuristic would misstate the figure by 100x half the time.
    """

    def test_computed_from_rate_and_price_when_both_present(self):
        value, basis = fm._dividend_yield(
            {"dividendRate": 5.5, "currentPrice": 1278.0, "dividendYield": 0.47}
        )
        assert value == pytest.approx(5.5 / 1278.0)
        assert "dividend rate / current price" in basis

    def test_falls_back_to_the_unambiguous_trailing_field(self):
        value, basis = fm._dividend_yield(
            {"trailingAnnualDividendYield": 0.0043, "dividendYield": 0.47}
        )
        assert value == pytest.approx(0.0043)
        assert "trailing annual" in basis

    def test_ambiguous_field_is_used_but_flagged(self):
        value, basis = fm._dividend_yield({"dividendYield": 0.47})
        assert value == pytest.approx(0.47)
        # The number is passed through unchanged; the basis warns the reader
        # rather than the code silently picking a convention.
        assert "approximate" in basis

    def test_absent_yield_reports_unavailable(self):
        value, basis = fm._dividend_yield({})
        assert value is None
        assert basis == "not available"


@pytest.mark.unit
class TestComputeFundamentalMetrics:
    """End-to-end against a stubbed yfinance handle — no network."""

    @staticmethod
    def _handle(monkeypatch, *, info, quarterly_income=None, annual_income=None,
                quarterly_balance=None, quarterly_cashflow=None):
        class FakeTicker:
            def __init__(self, symbol):
                self.symbol = symbol
                self.info = info
                self.quarterly_income_stmt = quarterly_income
                self.income_stmt = annual_income
                self.quarterly_balance_sheet = quarterly_balance
                self.quarterly_cashflow = quarterly_cashflow

        monkeypatch.setattr(fm.yf, "Ticker", FakeTicker)

    def test_growth_bases_are_computed_separately(self, monkeypatch):
        # Quarterly revenue newest-last; the latest quarter is 25% above the
        # year-ago quarter while the trailing year is only ~7% above the prior.
        periods = ["2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30",
                   "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]
        revenue = [100.0, 100.0, 100.0, 100.0, 102.0, 103.0, 104.0, 125.0]
        quarterly = pd.DataFrame(
            {pd.Timestamp(p): [revenue[i]] for i, p in enumerate(periods)},
            index=["Total Revenue"],
        )
        self._handle(monkeypatch, info={"financialCurrency": "INR"},
                     quarterly_income=quarterly)

        result = compute_fundamental_metrics("RELIANCE.NS", "2026-07-27")

        assert result.get("revenue_ttm").value == pytest.approx(434.0)
        assert result.get("revenue_growth_quarter").value == pytest.approx(0.25)
        assert result.get("revenue_growth_ttm").value == pytest.approx(0.085)
        # Both figures present and divergent -> the conflict is stated as a fact.
        assert result.growth_divergence_warning is not None

    def test_look_ahead_columns_are_excluded(self, monkeypatch):
        periods = ["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"]
        quarterly = pd.DataFrame(
            {pd.Timestamp(p): [100.0 + 10 * i] for i, p in enumerate(periods)},
            index=["Total Revenue"],
        )
        self._handle(monkeypatch, info={}, quarterly_income=quarterly)

        result = compute_fundamental_metrics("X.NS", "2026-07-27")
        # The 2026-09-30 column is after the analysis date and must not be summed.
        assert result.get("revenue_ttm").value == pytest.approx(100 + 110 + 120 + 130)

    def test_free_cash_flow_adds_negative_capex(self, monkeypatch):
        periods = ["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]
        cashflow = pd.DataFrame(
            {pd.Timestamp(p): [500.0, -200.0] for p in periods},
            index=["Operating Cash Flow", "Capital Expenditure"],
        )
        self._handle(monkeypatch, info={}, quarterly_cashflow=cashflow)

        result = compute_fundamental_metrics("X.NS", "2026-07-27")
        assert result.get("operating_cash_flow_ttm").value == pytest.approx(2000.0)
        assert result.get("capex_ttm").value == pytest.approx(-800.0)
        assert result.get("free_cash_flow_ttm").value == pytest.approx(1200.0)

    def test_leverage_and_liquidity_from_balance_sheet(self, monkeypatch):
        balance = pd.DataFrame(
            {pd.Timestamp("2026-06-30"): [4000.0, 8000.0, 5900.0, 4000.0]},
            index=["Total Debt", "Stockholders Equity", "Current Assets",
                   "Current Liabilities"],
        )
        self._handle(monkeypatch, info={}, quarterly_balance=balance)

        result = compute_fundamental_metrics("X.NS", "2026-07-27")
        assert result.get("debt_to_equity").value == pytest.approx(0.5)
        assert result.get("current_ratio").value == pytest.approx(1.475)

    def test_vendor_failure_degrades_instead_of_raising(self, monkeypatch):
        class ExplodingTicker:
            def __init__(self, symbol):
                raise RuntimeError("vendor down")

        monkeypatch.setattr(fm.yf, "Ticker", ExplodingTicker)
        result = compute_fundamental_metrics("X.NS", "2026-07-27")
        assert result.errors
        assert result.get("revenue_ttm").value is None

    def test_markdown_renders_na_for_missing_metrics(self, monkeypatch):
        self._handle(monkeypatch, info={"financialCurrency": "INR"})
        markdown = compute_fundamental_metrics("X.NS", "2026-07-27").as_markdown()
        assert "Revenue growth — latest quarter YoY" in markdown
        assert "Revenue growth — TTM YoY" in markdown
        assert "N/A" in markdown
        # Every row must carry the basis it was computed on.
        assert "Basis" in markdown
