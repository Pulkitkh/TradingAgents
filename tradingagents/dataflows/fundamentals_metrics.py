"""Deterministic fundamental metrics computed in code, never by an LLM.

Motivation, from an observed live failure on ``RELIANCE.NS``: a news headline
reported "25% revenue growth" for a single quarter. That figure was picked up by
the news analyst, echoed by the sentiment analyst as its only bullish evidence,
carried into the research plan, the trader proposal, and finally the portfolio
manager's memo — where it became the load-bearing justification for an
Overweight rating. Meanwhile the fundamentals report in the same run showed a
trailing-twelve-month revenue base implying roughly 7% growth. Both numbers were
defensible in isolation; nothing in the pipeline ever put them side by side.

The fix is not a better prompt. It is to compute every growth rate here, label
each one with the exact basis it was computed on, and hand the agents a single
block where a quarterly spike and a trailing trend cannot be confused for each
other. :attr:`FundamentalMetrics.growth_divergence_warning` fires automatically
when the bases disagree materially, so the discrepancy is stated as a fact
rather than left for an agent to notice.

Every value here carries a ``basis`` string naming the statement and periods it
came from. Anything that cannot be computed is ``None`` and renders as ``N/A`` —
never a guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yfinance as yf

from .india import format_inr, format_pct, format_ratio
from .stockstats_utils import filter_financials_by_date, yf_retry
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

# Statement row labels vary across yfinance versions and between issuers, so
# every lookup goes through an alias list rather than a single hardcoded key.
REVENUE_ROWS = ("Total Revenue", "Operating Revenue", "Revenue")
NET_INCOME_ROWS = (
    "Net Income",
    "Net Income Common Stockholders",
    "Net Income From Continuing Operation Net Minority Interest",
)
OPERATING_INCOME_ROWS = ("Operating Income", "Total Operating Income As Reported")
TOTAL_DEBT_ROWS = ("Total Debt",)
EQUITY_ROWS = ("Stockholders Equity", "Total Equity Gross Minority Interest")
CURRENT_ASSETS_ROWS = ("Current Assets", "Total Current Assets")
CURRENT_LIABILITIES_ROWS = ("Current Liabilities", "Total Current Liabilities")
OPERATING_CASHFLOW_ROWS = ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
CAPEX_ROWS = ("Capital Expenditure", "Purchase Of PPE")

# Growth measured on a single quarter versus a trailing year can legitimately
# differ. Beyond this spread (in percentage points) the difference is material
# enough that presenting either number alone would misrepresent the business.
GROWTH_DIVERGENCE_THRESHOLD_PP = 10.0


@dataclass(frozen=True)
class Metric:
    """One computed number plus the exact basis it was computed on."""

    value: float | None
    basis: str
    unit: str = "ratio"   # "ratio" | "pct" | "currency" | "multiple"

    def rendered(self) -> str:
        if self.value is None:
            return "N/A"
        if self.unit == "currency":
            return format_inr(self.value)
        if self.unit == "pct":
            return format_pct(self.value)
        return format_ratio(self.value)


@dataclass
class FundamentalMetrics:
    """Computed fundamentals for one instrument as of an analysis date."""

    ticker: str
    canonical: str
    curr_date: str
    currency: str | None = None
    metrics: dict[str, Metric] = field(default_factory=dict)
    growth_divergence_warning: str | None = None
    errors: list[str] = field(default_factory=list)

    def get(self, key: str) -> Metric:
        return self.metrics.get(key, Metric(None, "not available"))

    def as_markdown(self) -> str:
        """Render the verified fundamentals block for prompt injection."""
        lines = [
            f"### Verified fundamentals — {self.canonical} (computed, not model-generated)",
            "",
            f"Figures below are computed directly from filed statements as of "
            f"{self.curr_date}. Statement columns dated after the analysis date are "
            f"excluded. Currency: {self.currency or 'unknown'}.",
            "",
            "| Metric | Value | Basis |",
            "|---|---:|---|",
        ]
        for label, key in _DISPLAY_ORDER:
            metric = self.get(key)
            lines.append(f"| {label} | {metric.rendered()} | {metric.basis} |")

        if self.growth_divergence_warning:
            lines += ["", f"> **GROWTH BASIS CONFLICT.** {self.growth_divergence_warning}"]

        if self.errors:
            lines += ["", "Not computable this run: " + "; ".join(self.errors) + "."]
        return "\n".join(lines)


_DISPLAY_ORDER: tuple[tuple[str, str], ...] = (
    ("Revenue (TTM)", "revenue_ttm"),
    ("Revenue growth — TTM YoY", "revenue_growth_ttm"),
    ("Revenue growth — latest quarter YoY", "revenue_growth_quarter"),
    ("Revenue growth — last full year YoY", "revenue_growth_annual"),
    ("Net income (TTM)", "net_income_ttm"),
    ("Net income growth — TTM YoY", "net_income_growth_ttm"),
    ("Net margin (TTM)", "net_margin_ttm"),
    ("Operating margin (TTM)", "operating_margin_ttm"),
    ("Operating cash flow (TTM)", "operating_cash_flow_ttm"),
    ("Capital expenditure (TTM)", "capex_ttm"),
    ("Free cash flow (TTM)", "free_cash_flow_ttm"),
    ("Total debt", "total_debt"),
    ("Shareholders equity", "total_equity"),
    ("Debt / equity", "debt_to_equity"),
    ("Current ratio", "current_ratio"),
    ("Return on equity (TTM)", "return_on_equity"),
    ("Market capitalisation", "market_cap"),
    ("P/E (trailing)", "pe_trailing"),
    ("P/E (forward)", "pe_forward"),
    ("P/B", "price_to_book"),
    ("PEG", "peg"),
    ("EV / EBITDA", "ev_to_ebitda"),
    ("Dividend yield", "dividend_yield"),
    ("Beta", "beta"),
)


def _row(frame: pd.DataFrame | None, aliases: tuple[str, ...]) -> pd.Series | None:
    """First matching statement row, tolerant of label drift and casing."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    lookup = {str(idx).strip().lower(): idx for idx in frame.index}
    for alias in aliases:
        hit = lookup.get(alias.strip().lower())
        if hit is not None:
            series = frame.loc[hit]
            if isinstance(series, pd.DataFrame):    # duplicate labels
                series = series.iloc[0]
            return series.dropna()
    return None


def _ordered(series: pd.Series | None) -> list[float]:
    """Statement values newest-first as plain floats."""
    if series is None or series.empty:
        return []
    try:
        ordered = series.sort_index(ascending=False)
    except TypeError:
        ordered = series
    values = []
    for raw in ordered.tolist():
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if not pd.isna(number):
            values.append(number)
    return values


def _sum_window(values: list[float], start: int, count: int) -> float | None:
    window = values[start:start + count]
    return sum(window) if len(window) == count else None


def _growth(current: float | None, prior: float | None) -> float | None:
    """Period-over-period growth as a ratio; ``None`` when the base is unusable."""
    if current is None or prior is None:
        return None
    if prior == 0:
        return None
    if prior < 0:
        # Growth off a negative base is not meaningful as a percentage; report
        # nothing rather than a number that reads as a turnaround or a collapse
        # depending on sign conventions the reader can't see.
        return None
    return (current - prior) / abs(prior)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _info_float(info: dict[str, Any], key: str) -> float | None:
    value = info.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def compute_fundamental_metrics(ticker: str, curr_date: str) -> FundamentalMetrics:
    """Compute the verified fundamentals block for ``ticker`` as of ``curr_date``.

    Fails soft: any statement that cannot be fetched or parsed leaves its
    metrics as ``None`` and appends a line to ``errors``, so a partial data
    vendor degrades the fact sheet instead of aborting the run.
    """
    canonical = normalize_symbol(ticker)
    result = FundamentalMetrics(ticker=ticker, canonical=canonical, curr_date=curr_date)

    try:
        handle = yf.Ticker(canonical)
        info = yf_retry(lambda: handle.info) or {}
    except Exception as exc:  # noqa: BLE001 — never block a run on fundamentals
        logger.warning("Fundamentals unavailable for %s: %s", ticker, exc)
        result.errors.append(f"company profile ({type(exc).__name__})")
        return result

    result.currency = info.get("financialCurrency") or info.get("currency")

    quarterly_income = _statement(handle, "quarterly_income_stmt", curr_date, result)
    annual_income = _statement(handle, "income_stmt", curr_date, result)
    quarterly_balance = _statement(handle, "quarterly_balance_sheet", curr_date, result)
    quarterly_cashflow = _statement(handle, "quarterly_cashflow", curr_date, result)

    _add_revenue_metrics(result, quarterly_income, annual_income)
    _add_profitability_metrics(result, quarterly_income)
    _add_cashflow_metrics(result, quarterly_cashflow)
    _add_balance_metrics(result, quarterly_balance)
    _add_valuation_metrics(result, info)
    _flag_growth_divergence(result)
    return result


def _statement(
    handle: Any, attribute: str, curr_date: str, result: FundamentalMetrics
) -> pd.DataFrame | None:
    """Fetch one statement, look-ahead filtered to ``curr_date``."""
    try:
        frame = yf_retry(lambda: getattr(handle, attribute))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Statement %s unavailable for %s: %s", attribute, result.ticker, exc)
        result.errors.append(f"{attribute.replace('_', ' ')} ({type(exc).__name__})")
        return None
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        result.errors.append(f"{attribute.replace('_', ' ')} (empty)")
        return None
    return filter_financials_by_date(frame, curr_date)


def _add_revenue_metrics(
    result: FundamentalMetrics,
    quarterly_income: pd.DataFrame | None,
    annual_income: pd.DataFrame | None,
) -> None:
    quarters = _ordered(_row(quarterly_income, REVENUE_ROWS))
    annuals = _ordered(_row(annual_income, REVENUE_ROWS))

    ttm = _sum_window(quarters, 0, 4)
    prior_ttm = _sum_window(quarters, 4, 4)
    result.metrics["revenue_ttm"] = Metric(
        ttm, "sum of the 4 most recent reported quarters", "currency"
    )
    result.metrics["revenue_growth_ttm"] = Metric(
        _growth(ttm, prior_ttm),
        "trailing 4 quarters vs the 4 quarters before them",
        "pct",
    )
    result.metrics["revenue_growth_quarter"] = Metric(
        _growth(
            quarters[0] if quarters else None,
            quarters[4] if len(quarters) > 4 else None,
        ),
        "most recent quarter vs the same quarter one year earlier",
        "pct",
    )
    result.metrics["revenue_growth_annual"] = Metric(
        _growth(
            annuals[0] if annuals else None,
            annuals[1] if len(annuals) > 1 else None,
        ),
        "last full fiscal year vs the prior fiscal year",
        "pct",
    )


def _add_profitability_metrics(
    result: FundamentalMetrics, quarterly_income: pd.DataFrame | None
) -> None:
    net_quarters = _ordered(_row(quarterly_income, NET_INCOME_ROWS))
    op_quarters = _ordered(_row(quarterly_income, OPERATING_INCOME_ROWS))
    revenue_ttm = result.get("revenue_ttm").value

    net_ttm = _sum_window(net_quarters, 0, 4)
    prior_net_ttm = _sum_window(net_quarters, 4, 4)
    op_ttm = _sum_window(op_quarters, 0, 4)

    result.metrics["net_income_ttm"] = Metric(
        net_ttm, "sum of the 4 most recent reported quarters", "currency"
    )
    result.metrics["net_income_growth_ttm"] = Metric(
        _growth(net_ttm, prior_net_ttm),
        "trailing 4 quarters vs the 4 quarters before them",
        "pct",
    )
    result.metrics["net_margin_ttm"] = Metric(
        _safe_ratio(net_ttm, revenue_ttm), "TTM net income / TTM revenue", "pct"
    )
    result.metrics["operating_margin_ttm"] = Metric(
        _safe_ratio(op_ttm, revenue_ttm), "TTM operating income / TTM revenue", "pct"
    )


def _add_cashflow_metrics(
    result: FundamentalMetrics, quarterly_cashflow: pd.DataFrame | None
) -> None:
    ocf = _sum_window(_ordered(_row(quarterly_cashflow, OPERATING_CASHFLOW_ROWS)), 0, 4)
    capex = _sum_window(_ordered(_row(quarterly_cashflow, CAPEX_ROWS)), 0, 4)

    result.metrics["operating_cash_flow_ttm"] = Metric(
        ocf, "sum of the 4 most recent reported quarters", "currency"
    )
    result.metrics["capex_ttm"] = Metric(
        capex, "sum of the 4 most recent reported quarters", "currency"
    )
    # yfinance reports capital expenditure as a negative outflow, so free cash
    # flow is a sum rather than a difference.
    result.metrics["free_cash_flow_ttm"] = Metric(
        None if ocf is None or capex is None else ocf + capex,
        "TTM operating cash flow plus TTM capital expenditure (capex is negative)",
        "currency",
    )


def _add_balance_metrics(
    result: FundamentalMetrics, quarterly_balance: pd.DataFrame | None
) -> None:
    debt = _ordered(_row(quarterly_balance, TOTAL_DEBT_ROWS))
    equity = _ordered(_row(quarterly_balance, EQUITY_ROWS))
    current_assets = _ordered(_row(quarterly_balance, CURRENT_ASSETS_ROWS))
    current_liabilities = _ordered(_row(quarterly_balance, CURRENT_LIABILITIES_ROWS))

    latest_debt = debt[0] if debt else None
    latest_equity = equity[0] if equity else None

    result.metrics["total_debt"] = Metric(latest_debt, "most recent quarterly balance sheet", "currency")
    result.metrics["total_equity"] = Metric(latest_equity, "most recent quarterly balance sheet", "currency")
    result.metrics["debt_to_equity"] = Metric(
        _safe_ratio(latest_debt, latest_equity),
        "total debt / shareholders equity, most recent quarter",
        "multiple",
    )
    result.metrics["current_ratio"] = Metric(
        _safe_ratio(
            current_assets[0] if current_assets else None,
            current_liabilities[0] if current_liabilities else None,
        ),
        "current assets / current liabilities, most recent quarter",
        "multiple",
    )
    result.metrics["return_on_equity"] = Metric(
        _safe_ratio(result.get("net_income_ttm").value, latest_equity),
        "TTM net income / most recent shareholders equity",
        "pct",
    )


def _add_valuation_metrics(result: FundamentalMetrics, info: dict[str, Any]) -> None:
    result.metrics["market_cap"] = Metric(
        _info_float(info, "marketCap"), "vendor quote snapshot", "currency"
    )
    result.metrics["pe_trailing"] = Metric(
        _info_float(info, "trailingPE"), "vendor trailing P/E", "multiple"
    )
    result.metrics["pe_forward"] = Metric(
        _info_float(info, "forwardPE"), "vendor forward P/E on consensus estimates", "multiple"
    )
    result.metrics["price_to_book"] = Metric(
        _info_float(info, "priceToBook"), "vendor price / book", "multiple"
    )
    result.metrics["peg"] = Metric(
        _info_float(info, "pegRatio") or _info_float(info, "trailingPegRatio"),
        "vendor PEG — depends on the vendor's growth estimate, not computed here",
        "multiple",
    )
    result.metrics["ev_to_ebitda"] = Metric(
        _info_float(info, "enterpriseToEbitda"), "vendor enterprise value / EBITDA", "multiple"
    )
    yield_value, yield_basis = _dividend_yield(info)
    result.metrics["dividend_yield"] = Metric(yield_value, yield_basis, "pct")
    result.metrics["beta"] = Metric(_info_float(info, "beta"), "vendor 5-year monthly beta", "multiple")


def _dividend_yield(info: dict[str, Any]) -> tuple[float | None, str]:
    """Dividend yield as a ratio, computed rather than guessed.

    yfinance has shipped ``dividendYield`` as both a ratio (0.0047) and a
    percentage (0.47) across versions, and the two conventions overlap in the
    range real equities occupy — 0.47 is a plausible 0.47% *and* a plausible
    47%. A magnitude heuristic therefore cannot distinguish them, and guessing
    wrong misstates the figure by 100x in a report someone may trade on.

    So the ambiguous field is used only as a last resort. Preference order:

    1. ``dividendRate`` / ``currentPrice`` — both unambiguously absolute, so the
       ratio is exact.
    2. ``trailingAnnualDividendYield`` — consistently a ratio in every version.
    3. ``dividendYield`` with its convention flagged in the basis string, so a
       reader can see the number is not fully trusted.
    """
    rate = _info_float(info, "dividendRate")
    price = _info_float(info, "currentPrice") or _info_float(info, "regularMarketPrice")
    if rate is not None and price:
        return rate / price, "annual dividend rate / current price"

    trailing = _info_float(info, "trailingAnnualDividendYield")
    if trailing is not None:
        return trailing, "vendor trailing annual dividend yield"

    raw = _info_float(info, "dividendYield")
    if raw is None:
        return None, "not available"
    return raw, "vendor dividend yield — units follow the vendor's convention, treat as approximate"


def _flag_growth_divergence(result: FundamentalMetrics) -> None:
    """Raise a warning when quarterly and trailing growth tell different stories."""
    quarterly = result.get("revenue_growth_quarter").value
    trailing = result.get("revenue_growth_ttm").value
    if quarterly is None or trailing is None:
        return

    spread_pp = abs(quarterly - trailing) * 100
    if spread_pp < GROWTH_DIVERGENCE_THRESHOLD_PP:
        return

    faster = "quarterly" if quarterly > trailing else "trailing-twelve-month"
    result.growth_divergence_warning = (
        f"Latest-quarter revenue growth is {format_pct(quarterly)} while "
        f"trailing-twelve-month growth is {format_pct(trailing)} — a "
        f"{spread_pp:.1f} percentage-point spread, with the {faster} basis higher. "
        "These measure different things. Do NOT describe either figure as "
        "'the company's revenue growth' without naming its basis, and do not "
        "carry a single-quarter figure into a valuation argument (PEG, "
        "growth-adjusted multiples) that assumes a sustained rate."
    )
