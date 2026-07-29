"""Deterministic, volatility-scaled risk levels computed in code.

Observed failure this replaces: a portfolio manager memo that set a stop at
"1255.0" because it was "just below" a verified Bollinger band at 1268.33. The
band was real; the stop was invented. The memo read like risk management —
tiered entries, a "firm, non-negotiable stop", a claim that downside was capped
at "a fraction of a percent of portfolio capital" — while no risk calculation
had occurred anywhere in the system, and the ATR needed to do one was sitting
unused in the verified snapshot.

Levels here are derived from Average True Range, so stop distance scales with
the instrument's own volatility rather than a round number that reads well.
Position size follows from a fixed fractional risk budget: the rupee loss at the
stop is the input, and share count is the output. The agent is handed finished
numbers and told not to modify them.

NSE/BSE market structure is applied where it changes the answer: prices are
rounded to the ₹0.05 tick, and share counts are whole numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from stockstats import wrap

from .india import format_inr, format_ratio, is_indian_equity
from .stockstats_utils import load_ohlcv
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

# Default stop distance in ATR multiples. 2.0 is wide enough that ordinary daily
# noise does not trigger it, which is the usual failure of a tighter stop.
DEFAULT_ATR_MULTIPLE = 2.0
# Reward target as a multiple of the risked distance.
DEFAULT_REWARD_MULTIPLE = 2.0
# Fraction of capital risked on a single position, in percent.
DEFAULT_RISK_BUDGET_PCT = 1.0
# NSE/BSE tick size for most cash-segment equities.
INR_TICK_SIZE = 0.05


@dataclass(frozen=True)
class RiskLevels:
    """Volatility-scaled entry, stop, target and size for one instrument."""

    symbol: str
    curr_date: str
    reference_close: float
    atr: float
    atr_pct_of_price: float
    atr_multiple: float
    stop_loss: float
    target: float
    risk_per_share: float
    reward_per_share: float
    reward_to_risk: float
    risk_budget_pct: float
    account_value: float | None
    position_shares: int | None
    position_value: float | None
    position_pct_of_capital: float | None
    tick_size: float

    def as_markdown(self) -> str:
        lines = [
            f"### Computed risk levels — {self.symbol} (deterministic, not model-generated)",
            "",
            f"Derived from ATR({ATR_WINDOW}) on data through {self.curr_date}. "
            f"Long-side framing; invert for a short.",
            "",
            "| Level | Value | Derivation |",
            "|---|---:|---|",
            f"| Reference close | {format_inr(self.reference_close)} | latest verified close |",
            f"| ATR({ATR_WINDOW}) | {format_inr(self.atr)} "
            f"({self.atr_pct_of_price:.2f}% of price) | average true range |",
            f"| Stop loss | {format_inr(self.stop_loss)} | close − "
            f"{format_ratio(self.atr_multiple, digits=1)} × ATR |",
            f"| Target | {format_inr(self.target)} | close + "
            f"{format_ratio(self.atr_multiple * self.reward_to_risk, digits=1)} × ATR |",
            f"| Risk per share | {format_inr(self.risk_per_share)} | close − stop |",
            f"| Reward : risk | {format_ratio(self.reward_to_risk, digits=2)} : 1 | target vs stop |",
        ]
        if self.position_shares is not None:
            lines += [
                f"| Position size | {self.position_shares:,} shares | "
                f"{self.risk_budget_pct:.2f}% of {format_inr(self.account_value)} "
                f"÷ risk per share |",
                f"| Position value | {format_inr(self.position_value)} | "
                f"{self.position_pct_of_capital:.2f}% of capital |",
            ]
        else:
            per_lakh = int(
                (100_000 * self.risk_budget_pct / 100) / self.risk_per_share
            ) if self.risk_per_share > 0 else 0
            lines.append(
                f"| Position size | {per_lakh:,} shares per ₹1,00,000 of capital | "
                f"at a {self.risk_budget_pct:.2f}% risk budget; no account value supplied |"
            )

        lines += [
            "",
            f"Prices are rounded to the ₹{self.tick_size:.2f} exchange tick. "
            "Use these levels verbatim. Do not round them to psychological "
            "numbers, widen the stop to avoid being taken out, or state a stop "
            "or target that does not appear in this table — every level here is "
            "a computed consequence of measured volatility and the stated risk "
            "budget.",
        ]
        return "\n".join(lines)


ATR_WINDOW = 14


def _round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return round(price, 2)
    return round(round(price / tick) * tick, 2)


def compute_risk_levels(
    symbol: str,
    curr_date: str,
    *,
    atr_multiple: float = DEFAULT_ATR_MULTIPLE,
    reward_multiple: float = DEFAULT_REWARD_MULTIPLE,
    risk_budget_pct: float = DEFAULT_RISK_BUDGET_PCT,
    account_value: float | None = None,
) -> RiskLevels | None:
    """Compute volatility-scaled levels, or ``None`` when ATR is unavailable.

    Returning ``None`` is deliberate: with no measured volatility there is no
    defensible stop, and the caller must say so rather than fall back to a
    percentage guess.
    """
    canonical = normalize_symbol(symbol)
    try:
        data = load_ohlcv(canonical, curr_date)
    except Exception as exc:  # noqa: BLE001 — degrade, never abort the run
        logger.warning("Risk sizing unavailable for %s: %s", symbol, exc)
        return None

    if data is None or data.empty or len(data) < ATR_WINDOW + 1:
        logger.warning("Risk sizing needs >%d rows for %s", ATR_WINDOW, symbol)
        return None

    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date")
    frame = frame[frame["Date"] <= pd.to_datetime(curr_date)]
    if len(frame) < ATR_WINDOW + 1:
        return None

    try:
        stock = wrap(frame.copy())
        stock["atr"]
        atr = float(stock.iloc[-1]["atr"])
        close = float(frame.iloc[-1]["Close"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("ATR computation failed for %s: %s", symbol, exc)
        return None

    if not atr or pd.isna(atr) or atr <= 0 or not close or close <= 0:
        return None

    tick = INR_TICK_SIZE if is_indian_equity(canonical) else 0.01
    raw_stop = close - atr_multiple * atr
    if raw_stop <= 0:
        return None

    stop = _round_to_tick(raw_stop, tick)
    risk_per_share = round(close - stop, 2)
    if risk_per_share <= 0:
        return None

    target = _round_to_tick(close + reward_multiple * atr_multiple * atr, tick)
    reward_per_share = round(target - close, 2)

    shares = position_value = position_pct = None
    if account_value and account_value > 0:
        shares = int((account_value * risk_budget_pct / 100) // risk_per_share)
        position_value = round(shares * close, 2)
        position_pct = round(position_value / account_value * 100, 2) if shares else 0.0

    return RiskLevels(
        symbol=canonical,
        curr_date=curr_date,
        reference_close=close,
        atr=round(atr, 2),
        atr_pct_of_price=round(atr / close * 100, 2),
        atr_multiple=atr_multiple,
        stop_loss=stop,
        target=target,
        risk_per_share=risk_per_share,
        reward_per_share=reward_per_share,
        reward_to_risk=round(reward_per_share / risk_per_share, 2) if risk_per_share else 0.0,
        risk_budget_pct=risk_budget_pct,
        account_value=account_value,
        position_shares=shares,
        position_value=position_value,
        position_pct_of_capital=position_pct,
        tick_size=tick,
    )


def render_risk_levels(
    symbol: str,
    curr_date: str,
    *,
    atr_multiple: float = DEFAULT_ATR_MULTIPLE,
    reward_multiple: float = DEFAULT_REWARD_MULTIPLE,
    risk_budget_pct: float = DEFAULT_RISK_BUDGET_PCT,
    account_value: float | None = None,
) -> str:
    """Markdown risk block, or an explicit unavailable notice."""
    levels = compute_risk_levels(
        symbol,
        curr_date,
        atr_multiple=atr_multiple,
        reward_multiple=reward_multiple,
        risk_budget_pct=risk_budget_pct,
        account_value=account_value,
    )
    if levels is None:
        return (
            "### Computed risk levels\n\n"
            f"Unavailable — insufficient price history to measure volatility for "
            f"{symbol} as of {curr_date}. State that a volatility-scaled stop could "
            "not be computed. Do NOT invent a stop-loss, target, or position size."
        )
    return levels.as_markdown()
