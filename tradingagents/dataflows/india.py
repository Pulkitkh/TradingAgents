"""India-market primitives: exchange identity, IST sessions, INR formatting.

TradingAgents was built US-first (SPY benchmark, US market hours, US-centric
social sources). Running it against NSE/BSE tickers for live decisions needs a
few things the generic code path cannot infer:

* **Session awareness.** A decision taken at 18:00 IST on a trading day is
  acting on that day's close; the same decision at 09:00 IST is acting on the
  *previous* close. The generic pipeline has no notion of this, so a run could
  silently reason on a stale bar (observed: a Monday-evening run that priced off
  the previous Friday). :func:`session_state` and :func:`assess_freshness` make
  the gap explicit and quantified in trading sessions, not calendar days.

* **Rupee magnitudes.** Indian financial reporting uses lakh/crore, and a raw
  ``17321564307456`` in a report is unreadable to the desk that has to act on
  it. :func:`format_inr` renders Indian digit grouping and crore units.

Nothing here calls the network: session and freshness logic operate on dates the
caller already has, so it stays deterministic and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

# Indian exchange suffixes as Yahoo Finance spells them.
NSE_SUFFIX = ".NS"
BSE_SUFFIX = ".BO"
INDIAN_SUFFIXES = (NSE_SUFFIX, BSE_SUFFIX)

# India Standard Time. Fixed offset — India observes no daylight saving, so a
# fixed-offset tzinfo is exactly correct here and avoids a tzdata dependency.
IST = timezone(timedelta(hours=5, minutes=30), name="IST")

# NSE/BSE continuous equity session (pre-open auction runs 09:00-09:15).
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_OPEN_START = time(9, 0)

EXCHANGE_NAMES = {NSE_SUFFIX: "NSE", BSE_SUFFIX: "BSE"}

# Benchmarks per exchange, matching DEFAULT_CONFIG["benchmark_map"].
EXCHANGE_BENCHMARKS = {"NSE": "^NSEI", "BSE": "^BSESN"}


def is_indian_equity(ticker: str) -> bool:
    """Whether ``ticker`` is an NSE/BSE-listed equity by Yahoo suffix."""
    if not isinstance(ticker, str):
        return False
    return ticker.upper().endswith(INDIAN_SUFFIXES)


def exchange_name(ticker: str) -> str | None:
    """``"NSE"``, ``"BSE"``, or ``None`` for non-Indian tickers."""
    if not isinstance(ticker, str):
        return None
    upper = ticker.upper()
    for suffix, name in EXCHANGE_NAMES.items():
        if upper.endswith(suffix):
            return name
    return None


def benchmark_for(ticker: str) -> str | None:
    """Index ticker to measure alpha against, or ``None`` if not Indian."""
    name = exchange_name(ticker)
    return EXCHANGE_BENCHMARKS.get(name) if name else None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def session_state(moment: datetime | None = None) -> str:
    """Classify an instant against the NSE/BSE equity session.

    Returns one of ``"weekend"``, ``"pre_open"``, ``"open"``, ``"closed"``.
    Naive datetimes are read as already being IST; aware ones are converted.

    Exchange holidays are deliberately *not* modelled here — a hardcoded holiday
    table goes stale every year and silently lies once it does. Holiday-aware
    staleness comes from :func:`assess_freshness`, which counts actual sessions
    present in the price data instead of assuming a calendar.
    """
    moment = moment or datetime.now(IST)
    moment = moment.astimezone(IST) if moment.tzinfo else moment.replace(tzinfo=IST)

    if moment.weekday() >= 5:  # Saturday/Sunday
        return "weekend"

    clock = moment.time()
    if clock < PRE_OPEN_START:
        return "closed"
    if clock < MARKET_OPEN:
        return "pre_open"
    if clock <= MARKET_CLOSE:
        return "open"
    return "closed"


def is_trading_hours(moment: datetime | None = None) -> bool:
    """Whether the continuous session is currently running."""
    return session_state(moment) == "open"


# ---------------------------------------------------------------------------
# Data freshness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Freshness:
    """How current the price data is, measured in trading sessions.

    ``sessions_behind`` counts sessions *observed in the data itself* between
    the latest bar and the analysis date, so exchange holidays never inflate it
    the way a calendar-day difference would.
    """

    latest_bar: date | None
    analysis_date: date
    sessions_behind: int
    calendar_days_behind: int
    is_current: bool
    note: str

    def as_markdown(self) -> str:
        if self.latest_bar is None:
            return "- **Price data freshness:** UNKNOWN — no dated rows available."
        status = "CURRENT" if self.is_current else "STALE"
        return (
            f"- **Price data freshness:** {status} — latest bar "
            f"{self.latest_bar.isoformat()}, analysis date "
            f"{self.analysis_date.isoformat()} "
            f"({self.sessions_behind} trading session(s) behind). {self.note}"
        )


def assess_freshness(
    session_dates: object,
    analysis_date: str | date | datetime,
    *,
    max_sessions_behind: int = 0,
) -> Freshness:
    """Quantify the gap between the newest price bar and the analysis date.

    ``session_dates`` is any iterable of dates present in the OHLCV frame (the
    ``Date`` column or index). Sessions strictly after the latest bar and on or
    before ``analysis_date`` cannot be counted from the data itself — by
    definition they are missing — so the shortfall is estimated from weekdays,
    which is exact except across an exchange holiday, where it errs on the side
    of reporting *more* staleness than there is. For a live-trading guard that
    is the safe direction to be wrong in.
    """
    analysis = _to_date(analysis_date)

    dates = sorted({d for d in (_to_date(x) for x in _iter_dates(session_dates)) if d})
    if not dates:
        return Freshness(
            latest_bar=None,
            analysis_date=analysis,
            sessions_behind=0,
            calendar_days_behind=0,
            is_current=False,
            note="No dated price rows were returned; treat every price claim as unverified.",
        )

    on_or_before = [d for d in dates if d <= analysis]
    latest = on_or_before[-1] if on_or_before else dates[0]

    calendar_gap = max(0, (analysis - latest).days)
    sessions_behind = _weekdays_between(latest, analysis)

    is_current = sessions_behind <= max_sessions_behind
    if is_current:
        note = "Decision inputs reflect the most recent completed session."
    else:
        note = (
            f"At least {sessions_behind} completed session(s) are missing from the "
            "price data. Any intraday or same-day move is NOT reflected below; "
            "re-run after the vendor publishes the missing bar before acting on "
            "price-sensitive levels."
        )

    return Freshness(
        latest_bar=latest,
        analysis_date=analysis,
        sessions_behind=sessions_behind,
        calendar_days_behind=calendar_gap,
        is_current=is_current,
        note=note,
    )


def _iter_dates(source: object):
    if source is None:
        return []
    if isinstance(source, pd.DataFrame):
        if "Date" in source.columns:
            return list(source["Date"])
        return list(source.index)
    if isinstance(source, pd.Series):
        return list(source)
    if isinstance(source, pd.Index):
        return list(source)
    try:
        return list(source)  # type: ignore[arg-type]
    except TypeError:
        return []


def _to_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    stamp = pd.to_datetime(value, errors="coerce")
    if stamp is None or pd.isna(stamp):
        return None
    return stamp.date()


def _weekdays_between(start: date, end: date) -> int:
    """Count weekdays strictly after ``start`` and on or before ``end``."""
    if end <= start:
        return 0
    count = 0
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count


# ---------------------------------------------------------------------------
# INR formatting
# ---------------------------------------------------------------------------

CRORE = 10_000_000
LAKH = 100_000


def indian_digit_group(number: float | int, decimals: int = 0) -> str:
    """Group digits the Indian way: last three, then pairs (12,34,567.89)."""
    negative = number < 0
    magnitude = abs(float(number))
    quantised = f"{magnitude:.{decimals}f}"
    whole, _, fraction = quantised.partition(".")

    if len(whole) <= 3:
        grouped = whole
    else:
        head, tail = whole[:-3], whole[-3:]
        pairs = []
        while len(head) > 2:
            pairs.insert(0, head[-2:])
            head = head[:-2]
        if head:
            pairs.insert(0, head)
        grouped = ",".join(pairs) + "," + tail

    if fraction:
        grouped = f"{grouped}.{fraction}"
    return f"-{grouped}" if negative else grouped


def format_inr(
    value: float | int | None, *, unit: str = "auto", decimals: int = 2
) -> str:
    """Render a rupee amount in the units an Indian desk actually reads.

    ``unit`` is ``"auto"`` (crore above ₹1 crore, lakh above ₹1 lakh, else
    plain), or an explicit ``"crore"`` / ``"lakh"`` / ``"plain"``.

    Two decimals by default because this renders order levels as well as
    aggregates: a stop quoted as "₹1,238" when the computed level is 1238.05 is
    a different order than the one the sizing maths assumed. Callers wanting a
    headline figure can pass ``decimals=0``.
    """
    if value is None:
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(amount):
        return "N/A"

    magnitude = abs(amount)
    resolved = unit
    if unit == "auto":
        if magnitude >= CRORE:
            resolved = "crore"
        elif magnitude >= LAKH:
            resolved = "lakh"
        else:
            resolved = "plain"

    if resolved == "crore":
        return f"₹{indian_digit_group(amount / CRORE, decimals)} Cr"
    if resolved == "lakh":
        return f"₹{indian_digit_group(amount / LAKH, decimals)} L"
    return f"₹{indian_digit_group(amount, decimals)}"


def format_pct(value: float | None, *, digits: int = 1, already_pct: bool = False) -> str:
    """Render a ratio (0.25) or percentage (25.0) as ``"+25.0%"``."""
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(number):
        return "N/A"
    pct = number if already_pct else number * 100
    return f"{pct:+.{digits}f}%"


def format_ratio(value: float | None, *, digits: int = 2) -> str:
    """Render a plain multiple (P/E, D/E) with a fixed precision."""
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(number):
        return "N/A"
    return f"{number:.{digits}f}"
