"""Peer-group resolution and comparative valuation for Indian equities.

A multiple in isolation says nothing. "P/E of 23" is neither cheap nor dear
until it sits next to the sector — Indian FMCG routinely trades at 55x while
refiners trade in the teens, so an absolute number invites exactly the kind of
free-floating assertion this pipeline is trying to eliminate.

This module builds the comparison **in code**: it resolves a peer set, fetches
each peer's multiples, computes the group median, and expresses the subject's
premium or discount against it. The agent receives a finished table of numbers
it did not produce and cannot adjust, which is what makes the resulting
valuation commentary auditable.

Peer sets are curated per NSE sector rather than derived from the vendor's
industry string alone. Yahoo's classification is coarse — it files Reliance
under "Oil & Gas Refining & Marketing", which misses that the market prices it
substantially on Jio and Retail — so a hand-maintained map produces a more
honest comparison set, with vendor industry as the fallback.
"""

from __future__ import annotations

import functools
import logging

import pandas as pd
import yfinance as yf

from .india import format_inr, format_pct, format_ratio
from .stockstats_utils import yf_retry
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

MAX_PEERS = 6

# Curated NSE peer groups. Keys are internal group names; values are the
# constituent Yahoo tickers. Extend by adding rows — nothing else needs editing.
NSE_PEER_GROUPS: dict[str, tuple[str, ...]] = {
    "oil_gas": (
        "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS", "GAIL.NS",
    ),
    "it_services": (
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS",
    ),
    "banks": (
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS",
    ),
    "nbfc": (
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS",
        "MUTHOOTFIN.NS",
    ),
    "auto": (
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS",
    ),
    "fmcg": (
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
        "GODREJCP.NS",
    ),
    "pharma": (
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS",
        "AUROPHARMA.NS",
    ),
    "metals": (
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS",
        "SAIL.NS",
    ),
    "cement": (
        "ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS", "DALBHARAT.NS",
    ),
    "power": (
        "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIPOWER.NS", "JSWENERGY.NS",
    ),
    "telecom": ("BHARTIARTL.NS", "IDEA.NS", "RELIANCE.NS"),
    "infra": ("LT.NS", "ADANIPORTS.NS", "SIEMENS.NS", "ABB.NS"),
    "paints": ("ASIANPAINT.NS", "BERGEPAINT.NS", "KANSAINER.NS", "AKZOINDIA.NS"),
    "consumer_durables": ("TITAN.NS", "HAVELLS.NS", "VOLTAS.NS", "CROMPTON.NS"),
    "insurance": (
        "SBILIFE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS", "LICI.NS",
    ),
}

# Explicit ticker overrides, for names the vendor industry string misclassifies
# or that straddle several groups.
TICKER_GROUP_OVERRIDES: dict[str, str] = {
    "RELIANCE.NS": "oil_gas",
    "ITC.NS": "fmcg",
    "LT.NS": "infra",
    "BHARTIARTL.NS": "telecom",
    "JIOFIN.NS": "nbfc",
}

# Substrings of the vendor's sector/industry string mapped to a curated group.
_INDUSTRY_HINTS: tuple[tuple[str, str], ...] = (
    ("oil", "oil_gas"),
    ("gas", "oil_gas"),
    ("refin", "oil_gas"),
    ("petro", "oil_gas"),
    ("software", "it_services"),
    ("information technology", "it_services"),
    ("bank", "banks"),
    ("credit", "nbfc"),
    ("financial", "nbfc"),
    ("insurance", "insurance"),
    ("auto", "auto"),
    ("vehicle", "auto"),
    ("household", "fmcg"),
    ("packaged food", "fmcg"),
    ("beverage", "fmcg"),
    ("tobacco", "fmcg"),
    ("confection", "fmcg"),
    ("drug", "pharma"),
    ("pharma", "pharma"),
    ("biotech", "pharma"),
    ("steel", "metals"),
    ("alumin", "metals"),
    ("copper", "metals"),
    ("mining", "metals"),
    ("cement", "cement"),
    ("building material", "cement"),
    ("utilit", "power"),
    ("power", "power"),
    ("telecom", "telecom"),
    ("communication", "telecom"),
    ("engineering", "infra"),
    ("construction", "infra"),
    ("chemical", "paints"),
    ("specialty", "paints"),
)

# Columns rendered in the comparison table: (heading, info key, formatter).
_PEER_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("Mkt cap", "marketCap", "currency"),
    ("P/E (TTM)", "trailingPE", "multiple"),
    ("P/E (fwd)", "forwardPE", "multiple"),
    ("P/B", "priceToBook", "multiple"),
    ("EV/EBITDA", "enterpriseToEbitda", "multiple"),
    ("ROE", "returnOnEquity", "pct"),
    ("Op margin", "operatingMargins", "pct"),
)

# Multiples where the subject trading below the peer median is the cheaper
# outcome. ROE and margins invert: higher is better, so a "discount" there is a
# weakness, not an opportunity, and the rendered label must say so.
_LOWER_IS_CHEAPER = {"trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda"}

# Gaps this small are noise, not a valuation signal. Without a neutral band a
# subject sitting exactly on the peer median gets labelled "more expensive",
# which reads as a finding when it is a rounding artefact.
IN_LINE_BAND = 0.02


def resolve_peer_group(ticker: str, info: dict | None = None) -> tuple[str | None, tuple[str, ...]]:
    """Return ``(group_name, peers)`` for ``ticker``, excluding the subject itself.

    Resolution order: explicit override, then a vendor industry/sector keyword
    match. Returns ``(None, ())`` when neither resolves, which callers render as
    "peer comparison unavailable" rather than inventing a comparison set.
    """
    canonical = normalize_symbol(ticker).upper()

    group = TICKER_GROUP_OVERRIDES.get(canonical)
    if group is None and info:
        haystack = " ".join(
            str(info.get(key, "")) for key in ("industry", "sector", "industryKey", "sectorKey")
        ).lower()
        for hint, candidate in _INDUSTRY_HINTS:
            if hint in haystack:
                group = candidate
                break

    if group is None:
        return None, ()

    peers = tuple(p for p in NSE_PEER_GROUPS.get(group, ()) if p.upper() != canonical)
    return group, peers[:MAX_PEERS]


@functools.lru_cache(maxsize=128)
def _peer_info(ticker: str) -> dict:
    """Vendor snapshot for one peer. Cached; fails soft to an empty dict."""
    try:
        return yf_retry(lambda: yf.Ticker(ticker).info) or {}
    except Exception as exc:  # noqa: BLE001 — a missing peer must not sink the table
        logger.warning("Peer lookup failed for %s: %s", ticker, exc)
        return {}


def _numeric(info: dict, key: str) -> float | None:
    value = info.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _render(value: float | None, kind: str) -> str:
    if value is None:
        return "N/A"
    if kind == "currency":
        return format_inr(value)
    if kind == "pct":
        return format_pct(value)
    return format_ratio(value)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def build_peer_comparison(ticker: str, curr_date: str, subject_info: dict | None = None) -> str:
    """Render the deterministic peer-valuation block for prompt injection."""
    canonical = normalize_symbol(ticker).upper()

    info = subject_info
    if info is None:
        info = _peer_info(canonical)
    if not info:
        return (
            "### Peer valuation comparison\n\n"
            f"Unavailable — no vendor profile returned for {canonical}. "
            "Do not assert whether the stock is cheap or expensive relative to peers."
        )

    group, peers = resolve_peer_group(canonical, info)
    if not peers:
        return (
            "### Peer valuation comparison\n\n"
            f"Unavailable — no curated peer group matched {canonical} "
            f"(vendor industry: {info.get('industry') or 'unknown'}). "
            "State that relative valuation could not be computed rather than "
            "estimating a sector multiple."
        )

    rows = [(canonical, info)] + [(peer, _peer_info(peer)) for peer in peers]
    rows = [(name, data) for name, data in rows if data]

    header = "| Company | " + " | ".join(head for head, _, _ in _PEER_COLUMNS) + " |"
    divider = "|---|" + "|".join("---:" for _ in _PEER_COLUMNS) + "|"
    lines = [
        "### Peer valuation comparison (computed, not model-generated)",
        "",
        f"Peer group: **{group.replace('_', ' ')}** · as of {curr_date} · "
        f"{len(rows) - 1} peer(s) resolved.",
        "",
        header,
        divider,
    ]

    peer_values: dict[str, list[float]] = {key: [] for _, key, _ in _PEER_COLUMNS}
    for name, data in rows:
        cells = []
        for _, key, kind in _PEER_COLUMNS:
            value = _numeric(data, key)
            if name != canonical and value is not None:
                peer_values[key].append(value)
            cells.append(_render(value, kind))
        label = f"**{name}** (subject)" if name == canonical else name
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    medians = {key: _median(values) for key, values in peer_values.items()}
    lines.append(
        "| _Peer median_ | "
        + " | ".join(_render(medians[key], kind) for _, key, kind in _PEER_COLUMNS)
        + " |"
    )

    lines += ["", "**Subject vs peer median**", ""]
    for head, key, _ in _PEER_COLUMNS:
        subject_value = _numeric(info, key)
        median = medians.get(key)
        if subject_value is None or median in (None, 0):
            lines.append(f"- {head}: not comparable (missing data).")
            continue
        delta = (subject_value - median) / abs(median)
        if abs(delta) < IN_LINE_BAND:
            reading = "in line with peers"
        elif key in _LOWER_IS_CHEAPER:
            reading = "cheaper than peers" if delta < 0 else "more expensive than peers"
        else:
            reading = "better than peers" if delta > 0 else "worse than peers"
        lines.append(f"- {head}: {format_pct(delta)} vs peer median — {reading}.")

    lines += [
        "",
        "Use only these figures for relative-valuation claims. Peer medians are "
        "computed from the rows above; do not substitute a remembered sector "
        "multiple or extend the comparison to companies absent from this table.",
    ]
    return "\n".join(lines)
