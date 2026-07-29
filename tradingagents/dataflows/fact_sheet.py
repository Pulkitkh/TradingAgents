"""The verified fact sheet: every number the agents are allowed to state as fact.

This is the spine of the fact/opinion separation. One deterministic block is
computed before any agent runs — price and indicators, fundamentals with growth
bases spelled out, peer multiples, volatility-scaled risk levels, and a data
freshness verdict — and injected into every report-producing agent's prompt.

The contract given to the agents is deliberately narrow: a figure that appears
in this sheet may be stated as fact; a figure that does not must be labelled as
reported-by-a-source or as the agent's own inference. That turns "separate facts
from opinions" from a style request into something a reader can check, because
the fact sheet is reproducible without an LLM and is written to the report tree
alongside the prose.

Assembly fails soft section by section: a dead vendor costs one block, not the
run.
"""

from __future__ import annotations

import logging

from .fundamentals_metrics import compute_fundamental_metrics
from .india import assess_freshness, exchange_name, is_indian_equity, session_state
from .market_data_validator import build_verified_market_snapshot
from .peers import build_peer_comparison
from .risk_sizing import render_risk_levels
from .stockstats_utils import load_ohlcv
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

PROVENANCE_CONTRACT = """\
## How to use this fact sheet

Every figure below was computed deterministically from vendor data before any
analysis began. No language model produced any number in this sheet.

Three rules govern every report you write:

1. **VERIFIED** — a figure that appears in this fact sheet. State it plainly.
   Quote it exactly; do not re-round, re-derive, or adjust it.
2. **REPORTED** — a figure from a news article, filing summary, or social post.
   You must attribute it and name its basis, e.g. "management reported 25%
   revenue growth for the September quarter (source: earnings coverage)". Never
   restate a REPORTED figure as though it were verified, and never use one as
   the sole support for a valuation conclusion.
3. **INFERRED** — your own estimate, projection, or judgement. Mark it as such.
   Inference is welcome; inference disguised as measurement is not.

If a figure you want to cite is absent from this sheet and absent from your
sources, say it is unavailable. Do not estimate it. An analysis that states
"this could not be verified" is more useful to the desk than one that fills the
gap with a plausible number.
"""


def build_fact_sheet(
    ticker: str,
    curr_date: str,
    *,
    asset_type: str = "stock",
    include_peers: bool = True,
    include_risk: bool = True,
    account_value: float | None = None,
    risk_budget_pct: float = 1.0,
) -> str:
    """Assemble the full verified fact sheet for ``ticker`` as of ``curr_date``."""
    canonical = normalize_symbol(ticker)
    sections: list[str] = [
        f"# VERIFIED FACT SHEET — {canonical} as of {curr_date}",
        "",
        PROVENANCE_CONTRACT,
        "",
        _market_context_block(canonical, curr_date),
    ]

    sections.append(_safe(
        lambda: build_verified_market_snapshot(canonical, curr_date),
        "verified market snapshot",
        "Price and indicator verification unavailable; treat every price-level "
        "claim as unverified and say so.",
    ))

    if asset_type != "crypto":
        sections.append(_safe(
            lambda: compute_fundamental_metrics(canonical, curr_date).as_markdown(),
            "fundamental metrics",
            "Computed fundamentals unavailable; do not assert growth rates, "
            "margins, or leverage figures.",
        ))

        if include_peers:
            sections.append(_safe(
                lambda: build_peer_comparison(canonical, curr_date),
                "peer comparison",
                "Peer comparison unavailable; do not claim the stock is cheap or "
                "expensive relative to its sector.",
            ))

    if include_risk:
        sections.append(_safe(
            lambda: render_risk_levels(
                canonical,
                curr_date,
                risk_budget_pct=risk_budget_pct,
                account_value=account_value,
            ),
            "risk levels",
            "Risk levels unavailable; do not state a stop-loss, target, or "
            "position size.",
        ))

    return "\n\n".join(section for section in sections if section)


def _market_context_block(canonical: str, curr_date: str) -> str:
    """Exchange identity, session state, and price-data freshness."""
    lines = ["## Market context", ""]

    exchange = exchange_name(canonical)
    if exchange:
        lines.append(
            f"- **Listing:** {exchange} · quoted in INR · settlement T+1 · "
            f"continuous session 09:15–15:30 IST."
        )
        lines.append(f"- **Session at analysis time:** {session_state().replace('_', ' ')}.")
    else:
        lines.append(f"- **Listing:** {canonical} (non-Indian listing).")

    try:
        data = load_ohlcv(canonical, curr_date)
        freshness = assess_freshness(data, curr_date)
        lines.append(freshness.as_markdown())
        if not freshness.is_current and is_indian_equity(canonical):
            lines.append(
                "- **Action gate:** the most recent completed session is missing "
                "from this data. Any level-sensitive instruction (entry, stop, "
                "target) is provisional until the run is repeated against "
                "current data."
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Freshness check failed for %s: %s", canonical, exc)
        lines.append(
            "- **Price data freshness:** UNKNOWN — the freshness check could not "
            "run. Treat price levels as unverified."
        )

    return "\n".join(lines)


def _safe(builder, label: str, fallback_note: str) -> str:
    """Run one fact-sheet section, degrading to an explicit notice on failure."""
    try:
        return builder()
    except Exception as exc:  # noqa: BLE001 — one dead vendor must not sink the sheet
        logger.warning("Fact sheet section %r failed: %s", label, exc)
        return f"### {label.title()}\n\nUnavailable ({type(exc).__name__}). {fallback_note}"
