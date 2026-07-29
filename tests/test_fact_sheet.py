"""Fact sheet assembly, fail-soft behaviour, and delivery to every agent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from tradingagents.agents.utils.agent_utils import (
    EVIDENCE_RULES,
    get_fact_sheet_from_state,
)
from tradingagents.dataflows import fact_sheet as fs
from tradingagents.dataflows.fact_sheet import PROVENANCE_CONTRACT, build_fact_sheet


@pytest.fixture()
def stub_sections(monkeypatch):
    """Replace every fact-sheet section with a marker; no network, no yfinance."""
    monkeypatch.setattr(fs, "build_verified_market_snapshot", lambda *a, **k: "SNAPSHOT")
    monkeypatch.setattr(fs, "build_peer_comparison", lambda *a, **k: "PEERS")
    monkeypatch.setattr(fs, "render_risk_levels", lambda *a, **k: "RISK")
    monkeypatch.setattr(
        fs, "compute_fundamental_metrics",
        lambda *a, **k: MagicMock(as_markdown=lambda: "FUNDAMENTALS"),
    )
    monkeypatch.setattr(
        fs, "load_ohlcv",
        lambda *a, **k: pd.DataFrame({"Date": pd.to_datetime(["2026-07-24"])}),
    )


@pytest.mark.unit
class TestProvenanceContract:
    def test_contract_defines_all_three_tiers(self):
        for tier in ("VERIFIED", "REPORTED", "INFERRED"):
            assert tier in PROVENANCE_CONTRACT

    def test_contract_forbids_filling_gaps(self):
        assert "Do not estimate it" in PROVENANCE_CONTRACT
        assert "could not be verified" in PROVENANCE_CONTRACT

    def test_contract_states_no_llm_produced_the_numbers(self):
        assert "No language model produced any number" in PROVENANCE_CONTRACT


@pytest.mark.unit
class TestFactSheetAssembly:
    def test_all_sections_present_for_an_indian_equity(self, stub_sections):
        sheet = build_fact_sheet("RELIANCE.NS", "2026-07-27")
        for marker in ("SNAPSHOT", "FUNDAMENTALS", "PEERS", "RISK"):
            assert marker in sheet
        assert "VERIFIED FACT SHEET — RELIANCE.NS as of 2026-07-27" in sheet

    def test_nse_listing_context_is_stated(self, stub_sections):
        sheet = build_fact_sheet("RELIANCE.NS", "2026-07-27")
        assert "NSE" in sheet
        assert "09:15–15:30 IST" in sheet
        assert "T+1" in sheet

    def test_stale_data_raises_an_action_gate(self, stub_sections):
        """Friday's bar on a Monday run must gate level-sensitive advice."""
        sheet = build_fact_sheet("RELIANCE.NS", "2026-07-27")
        assert "STALE" in sheet
        assert "Action gate" in sheet
        assert "provisional" in sheet

    def test_current_data_has_no_action_gate(self, stub_sections, monkeypatch):
        monkeypatch.setattr(
            fs, "load_ohlcv",
            lambda *a, **k: pd.DataFrame({"Date": pd.to_datetime(["2026-07-27"])}),
        )
        sheet = build_fact_sheet("RELIANCE.NS", "2026-07-27")
        assert "CURRENT" in sheet
        assert "Action gate" not in sheet

    def test_crypto_skips_fundamentals_and_peers(self, stub_sections):
        sheet = build_fact_sheet("BTC-USD", "2026-07-27", asset_type="crypto")
        assert "SNAPSHOT" in sheet
        assert "FUNDAMENTALS" not in sheet
        assert "PEERS" not in sheet

    def test_sections_can_be_disabled(self, stub_sections):
        sheet = build_fact_sheet(
            "RELIANCE.NS", "2026-07-27", include_peers=False, include_risk=False
        )
        assert "PEERS" not in sheet
        assert "RISK" not in sheet
        assert "SNAPSHOT" in sheet

    def test_risk_budget_and_account_value_reach_the_risk_block(self, stub_sections, monkeypatch):
        seen = {}

        def capture(symbol, curr_date, **kwargs):
            seen.update(kwargs)
            return "RISK"

        monkeypatch.setattr(fs, "render_risk_levels", capture)
        build_fact_sheet(
            "RELIANCE.NS", "2026-07-27", account_value=2_500_000, risk_budget_pct=0.5
        )
        assert seen["account_value"] == 2_500_000
        assert seen["risk_budget_pct"] == 0.5


@pytest.mark.unit
class TestFailSoft:
    """One dead vendor costs one block, never the run."""

    def test_broken_section_degrades_to_a_notice(self, stub_sections, monkeypatch):
        def explode(*a, **k):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(fs, "build_peer_comparison", explode)
        sheet = build_fact_sheet("RELIANCE.NS", "2026-07-27")
        assert "Unavailable (RuntimeError)" in sheet
        assert "do not claim the stock is cheap or expensive" in sheet
        # The surviving sections are still present.
        assert "SNAPSHOT" in sheet and "RISK" in sheet

    def test_broken_freshness_check_says_unknown(self, stub_sections, monkeypatch):
        def explode(*a, **k):
            raise RuntimeError("no data")

        monkeypatch.setattr(fs, "load_ohlcv", explode)
        sheet = build_fact_sheet("RELIANCE.NS", "2026-07-27")
        assert "freshness:** UNKNOWN" in sheet

    def test_broken_snapshot_forbids_price_claims(self, stub_sections, monkeypatch):
        def explode(*a, **k):
            raise ValueError("no rows")

        monkeypatch.setattr(fs, "build_verified_market_snapshot", explode)
        sheet = build_fact_sheet("RELIANCE.NS", "2026-07-27")
        assert "treat every price-level claim as unverified" in sheet


@pytest.mark.unit
class TestFactSheetDelivery:
    """Agents must never silently receive nothing."""

    def test_present_sheet_is_returned_verbatim(self):
        assert get_fact_sheet_from_state({"fact_sheet": "REAL SHEET"}) == "REAL SHEET"

    def test_missing_sheet_yields_an_explicit_no_evidence_notice(self):
        for state in ({}, {"fact_sheet": ""}, {"fact_sheet": "   "}, {"fact_sheet": None}):
            text = get_fact_sheet_from_state(state)
            assert "UNAVAILABLE" in text
            assert "You have no verified figures" in text

    def test_evidence_rules_forbid_inventing_figures(self):
        assert "unavailable rather than supplying it" in EVIDENCE_RULES
        assert "single quarter" in EVIDENCE_RULES

    def test_evidence_rules_carry_no_template_braces(self):
        # Embedded in ChatPromptTemplate strings, where braces become variables.
        assert "{" not in EVIDENCE_RULES and "}" not in EVIDENCE_RULES
