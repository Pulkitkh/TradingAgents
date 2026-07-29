"""Peer-group resolution and the computed relative-valuation table."""

from __future__ import annotations

import pytest

from tradingagents.dataflows import peers
from tradingagents.dataflows.peers import (
    NSE_PEER_GROUPS,
    build_peer_comparison,
    resolve_peer_group,
)


@pytest.fixture(autouse=True)
def _clear_peer_cache():
    """Reset the peer lookup cache around each test.

    Teardown clears the *original* cached function rather than whatever is bound
    at the end: tests that monkeypatch ``_peer_info`` replace it with a plain
    lambda, which has no ``cache_clear``.
    """
    cached = peers._peer_info
    cached.cache_clear()
    yield
    cached.cache_clear()


@pytest.mark.unit
class TestPeerGroupResolution:
    def test_explicit_override_wins_over_industry_string(self):
        # Yahoo files Reliance under refining, which misses Jio and Retail; the
        # curated override is what keeps the comparison honest.
        group, group_peers = resolve_peer_group(
            "RELIANCE.NS", {"industry": "Software - Infrastructure"}
        )
        assert group == "oil_gas"
        assert "ONGC.NS" in group_peers

    def test_subject_is_excluded_from_its_own_peer_set(self):
        _, group_peers = resolve_peer_group("RELIANCE.NS", {})
        assert "RELIANCE.NS" not in [p.upper() for p in group_peers]

    def test_industry_keyword_fallback(self):
        group, group_peers = resolve_peer_group(
            "SOMECO.NS", {"industry": "Drug Manufacturers - Specialty & Generic"}
        )
        assert group == "pharma"
        assert "SUNPHARMA.NS" in group_peers

    def test_sector_keyword_also_matches(self):
        group, _ = resolve_peer_group("SOMEBANK.NS", {"sector": "Banks - Regional"})
        assert group == "banks"

    def test_unmatched_industry_resolves_to_nothing(self):
        group, group_peers = resolve_peer_group(
            "OBSCURE.NS", {"industry": "Interstellar Freight"}
        )
        assert group is None
        assert group_peers == ()

    def test_missing_info_resolves_to_nothing_for_unknown_tickers(self):
        assert resolve_peer_group("OBSCURE.NS", None) == (None, ())

    def test_peer_count_is_capped(self):
        _, group_peers = resolve_peer_group("SOMECO.NS", {"industry": "steel"})
        assert len(group_peers) <= peers.MAX_PEERS

    def test_every_curated_group_is_non_trivial(self):
        for name, members in NSE_PEER_GROUPS.items():
            assert len(members) >= 2, f"{name} needs at least two members"
            assert all(m.endswith((".NS", ".BO")) for m in members), name


def _install_infos(monkeypatch, infos: dict[str, dict]):
    monkeypatch.setattr(peers, "_peer_info", lambda ticker: infos.get(ticker, {}))


@pytest.mark.unit
class TestPeerComparisonTable:
    SUBJECT = {
        "industry": "Oil & Gas Refining & Marketing",
        "marketCap": 17_321_564_307_456,
        "trailingPE": 23.18,
        "forwardPE": 17.90,
        "priceToBook": 1.92,
        "enterpriseToEbitda": 11.0,
        "returnOnEquity": 0.09,
        "operatingMargins": 0.12,
    }

    def _peer(self, pe, pb, roe):
        return {
            "marketCap": 1_000_000_000_000,
            "trailingPE": pe,
            "forwardPE": pe - 3,
            "priceToBook": pb,
            "enterpriseToEbitda": 8.0,
            "returnOnEquity": roe,
            "operatingMargins": 0.10,
        }

    def test_table_includes_subject_peers_and_median(self, monkeypatch):
        _install_infos(monkeypatch, {
            "ONGC.NS": self._peer(9.0, 0.9, 0.16),
            "IOC.NS": self._peer(11.0, 1.1, 0.14),
            "BPCL.NS": self._peer(13.0, 1.3, 0.18),
        })
        table = build_peer_comparison("RELIANCE.NS", "2026-07-27", subject_info=self.SUBJECT)

        assert "RELIANCE.NS" in table and "(subject)" in table
        assert "ONGC.NS" in table
        assert "_Peer median_" in table
        assert "Subject vs peer median" in table

    def test_premium_on_pe_is_labelled_more_expensive(self, monkeypatch):
        _install_infos(monkeypatch, {
            "ONGC.NS": self._peer(9.0, 0.9, 0.16),
            "IOC.NS": self._peer(11.0, 1.1, 0.14),
        })
        table = build_peer_comparison("RELIANCE.NS", "2026-07-27", subject_info=self.SUBJECT)
        pe_line = next(line for line in table.splitlines() if line.startswith("- P/E (TTM):"))
        # Subject 23.18 against a peer median of 10.0 is a large premium.
        assert "more expensive than peers" in pe_line
        assert "+" in pe_line

    def test_roe_inverts_the_reading(self, monkeypatch):
        """Lower ROE than peers is a weakness, not a bargain."""
        _install_infos(monkeypatch, {
            "ONGC.NS": self._peer(9.0, 0.9, 0.16),
            "IOC.NS": self._peer(11.0, 1.1, 0.18),
        })
        table = build_peer_comparison("RELIANCE.NS", "2026-07-27", subject_info=self.SUBJECT)
        roe_line = next(line for line in table.splitlines() if line.startswith("- ROE:"))
        assert "worse than peers" in roe_line

    def test_sitting_on_the_median_reads_as_in_line(self, monkeypatch):
        """Without a neutral band, an exact match got labelled 'more expensive'."""
        _install_infos(monkeypatch, {
            "ONGC.NS": self._peer(23.18, 1.92, 0.09),
            "IOC.NS": self._peer(23.18, 1.92, 0.09),
        })
        table = build_peer_comparison("RELIANCE.NS", "2026-07-27", subject_info=self.SUBJECT)
        pe_line = next(line for line in table.splitlines() if line.startswith("- P/E (TTM):"))
        roe_line = next(line for line in table.splitlines() if line.startswith("- ROE:"))
        assert "in line with peers" in pe_line
        assert "in line with peers" in roe_line

    def test_missing_metric_is_reported_not_estimated(self, monkeypatch):
        _install_infos(monkeypatch, {"ONGC.NS": {"marketCap": 1e12}})
        subject = dict(self.SUBJECT)
        subject.pop("enterpriseToEbitda")
        table = build_peer_comparison("RELIANCE.NS", "2026-07-27", subject_info=subject)
        assert "not comparable (missing data)" in table

    def test_unresolvable_peer_group_refuses_to_compare(self, monkeypatch):
        _install_infos(monkeypatch, {})
        table = build_peer_comparison(
            "OBSCURE.NS", "2026-07-27", subject_info={"industry": "Interstellar Freight"}
        )
        assert "Unavailable" in table
        assert "could not be computed" in table

    def test_no_vendor_profile_refuses_to_compare(self, monkeypatch):
        _install_infos(monkeypatch, {})
        table = build_peer_comparison("OBSCURE.NS", "2026-07-27")
        assert "Unavailable" in table
        assert "cheap or expensive" in table

    def test_dead_peer_lookup_does_not_sink_the_table(self, monkeypatch):
        _install_infos(monkeypatch, {
            "ONGC.NS": self._peer(9.0, 0.9, 0.16),
            "IOC.NS": {},   # vendor returned nothing for this peer
        })
        table = build_peer_comparison("RELIANCE.NS", "2026-07-27", subject_info=self.SUBJECT)
        assert "ONGC.NS" in table
        assert "_Peer median_" in table

    def test_block_forbids_substituting_remembered_multiples(self, monkeypatch):
        _install_infos(monkeypatch, {"ONGC.NS": self._peer(9.0, 0.9, 0.16)})
        table = build_peer_comparison("RELIANCE.NS", "2026-07-27", subject_info=self.SUBJECT)
        assert "do not substitute a remembered sector" in table


@pytest.mark.unit
class TestMedian:
    def test_odd_and_even_counts(self):
        assert peers._median([3.0, 1.0, 2.0]) == 2.0
        assert peers._median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_empty_is_none(self):
        assert peers._median([]) is None
