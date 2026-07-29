"""Fact/opinion provenance, calibrated probabilities, and the guards on both.

These cover the two structural failures seen in live output:

* a decision whose entire evidence base was reported figures presented as
  established fact, and
* a bullish rating with no stated confidence, which gave the desk no way to
  size it or to know how much of it was conviction.
"""

from __future__ import annotations

import pytest

from tradingagents.agents.schemas import (
    Claim,
    PortfolioDecision,
    PortfolioRating,
    Provenance,
    ScenarioProbabilities,
    SentimentBand,
    SentimentReport,
    TraderAction,
    TraderProposal,
    ValuationAssessment,
    ValuationStance,
    render_pm_decision,
    render_sentiment_report,
    render_trader_proposal,
    render_valuation_assessment,
)


def _probs(up=0.4, flat=0.35, down=0.25, horizon=21):
    return ScenarioProbabilities(
        prob_upside=up, prob_flat=flat, prob_downside=down, horizon_days=horizon
    )


def _decision(**overrides):
    base = {
        "rating": PortfolioRating.OVERWEIGHT,
        "probabilities": _probs(),
        "conviction": "medium",
        "executive_summary": "Accumulate on weakness.",
        "investment_thesis": "Diversification offsets refining cyclicality.",
    }
    base.update(overrides)
    return PortfolioDecision(**base)


@pytest.mark.unit
class TestScenarioProbabilities:
    def test_well_formed_distribution_is_untouched(self):
        p = _probs(0.5, 0.3, 0.2)
        assert p.prob_upside == pytest.approx(0.5)
        assert p.edge == pytest.approx(0.3)

    def test_distribution_is_renormalised_rather_than_rejected(self):
        """A model emitting 0.45/0.35/0.25 must not discard the whole response."""
        p = _probs(0.45, 0.35, 0.25)  # sums to 1.05
        total = p.prob_upside + p.prob_flat + p.prob_downside
        assert total == pytest.approx(1.0)
        assert p.prob_upside == pytest.approx(0.45 / 1.05)

    def test_all_zero_collapses_to_range_bound(self):
        p = _probs(0.0, 0.0, 0.0)
        assert p.prob_flat == pytest.approx(1.0)
        assert p.edge == pytest.approx(0.0)

    def test_out_of_range_values_are_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ScenarioProbabilities(prob_upside=1.5, prob_flat=0.0, prob_downside=0.0)

    def test_modal_outcome(self):
        assert _probs(0.6, 0.3, 0.1).expected_direction == "Upside"
        assert _probs(0.1, 0.3, 0.6).expected_direction == "Downside"
        assert _probs(0.2, 0.6, 0.2).expected_direction == "Range-bound"

    def test_rendered_table_states_the_horizon_and_edge(self):
        text = _probs(0.5, 0.3, 0.2, horizon=21).rendered()
        assert "21 trading days" in text
        assert "50%" in text
        assert "Directional edge: +30%" in text


@pytest.mark.unit
class TestClaimProvenance:
    def test_rendered_claim_carries_its_tag(self):
        claim = Claim(
            statement="Trailing-twelve-month revenue growth is 8.5%.",
            provenance=Provenance.VERIFIED,
            source="fact sheet, revenue growth TTM YoY",
        )
        rendered = claim.rendered()
        assert "`VERIFIED`" in rendered
        assert "fact sheet" in rendered

    def test_reported_claim_keeps_its_attribution(self):
        claim = Claim(
            statement="Management reported 25% revenue growth.",
            provenance=Provenance.REPORTED,
            source="earnings coverage, Q1 FY2027",
        )
        assert "`REPORTED`" in claim.rendered()
        assert "Q1 FY2027" in claim.rendered()

    def test_source_is_optional(self):
        claim = Claim(statement="Momentum is deteriorating.", provenance=Provenance.INFERRED)
        assert "`INFERRED`" in claim.rendered()


@pytest.mark.unit
class TestRatingProbabilityConflict:
    """A bullish label on a bearish distribution is the failure to surface."""

    def test_overweight_with_negative_edge_conflicts(self):
        decision = _decision(
            rating=PortfolioRating.OVERWEIGHT, probabilities=_probs(0.2, 0.3, 0.5)
        )
        conflict = decision.rating_probability_conflict
        assert conflict is not None
        assert "favours downside" in conflict

    def test_sell_with_positive_edge_conflicts(self):
        decision = _decision(
            rating=PortfolioRating.SELL, probabilities=_probs(0.6, 0.2, 0.2)
        )
        assert "favours upside" in decision.rating_probability_conflict

    def test_consistent_call_has_no_conflict(self):
        assert _decision(
            rating=PortfolioRating.OVERWEIGHT, probabilities=_probs(0.5, 0.3, 0.2)
        ).rating_probability_conflict is None

    def test_hold_never_conflicts(self):
        assert _decision(
            rating=PortfolioRating.HOLD, probabilities=_probs(0.1, 0.2, 0.7)
        ).rating_probability_conflict is None

    def test_conflict_is_surfaced_in_the_rendered_memo(self):
        decision = _decision(
            rating=PortfolioRating.BUY, probabilities=_probs(0.2, 0.2, 0.6)
        )
        assert "RATING / PROBABILITY CONFLICT" in render_pm_decision(decision)


@pytest.mark.unit
class TestRenderedDecision:
    def test_conviction_and_distribution_appear(self):
        markdown = render_pm_decision(_decision(conviction="high"))
        assert "**Conviction**: High" in markdown
        assert "Outcome distribution" in markdown

    def test_evidence_base_is_counted_by_provenance(self):
        decision = _decision(key_facts=[
            Claim(statement="a", provenance=Provenance.VERIFIED),
            Claim(statement="b", provenance=Provenance.REPORTED),
            Claim(statement="c", provenance=Provenance.REPORTED),
            Claim(statement="d", provenance=Provenance.INFERRED),
        ])
        markdown = render_pm_decision(decision)
        assert "1 verified · 2 reported · 1 inferred" in markdown

    def test_all_unverified_evidence_raises_a_banner(self):
        """The shape of the live failure: nothing measured underneath."""
        decision = _decision(key_facts=[
            Claim(statement="25% revenue growth", provenance=Provenance.REPORTED),
            Claim(statement="Jio will keep compounding", provenance=Provenance.INFERRED),
        ])
        assert "NO VERIFIED EVIDENCE" in render_pm_decision(decision)

    def test_one_verified_claim_clears_the_banner(self):
        decision = _decision(key_facts=[
            Claim(statement="TTM growth 8.5%", provenance=Provenance.VERIFIED),
            Claim(statement="25% quarterly", provenance=Provenance.REPORTED),
        ])
        assert "NO VERIFIED EVIDENCE" not in render_pm_decision(decision)

    def test_invalidation_and_gaps_are_rendered(self):
        decision = _decision(
            invalidation_triggers=["Close below 1238 on the daily"],
            data_gaps=["StockTwits and Reddit returned nothing"],
        )
        markdown = render_pm_decision(decision)
        assert "Invalidation triggers" in markdown
        assert "Close below 1238" in markdown
        assert "Data gaps this run" in markdown

    def test_stop_loss_is_rendered_when_present(self):
        assert "**Stop Loss**: 1238.0" in render_pm_decision(_decision(stop_loss=1238.0))

    def test_nullish_levels_coerce_to_none(self):
        decision = _decision(price_target="N/A", stop_loss="unknown")
        assert decision.price_target is None
        assert decision.stop_loss is None


@pytest.mark.unit
class TestSentimentIndependenceGate:
    """An empty social read must not enter the debate as a second opinion."""

    def _report(self, sources, band=SentimentBand.MILDLY_BULLISH, score=5.8):
        return SentimentReport(
            overall_band=band,
            overall_score=score,
            confidence="low",
            sources_with_data=sources,
            narrative="Narrative body.",
        )

    def test_news_only_is_not_an_independent_signal(self):
        assert not self._report(["news"]).is_independent_signal

    def test_no_sources_is_not_an_independent_signal(self):
        assert not self._report([]).is_independent_signal

    def test_any_social_source_makes_it_independent(self):
        assert self._report(["news", "stocktwits"]).is_independent_signal
        assert self._report(["reddit"]).is_independent_signal

    def test_source_matching_is_case_and_space_tolerant(self):
        assert self._report([" StockTwits "]).is_independent_signal

    def test_rendered_report_warns_when_not_independent(self):
        """The exact live case: RELIANCE.NS with both social sources empty."""
        text = render_sentiment_report(self._report(["news"]))
        assert "NOT AN INDEPENDENT SIGNAL" in text
        assert "must not treat it as a second confirming opinion" in text

    def test_rendered_report_is_clean_when_independent(self):
        text = render_sentiment_report(self._report(["news", "stocktwits", "reddit"]))
        assert "NOT AN INDEPENDENT SIGNAL" not in text
        assert "news, stocktwits, reddit" in text

    def test_sources_default_to_empty(self):
        report = SentimentReport(
            overall_band=SentimentBand.NEUTRAL, overall_score=5.0,
            confidence="low", narrative="n",
        )
        assert report.sources_with_data == []
        assert not report.is_independent_signal


@pytest.mark.unit
class TestTraderConfidence:
    def test_confidence_defaults_to_medium_and_renders(self):
        proposal = TraderProposal(action=TraderAction.BUY, reasoning="Because.")
        assert proposal.confidence == "medium"
        assert "**Confidence**: Medium" in render_trader_proposal(proposal)

    def test_low_confidence_is_preserved(self):
        proposal = TraderProposal(
            action=TraderAction.HOLD, reasoning="Signals conflict.", confidence="low"
        )
        assert "**Confidence**: Low" in render_trader_proposal(proposal)

    def test_final_proposal_line_is_preserved(self):
        proposal = TraderProposal(action=TraderAction.SELL, reasoning="x")
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in render_trader_proposal(proposal)


@pytest.mark.unit
class TestValuationAssessment:
    def _assessment(self, **overrides):
        base = {
            "stance": ValuationStance.DISCOUNT,
            "discount_is_justified": False,
            "justification": "Growth and margins match peers.",
            "peer_context": "Trades 30% below the peer P/E median.",
        }
        base.update(overrides)
        return ValuationAssessment(**base)

    def test_unjustified_discount_reads_as_a_mispricing(self):
        text = render_valuation_assessment(self._assessment())
        assert "**Discount**" in text
        assert "NOT justified" in text
        assert "potential mispricing" in text

    def test_justified_discount_says_so(self):
        text = render_valuation_assessment(self._assessment(discount_is_justified=True))
        assert "justified by fundamentals" in text
        assert "NOT justified" not in text

    def test_multiples_carry_provenance(self):
        assessment = self._assessment(key_multiples=[
            Claim(statement="P/E 23.18 vs peer median 10.0",
                  provenance=Provenance.VERIFIED, source="computed peer table"),
        ])
        text = render_valuation_assessment(assessment)
        assert "`VERIFIED`" in text
        assert "peer median" in text

    def test_caveats_are_rendered_when_present(self):
        text = render_valuation_assessment(
            self._assessment(caveats="Conglomerate segments warrant different multiples.")
        )
        assert "**Caveats**" in text
