"""Every decision agent must receive the analysis date and the fact sheet.

Two live failures motivate this file:

* The portfolio manager's memo was headed "October 24, 2023" on a run whose
  analysis date was 2026-07-27. No downstream agent received ``trade_date`` at
  all — only the analysts did — so a model asked to date a memo invented one.
* Analyst reports ended with "FINAL TRANSACTION PROPOSAL: **SELL**" while the
  final decision was Overweight, because every analyst prompt carried
  decision-agent boilerplate the analyst had no business acting on.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    ScenarioProbabilities,
    TraderAction,
    TraderProposal,
)
from tradingagents.agents.trader.trader import create_trader

TRADE_DATE = "2026-07-27"
FACT_SHEET = "VERIFIED FACT SHEET SENTINEL"


def _capturing_llm(captured, result):
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or result
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _prompt_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    parts = [
        m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        for m in prompt
    ]
    return "\n".join(str(p) for p in parts)


_RISK_STATE = {
    "history": "h", "aggressive_history": "a", "conservative_history": "c",
    "neutral_history": "n", "current_aggressive_response": "",
    "current_conservative_response": "", "current_neutral_response": "",
    "latest_speaker": "Neutral", "count": 1,
}
_DEBATE_STATE = {
    "history": "h", "bull_history": "b", "bear_history": "r",
    "current_response": "", "judge_decision": "", "count": 1,
}

_PM_RESULT = PortfolioDecision(
    rating=PortfolioRating.HOLD,
    probabilities=ScenarioProbabilities(
        prob_upside=0.33, prob_flat=0.34, prob_downside=0.33
    ),
    conviction="low",
    executive_summary="x",
    investment_thesis="y",
)


def _run_pm(captured, **state_extra):
    state = {
        "company_of_interest": "RELIANCE.NS",
        "trade_date": TRADE_DATE,
        "fact_sheet": FACT_SHEET,
        "risk_debate_state": _RISK_STATE,
        "investment_plan": "plan",
        "trader_investment_plan": "trader plan",
    }
    state.update(state_extra)
    create_portfolio_manager(_capturing_llm(captured, _PM_RESULT))(state)
    return _prompt_text(captured["prompt"])


def _run_trader(captured, **state_extra):
    state = {
        "company_of_interest": "RELIANCE.NS",
        "trade_date": TRADE_DATE,
        "fact_sheet": FACT_SHEET,
        "investment_plan": "**Recommendation**: Buy",
    }
    state.update(state_extra)
    result = TraderProposal(action=TraderAction.BUY, reasoning="x")
    create_trader(_capturing_llm(captured, result))(state)
    return _prompt_text(captured["prompt"])


def _run_research_manager(captured, **state_extra):
    state = {
        "company_of_interest": "RELIANCE.NS",
        "trade_date": TRADE_DATE,
        "fact_sheet": FACT_SHEET,
        "investment_debate_state": _DEBATE_STATE,
    }
    state.update(state_extra)
    result = ResearchPlan(
        recommendation=PortfolioRating.BUY, rationale="x", strategic_actions="y"
    )
    create_research_manager(_capturing_llm(captured, result))(state)
    return _prompt_text(captured["prompt"])


@pytest.mark.unit
class TestAnalysisDateReachesDecisionAgents:
    """The bug that produced an 'October 2023' header on a July 2026 run."""

    def test_portfolio_manager_receives_the_date(self):
        text = _run_pm({})
        assert TRADE_DATE in text
        assert "do not infer a date from the content" in text

    def test_trader_receives_the_date(self):
        assert TRADE_DATE in _run_trader({})

    def test_research_manager_receives_the_date(self):
        assert TRADE_DATE in _run_research_manager({})

    @pytest.mark.parametrize("runner", [_run_pm, _run_trader, _run_research_manager])
    def test_missing_date_degrades_without_crashing(self, runner):
        captured = {}
        text = runner(captured, trade_date=None)
        assert text  # a bare state must still produce a usable prompt

    def test_debaters_receive_the_date_and_fact_sheet(self):
        """All five debate agents read the same evidence block."""
        import tradingagents.agents.researchers.bear_researcher as bear
        import tradingagents.agents.researchers.bull_researcher as bull
        import tradingagents.agents.risk_mgmt.aggressive_debator as aggressive
        import tradingagents.agents.risk_mgmt.conservative_debator as conservative
        import tradingagents.agents.risk_mgmt.neutral_debator as neutral

        for module in (bull, bear, aggressive, conservative, neutral):
            source = inspect.getsource(module)
            assert "get_fact_sheet_from_state" in source, module.__name__
            assert "trade_date" in source, module.__name__


@pytest.mark.unit
class TestFactSheetReachesDecisionAgents:
    def test_portfolio_manager_receives_the_fact_sheet(self):
        assert FACT_SHEET in _run_pm({})

    def test_trader_receives_the_fact_sheet(self):
        assert FACT_SHEET in _run_trader({})

    def test_research_manager_receives_the_fact_sheet(self):
        assert FACT_SHEET in _run_research_manager({})

    def test_absent_fact_sheet_becomes_an_explicit_notice(self):
        text = _run_pm({}, fact_sheet="")
        assert "UNAVAILABLE" in text
        assert "You have no verified figures" in text


@pytest.mark.unit
class TestPortfolioManagerConstraints:
    def test_levels_must_come_from_the_computed_risk_block(self):
        text = _run_pm({})
        assert "Price levels are computed, not chosen" in text
        assert "an invented stop is not" in text

    def test_rating_must_agree_with_probabilities(self):
        text = _run_pm({})
        assert "Your rating and your probabilities must agree" in text

    def test_non_independent_sentiment_must_be_discounted(self):
        assert "not an independent signal" in _run_pm({})

    def test_valuation_report_is_injected_when_present(self):
        text = _run_pm({}, valuation_report="VALUATION SENTINEL")
        assert "VALUATION SENTINEL" in text

    def test_valuation_section_is_omitted_when_absent(self):
        assert "Relative Valuation (peer-anchored)" not in _run_pm({})


@pytest.mark.unit
class TestAnalystsDoNotIssueDecisions:
    """Analyst reports must not contradict the portfolio manager's memo."""

    ANALYST_MODULES = (
        "tradingagents.agents.analysts.market_analyst",
        "tradingagents.agents.analysts.news_analyst",
        "tradingagents.agents.analysts.fundamentals_analyst",
        "tradingagents.agents.analysts.sentiment_analyst",
    )

    def test_no_analyst_is_told_to_emit_a_transaction_proposal(self):
        import importlib

        for name in self.ANALYST_MODULES:
            source = inspect.getsource(importlib.import_module(name))
            assert "prefix your response with FINAL TRANSACTION PROPOSAL" not in source, name

    def test_every_analyst_is_told_it_is_not_the_decision_maker(self):
        import importlib

        for name in self.ANALYST_MODULES:
            source = inspect.getsource(importlib.import_module(name))
            assert "You are an analyst, not a decision maker" in source, name

    def test_fundamentals_system_message_is_a_string_not_a_tuple(self):
        """A stray trailing comma once made this a 1-tuple, so the rendered
        prompt contained a Python repr instead of the instructions."""
        import tradingagents.agents.analysts.fundamentals_analyst as fundamentals

        captured = {}

        def bound(prompt_value, *args, **kwargs):
            # LangChain coerces a non-Runnable into a RunnableLambda, so the
            # piped chain calls the bound object directly with the rendered
            # ChatPromptValue rather than going through `.invoke`.
            captured["messages"] = prompt_value.to_messages()
            return MagicMock(tool_calls=[], content="report")

        llm = MagicMock()
        llm.bind_tools.return_value = bound
        fundamentals.create_fundamentals_analyst(llm)({
            "company_of_interest": "RELIANCE.NS",
            "trade_date": TRADE_DATE,
            "fact_sheet": FACT_SHEET,
            "messages": [],
        })
        rendered = _prompt_text(captured["messages"])
        assert "', '" not in rendered
        assert FACT_SHEET in rendered
