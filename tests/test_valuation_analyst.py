"""Valuation analyst node, and its wiring into the graph as a tool-free analyst."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import tradingagents.agents.analysts.valuation_analyst as va
from tradingagents.agents.schemas import ValuationAssessment, ValuationStance
from tradingagents.graph.analyst_execution import (
    ANALYST_NODE_SPECS,
    build_analyst_execution_plan,
)
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup


@pytest.fixture()
def stub_evidence(monkeypatch):
    monkeypatch.setattr(va, "build_peer_comparison", lambda *a, **k: "PEER TABLE")
    monkeypatch.setattr(
        va, "compute_fundamental_metrics",
        lambda *a, **k: MagicMock(as_markdown=lambda: "FUNDAMENTALS BLOCK"),
    )


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


def _state(**overrides):
    base = {
        "company_of_interest": "RELIANCE.NS",
        "trade_date": "2026-07-27",
        "asset_type": "stock",
        "messages": [],
    }
    base.update(overrides)
    return base


ASSESSMENT = ValuationAssessment(
    stance=ValuationStance.PREMIUM,
    discount_is_justified=True,
    justification="Premium is earned by consumer-segment growth.",
    peer_context="Trades above the refiner median on P/E and P/B.",
)


@pytest.mark.unit
class TestValuationAnalystNode:
    def test_writes_the_valuation_report_key(self, stub_evidence):
        llm = _capturing_llm({}, ASSESSMENT)
        result = va.create_valuation_analyst(llm)(_state())
        assert "valuation_report" in result
        assert "**Relative Valuation:** **Premium**" in result["valuation_report"]

    def test_computed_evidence_reaches_the_prompt(self, stub_evidence):
        captured = {}
        va.create_valuation_analyst(_capturing_llm(captured, ASSESSMENT))(_state())
        text = _prompt_text(captured["prompt"])
        assert "PEER TABLE" in text
        assert "FUNDAMENTALS BLOCK" in text

    def test_prompt_states_the_agent_produces_no_numbers(self, stub_evidence):
        captured = {}
        va.create_valuation_analyst(_capturing_llm(captured, ASSESSMENT))(_state())
        text = _prompt_text(captured["prompt"])
        assert "You are not being asked to produce numbers" in text
        assert "is the discount or premium deserved" in text

    def test_prompt_guards_peg_against_the_growth_basis_conflict(self, stub_evidence):
        captured = {}
        va.create_valuation_analyst(_capturing_llm(captured, ASSESSMENT))(_state())
        text = _prompt_text(captured["prompt"])
        assert "Interrogate PEG" in text
        assert "quarterly and trailing growth" in text

    def test_prompt_forbids_supplying_missing_multiples(self, stub_evidence):
        captured = {}
        va.create_valuation_analyst(_capturing_llm(captured, ASSESSMENT))(_state())
        assert "Never supply a missing number" in _prompt_text(captured["prompt"])

    def test_analysis_date_reaches_the_prompt(self, stub_evidence):
        captured = {}
        va.create_valuation_analyst(_capturing_llm(captured, ASSESSMENT))(_state())
        assert "2026-07-27" in _prompt_text(captured["prompt"])

    def test_dead_peer_lookup_degrades_instead_of_raising(self, monkeypatch):
        def explode(*a, **k):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(va, "build_peer_comparison", explode)
        monkeypatch.setattr(
            va, "compute_fundamental_metrics",
            lambda *a, **k: MagicMock(as_markdown=lambda: "FUNDAMENTALS BLOCK"),
        )
        captured = {}
        result = va.create_valuation_analyst(_capturing_llm(captured, ASSESSMENT))(_state())
        assert result["valuation_report"]
        assert "Do not assert a relative valuation" in _prompt_text(captured["prompt"])


@pytest.mark.unit
class TestGraphWiring:
    def test_valuation_spec_is_tool_free(self):
        assert ANALYST_NODE_SPECS["valuation"].tool_free is True
        assert ANALYST_NODE_SPECS["market"].tool_free is False

    def test_plan_accepts_valuation(self):
        plan = build_analyst_execution_plan(["market", "valuation"])
        assert [spec.key for spec in plan.specs] == ["market", "valuation"]

    def _compiled(self, analysts):
        llm = MagicMock()
        tool_nodes = {
            key: MagicMock()
            for key in ("market", "social", "news", "fundamentals")
        }
        setup = GraphSetup(llm, llm, tool_nodes, ConditionalLogic())
        return setup.setup_graph(analysts).compile().get_graph()

    def test_no_dead_tool_node_is_created(self):
        nodes = set(self._compiled(("market", "valuation")).nodes)
        assert "Valuation Analyst" in nodes
        assert "Msg Clear Valuation" in nodes
        assert "tools_valuation" not in nodes
        # The tool-using analyst still gets one.
        assert "tools_market" in nodes

    def test_valuation_only_graph_compiles(self):
        nodes = set(self._compiled(("valuation",)).nodes)
        assert "Valuation Analyst" in nodes
        assert "Bull Researcher" in nodes

    def test_full_analyst_team_compiles(self):
        nodes = set(
            self._compiled(
                ("market", "social", "news", "fundamentals", "valuation")
            ).nodes
        )
        assert "Portfolio Manager" in nodes
        assert "tools_valuation" not in nodes


@pytest.mark.unit
class TestCliRegistration:
    def test_valuation_is_selectable_for_stocks(self):
        from cli.models import AnalystType, AssetType
        from cli.utils import ANALYST_ORDER, filter_analysts_for_asset_type

        values = [value for _, value in ANALYST_ORDER]
        assert AnalystType.VALUATION in values
        assert AnalystType.VALUATION in filter_analysts_for_asset_type(
            values, AssetType.STOCK
        )

    def test_valuation_is_hidden_for_crypto(self):
        from cli.models import AnalystType, AssetType
        from cli.utils import ANALYST_ORDER, filter_analysts_for_asset_type

        available = filter_analysts_for_asset_type(
            [value for _, value in ANALYST_ORDER], AssetType.CRYPTO
        )
        assert AnalystType.VALUATION not in available
        assert AnalystType.FUNDAMENTALS not in available

    def test_report_section_and_status_maps_agree(self):
        from cli.main import ANALYST_AGENT_NAMES, ANALYST_ORDER, ANALYST_REPORT_MAP, MessageBuffer

        assert "valuation" in ANALYST_ORDER
        assert ANALYST_AGENT_NAMES["valuation"] == "Valuation Analyst"
        assert ANALYST_REPORT_MAP["valuation"] == "valuation_report"
        assert MessageBuffer.REPORT_SECTIONS["valuation_report"] == (
            "valuation", "Valuation Analyst"
        )
