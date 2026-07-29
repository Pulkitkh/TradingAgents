"""The CLI and the programmatic API must set up and tear down runs identically.

The CLI streams ``graph.graph`` directly instead of calling ``propagate()``, so
anything done only inside the propagate path silently does not happen for CLI
users — which is how nearly every run is started. This has bitten the project
repeatedly:

* Instrument identity (#814) had to be duplicated onto the CLI path.
* The verified fact sheet was added to ``propagate()`` only, so a live CLI run
  produced reports headed "No verified fact sheet was available" while the
  deterministic evidence layer sat unused.
* The decision log was never written on the CLI path at all, so the reflection
  loop the framework documents had never actually run for CLI users.

``prepare_run`` and ``finalize_run`` exist to make that impossible. These tests
assert both entry points go through them, and that they carry everything.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph

CLI_SOURCE = Path(__file__).resolve().parents[1] / "cli" / "main.py"


@pytest.fixture()
def graph(monkeypatch):
    """A TradingAgentsGraph with LLM construction and graph setup stubbed out."""
    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.create_llm_client",
        lambda **kwargs: MagicMock(get_llm=lambda: MagicMock()),
    )
    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.GraphSetup.setup_graph",
        lambda self, analysts: MagicMock(compile=lambda **kw: MagicMock()),
    )
    instance = TradingAgentsGraph(config={
        "llm_provider": "openai",
        "deep_think_llm": "m",
        "quick_think_llm": "m",
        "backend_url": None,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "data_cache_dir": "/tmp/ta-test-cache",
        "results_dir": "/tmp/ta-test-results",
        "memory_log_path": None,
        "fact_sheet_enabled": True,
    })
    return instance


@pytest.mark.unit
class TestPrepareRun:
    def test_state_carries_every_setup_input(self, graph):
        with patch.object(graph, "build_fact_sheet", return_value="FACT SHEET"), \
             patch.object(graph, "resolve_instrument_context", return_value="IDENTITY"), \
             patch.object(graph.memory_log, "get_past_context", return_value="LESSONS"), \
             patch.object(graph, "_resolve_pending_entries"):
            state = graph.prepare_run("RELIANCE.NS", "2026-07-29")

        assert state["fact_sheet"] == "FACT SHEET"
        assert state["instrument_context"] == "IDENTITY"
        assert state["past_context"] == "LESSONS"
        assert state["company_of_interest"] == "RELIANCE.NS"
        assert state["trade_date"] == "2026-07-29"

    def test_pending_outcomes_resolve_before_context_is_read(self, graph):
        """Reflections on prior trades must be scored before this run reads them."""
        order = []
        with patch.object(graph, "build_fact_sheet", return_value=""), \
             patch.object(graph, "resolve_instrument_context", return_value=""), \
             patch.object(
                 graph.memory_log, "get_past_context",
                 side_effect=lambda t: order.append("read") or "",
             ), \
             patch.object(
                 graph, "_resolve_pending_entries",
                 side_effect=lambda t: order.append("resolve"),
             ):
            graph.prepare_run("RELIANCE.NS", "2026-07-29")

        assert order == ["resolve", "read"]

    def test_asset_type_reaches_the_fact_sheet(self, graph):
        with patch.object(graph, "build_fact_sheet", return_value="") as builder, \
             patch.object(graph, "resolve_instrument_context", return_value=""), \
             patch.object(graph.memory_log, "get_past_context", return_value=""), \
             patch.object(graph, "_resolve_pending_entries"):
            graph.prepare_run("BTC-USD", "2026-07-29", "crypto")

        builder.assert_called_once_with("BTC-USD", "2026-07-29", "crypto")

    def test_disabled_fact_sheet_yields_empty_string(self, graph):
        graph.config["fact_sheet_enabled"] = False
        assert graph.build_fact_sheet("RELIANCE.NS", "2026-07-29") == ""

    def test_fact_sheet_failure_never_blocks_a_run(self, graph):
        with patch(
            "tradingagents.graph.trading_graph.build_fact_sheet",
            side_effect=RuntimeError("vendor down"),
        ):
            assert graph.build_fact_sheet("RELIANCE.NS", "2026-07-29") == ""


@pytest.mark.unit
class TestFinalizeRun:
    def _state(self):
        return {
            "final_trade_decision": "**Rating**: Hold",
            "company_of_interest": "RELIANCE.NS",
            "trade_date": "2026-07-29",
            "market_report": "m", "sentiment_report": "s",
            "news_report": "n", "fundamentals_report": "f",
            "investment_debate_state": {
                "bull_history": "", "bear_history": "", "history": "",
                "current_response": "", "judge_decision": "",
            },
            "trader_investment_plan": "t",
            "risk_debate_state": {
                "aggressive_history": "", "conservative_history": "",
                "neutral_history": "", "history": "", "judge_decision": "",
            },
            "investment_plan": "p",
        }

    def test_decision_is_written_to_the_memory_log(self, graph):
        with patch.object(graph, "_log_state"), \
             patch.object(graph.memory_log, "store_decision") as store:
            graph.finalize_run(self._state(), "RELIANCE.NS", "2026-07-29")

        store.assert_called_once()
        assert store.call_args.kwargs["ticker"] == "RELIANCE.NS"
        assert store.call_args.kwargs["trade_date"] == "2026-07-29"

    def test_checkpoint_is_cleared_only_when_enabled(self, graph):
        with patch.object(graph, "_log_state"), \
             patch.object(graph.memory_log, "store_decision"), \
             patch("tradingagents.graph.trading_graph.clear_checkpoint") as clear:
            graph.finalize_run(self._state(), "RELIANCE.NS", "2026-07-29")
            clear.assert_not_called()

            graph.config["checkpoint_enabled"] = True
            graph.finalize_run(self._state(), "RELIANCE.NS", "2026-07-29")
            clear.assert_called_once()


@pytest.mark.unit
class TestCliUsesTheSharedPath:
    """Static checks: the CLI must not rebuild run setup by hand."""

    def _cli_source(self) -> str:
        return CLI_SOURCE.read_text(encoding="utf-8")

    def test_cli_calls_prepare_run(self):
        assert "graph.prepare_run(" in self._cli_source()

    def test_cli_calls_finalize_run(self):
        assert "graph.finalize_run(" in self._cli_source()

    def test_cli_does_not_build_initial_state_directly(self):
        """Constructing state by hand is how each of these bugs was introduced."""
        assert "create_initial_state(" not in self._cli_source()

    def test_cli_restores_the_fact_sheet_onto_the_final_state(self):
        # No node emits fact_sheet, so the merged stream would otherwise drop it
        # and the saved report tree would lose the audit trail.
        assert 'final_state.setdefault("fact_sheet"' in self._cli_source()

    def test_prepare_run_is_the_only_place_state_is_built(self):
        """Exactly one call site for create_initial_state across the package."""
        root = Path(__file__).resolve().parents[1]
        call_sites = []
        for path in list(root.glob("tradingagents/**/*.py")) + list(root.glob("cli/**/*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "create_initial_state"
                ):
                    call_sites.append(f"{path.name}:{node.lineno}")
        assert len(call_sites) == 1, f"expected one call site, found {call_sites}"

    def test_propagate_delegates_to_the_shared_helpers(self):
        source = inspect.getsource(TradingAgentsGraph._run_graph)
        assert "self.prepare_run(" in source
        assert "self.finalize_run(" in source
