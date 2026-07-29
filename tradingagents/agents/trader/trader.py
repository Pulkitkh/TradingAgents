"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    EVIDENCE_RULES,
    get_fact_sheet_from_state,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        fact_sheet = get_fact_sheet_from_state(state)
        trade_date = state.get("trade_date", "the analysis date")
        investment_plan = state["investment_plan"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    "Entry, stop-loss, and sizing must be taken from the computed risk "
                    "levels in the fact sheet, quoted exactly — never a rounded or "
                    "invented level. If no risk levels were computed, leave those "
                    "fields empty rather than estimating them. "
                    + NO_EXTERNAL_TOOLS
                    + EVIDENCE_RULES
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}, as of {trade_date}. {instrument_context} "
                    f"This plan incorporates insights from current technical market trends, "
                    f"macroeconomic indicators, and sentiment.\n\n"
                    f"{fact_sheet}\n\n"
                    f"Proposed Investment Plan: {investment_plan}\n\n"
                    f"Leverage these insights to make an informed and strategic decision. "
                    f"Set your confidence honestly: lower it when the fundamental and "
                    f"technical evidence disagree, or when the levels you need were "
                    f"not computed."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
