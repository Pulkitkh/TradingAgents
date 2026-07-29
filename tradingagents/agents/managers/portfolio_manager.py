"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
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


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        fact_sheet = get_fact_sheet_from_state(state)
        # Downstream agents historically received no date at all, so a model
        # asked to write a dated memo invented one — observed producing an
        # "October 2023" header on a July 2026 run.
        trade_date = state.get("trade_date", "the analysis date")

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        valuation_report = state.get("valuation_report", "")

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )
        valuation_line = (
            f"\n**Relative Valuation (peer-anchored):**\n{valuation_report}\n"
            if valuation_report
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

The analysis date is **{trade_date}**. Use this date in any memo header or time reference; do not infer a date from the content of the reports.

{instrument_context}

---

{fact_sheet}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}{valuation_line}
**Risk Analysts Debate History:**
{history}

---

## Decision requirements

**Price levels are computed, not chosen.** Every stop-loss, target, and position
size must come from the computed risk levels in the fact sheet above, quoted
exactly. Do not round a level to a psychological number, do not place a stop
"just below" a band you saw elsewhere, and do not state a level the fact sheet
does not contain. If risk levels were unavailable, say so and issue no levels —
an unpriced recommendation is honest; an invented stop is not.

**Your rating and your probabilities must agree.** You are giving a distribution
over outcomes, not just a label. If you assign downside the highest probability,
you cannot rate Buy or Overweight. A 45/35/20 split is a real answer; a false
80/10/10 is worse than useless to a desk sizing on it.

**Discount evidence that isn't independent.** If the sentiment read is flagged
as not an independent signal, it is the same news the News Analyst already
covered — it must not count as a second confirming opinion. If a figure appears
in the reports but not in the fact sheet, it is reported, not verified, and
cannot be the sole support for the decision. Where a fundamental and a technical
view genuinely conflict, resolving it by asserting one side is stronger is not a
resolution — lower your conviction and say which observation would settle it.

**State what you could not check.** Fill `data_gaps` with anything unavailable
this run. An empty list is a claim that nothing material was missing.

Be decisive where the evidence supports it and explicitly uncertain where it
does not.

{NO_EXTERNAL_TOOLS}{EVIDENCE_RULES}{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
