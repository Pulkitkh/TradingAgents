"""Valuation analyst — relative valuation against a computed peer group.

The original analyst team could describe a company's financials but had no way
to answer the question a desk actually asks: is this cheap? A P/E of 23 means
nothing without the sector beside it, and Indian sector dispersion is wide
enough that an unanchored multiple invites exactly the free-floating assertion
this pipeline is trying to remove — FMCG names trade in the fifties while
refiners sit in the teens, so "reasonably valued" is unfalsifiable without a
comparison set.

Like the sentiment analyst, this agent pre-fetches its evidence rather than
tool-calling for it. The peer table and the fundamentals block are computed
deterministically in ``dataflows/`` and injected complete, so the model never
chooses a peer set, never fetches a multiple, and never has an opportunity to
produce a number. Its entire contribution is the judgement the numbers cannot
make on their own: whether a discount is deserved.

Pre-fetching also makes this the cheapest analyst in the graph — one LLM call,
no tool round-trips — which matters on rate-limited free-tier API keys.
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import ValuationAssessment, render_valuation_assessment
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.fundamentals_metrics import compute_fundamental_metrics
from tradingagents.dataflows.peers import build_peer_comparison


def create_valuation_analyst(llm):
    """Create a valuation analyst node backed by a computed peer comparison."""
    structured_llm = bind_structured(llm, ValuationAssessment, "Valuation Analyst")

    def valuation_analyst_node(state):
        ticker = state["company_of_interest"]
        curr_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        # Both blocks fail soft and always return a string, so the model sees
        # either real numbers or an explicit unavailable notice — never nothing.
        try:
            peer_block = build_peer_comparison(ticker, curr_date)
        except Exception as exc:  # noqa: BLE001 — degrade, never abort the graph
            peer_block = (
                f"### Peer valuation comparison\n\nUnavailable ({type(exc).__name__}). "
                "Do not assert a relative valuation."
            )
        try:
            fundamentals_block = compute_fundamental_metrics(ticker, curr_date).as_markdown()
        except Exception as exc:  # noqa: BLE001
            fundamentals_block = (
                f"### Verified fundamentals\n\nUnavailable ({type(exc).__name__}). "
                "Do not assert growth, margin, or leverage figures."
            )

        system_message = _build_system_message(
            ticker=ticker,
            curr_date=curr_date,
            peer_block=peer_block,
            fundamentals_block=fundamentals_block,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a valuation analyst on a buy-side desk, collaborating "
                    "with other analysts."
                    " Today's date is {current_date}; treat it as 'now' for all analysis."
                    " {instrument_context}"
                    " " + NO_EXTERNAL_TOOLS +
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=curr_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_valuation_assessment,
            "Valuation Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "valuation_report": report_text,
        }

    return valuation_analyst_node


def _build_system_message(
    *,
    ticker: str,
    curr_date: str,
    peer_block: str,
    fundamentals_block: str,
) -> str:
    """Assemble the valuation-analyst system message with computed evidence."""
    return f"""Your task is to judge whether {ticker} is mispriced relative to its peer group as of {curr_date}.

Every multiple, median, and premium/discount figure below was computed in code before you were called. You are not being asked to produce numbers — you are being asked to answer the one question the numbers cannot: **is the discount or premium deserved?**

{peer_block}

{fundamentals_block}

## How to reach a verdict

1. **Anchor on the peer median, not on absolutes.** A P/E of 23 is meaningless alone. What matters is where it sits against the peer median in the table above, and whether the gap is explained by the fundamentals block.

2. **Test whether the gap is earned.** A discount is only an opportunity if the subject's growth, margins, returns, and leverage are at least comparable to peers. A company trading 30% below the peer P/E with materially worse ROE and higher leverage is priced correctly, not cheaply. Say so plainly when that is the case — a rejected value thesis is a useful output.

3. **Interrogate PEG before relying on it.** PEG in the table comes from the vendor's growth estimate, not from a rate computed here. If the verified fundamentals show a conflict between quarterly and trailing growth, PEG inherits that ambiguity and must not be used as the primary support for a conclusion. Name the growth basis you are relying on.

4. **Name the true comparables.** Sector classification is coarse. For a conglomerate, segments may warrant different multiples than the group median implies, and some listed "peers" are peers only by classification. Identify which rows are genuinely comparable and which are not, and put that in the caveats.

5. **Never supply a missing number.** Where the table shows N/A, treat the comparison as unavailable on that metric. Do not substitute a remembered figure or a sector rule of thumb.

6. **Distinguish measurement from judgement.** Multiples from the table are Verified. Your read on whether a gap is deserved is Inferred. Tag the claims in `key_multiples` accordingly.

## Output fields

- **stance**: Deep Discount / Discount / In Line / Premium / Rich Premium versus the peer median.
- **discount_is_justified**: true if the gap is warranted by fundamentals, false if it looks like a mispricing.
- **justification**: why, citing specific rows from the computed tables.
- **peer_context**: metric-by-metric comparison, and which peers are true comparables.
- **key_multiples**: the multiples driving your stance, each tagged with provenance.
- **caveats**: where this comparison breaks down.

{get_language_instruction()}"""
