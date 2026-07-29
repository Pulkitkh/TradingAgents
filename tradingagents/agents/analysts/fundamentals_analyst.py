from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    EVIDENCE_RULES,
    get_balance_sheet,
    get_cashflow,
    get_fact_sheet_from_state,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        fact_sheet = get_fact_sheet_from_state(state)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        # NB: the trailing comma this expression used to carry made it a 1-tuple,
        # so the rendered prompt contained a Python tuple repr instead of the
        # instructions. Keep it a plain string.
        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + " Growth rates have already been computed for you on three separate bases"
            + " (latest quarter year-on-year, trailing twelve months year-on-year, and"
            + " last full fiscal year) in the fact sheet below. Always name the basis when"
            + " you cite a growth figure. Never describe a single quarter's growth as"
            + " 'the company's revenue growth', and never carry a quarterly figure into a"
            + " valuation argument such as PEG that assumes a sustained rate. If the fact"
            + " sheet flags a growth-basis conflict, address it explicitly rather than"
            + " picking whichever number better fits a narrative."
            + "\n\n"
            + fact_sheet
            + EVIDENCE_RULES
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " You are an analyst, not a decision maker: report what your"
                    " domain shows and stop there. Do NOT issue a buy, sell, or hold"
                    " recommendation, and do not emit a transaction proposal — the"
                    " Portfolio Manager owns the decision after the full debate, and"
                    " a directional call here contradicts the final memo in the saved"
                    " report."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
