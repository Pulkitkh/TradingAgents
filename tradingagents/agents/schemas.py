"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    if isinstance(value, str) and value.strip().lower() in _NULLISH_FLOAT:
        return None
    return value


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Evidence provenance — separating facts from opinions
# ---------------------------------------------------------------------------


class Provenance(str, Enum):
    """Where a claim's supporting figure came from.

    The pipeline's most damaging observed failure was not a wrong number but an
    *unlabelled* one: a single-quarter growth figure from a news headline was
    restated downstream as the company's growth rate and became the basis of a
    valuation argument. Forcing every load-bearing claim to declare its
    provenance makes that substitution visible in the output instead of
    invisible in the reasoning.
    """

    VERIFIED = "Verified"     # appears in the deterministic fact sheet
    REPORTED = "Reported"     # from news, filings coverage, or social sources
    INFERRED = "Inferred"     # the agent's own estimate or judgement


class Claim(BaseModel):
    """One load-bearing statement plus the provenance of its evidence."""

    statement: str = Field(
        description=(
            "A single specific claim, stated in one sentence. Include the figure "
            "it rests on where there is one."
        ),
    )
    provenance: Provenance = Field(
        description=(
            "Verified = the figure appears in the verified fact sheet. "
            "Reported = it comes from a news article, filing summary, or social "
            "post and must be attributed. Inferred = it is your own estimate, "
            "projection, or judgement."
        ),
    )
    source: str | None = Field(
        default=None,
        description=(
            "For Verified: which fact sheet line. For Reported: the source and "
            "the period the figure covers, e.g. 'earnings coverage, Q1 FY2027 "
            "revenue'. For Inferred: the reasoning in a few words."
        ),
    )

    def rendered(self) -> str:
        tag = f"`{self.provenance.value.upper()}`"
        suffix = f" — _{self.source}_" if self.source else ""
        return f"- {tag} {self.statement}{suffix}"


class ScenarioProbabilities(BaseModel):
    """A calibrated distribution over outcomes, replacing a bare rating.

    A single label ("Overweight") hides how much of the decision rests on
    conviction and how much on hope. Institutional risk desks size on
    distributions, so the model is asked for one directly and the numbers are
    normalised here rather than trusted to sum correctly.
    """

    horizon_days: int = Field(
        default=21,
        ge=1,
        le=365,
        description="Trading-day horizon the probabilities apply to.",
    )
    prob_upside: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Probability the instrument outperforms its benchmark by more than "
            "the flat band over the horizon. A number between 0 and 1."
        ),
    )
    prob_flat: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Probability of a range-bound outcome within roughly ±1 ATR of the "
            "reference close. A number between 0 and 1."
        ),
    )
    prob_downside: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Probability the instrument underperforms its benchmark by more than "
            "the flat band over the horizon. A number between 0 and 1."
        ),
    )

    @model_validator(mode="after")
    def _normalise(self):
        total = self.prob_upside + self.prob_flat + self.prob_downside
        # Models routinely emit 0.45/0.35/0.25. Rejecting that would discard the
        # whole structured response and fall back to free text, losing far more
        # than the rounding error costs — so renormalise instead.
        if total <= 0:
            object.__setattr__(self, "prob_upside", 0.0)
            object.__setattr__(self, "prob_flat", 1.0)
            object.__setattr__(self, "prob_downside", 0.0)
            return self
        if abs(total - 1.0) > 1e-6:
            object.__setattr__(self, "prob_upside", self.prob_upside / total)
            object.__setattr__(self, "prob_flat", self.prob_flat / total)
            object.__setattr__(self, "prob_downside", self.prob_downside / total)
        return self

    @property
    def expected_direction(self) -> str:
        best = max(
            (self.prob_upside, "Upside"),
            (self.prob_flat, "Range-bound"),
            (self.prob_downside, "Downside"),
        )
        return best[1]

    @property
    def edge(self) -> float:
        """Directional edge: upside probability minus downside probability."""
        return self.prob_upside - self.prob_downside

    def rendered(self) -> str:
        return "\n".join([
            f"**Outcome distribution ({self.horizon_days} trading days)**",
            "",
            "| Scenario | Probability |",
            "|---|---:|",
            f"| Upside (outperforms benchmark) | {self.prob_upside:.0%} |",
            f"| Range-bound | {self.prob_flat:.0%} |",
            f"| Downside (underperforms benchmark) | {self.prob_downside:.0%} |",
            "",
            f"Directional edge: {self.edge:+.0%} · modal outcome: {self.expected_direction}",
        ])


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=(
            "Confidence in this proposal. Lower it when the fundamental and "
            "technical evidence point in opposite directions, or when the levels "
            "you need were not available in the fact sheet."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Entry price in the instrument's quote currency. Use the reference "
            "close or a computed level from the fact sheet's risk block; do not "
            "invent a round number."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Stop-loss price. Must be the computed volatility-scaled stop from "
            "the fact sheet when one is available. Leave empty rather than "
            "estimating one."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description=(
            "Sizing guidance. Use the computed position size from the fact "
            "sheet's risk block when present, quoting its risk budget."
        ),
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Confidence**: {proposal.confidence.capitalize()}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate. This must "
            "be consistent with the probability distribution you provide: do not "
            "rate Buy or Overweight while assigning downside the highest "
            "probability."
        ),
    )
    probabilities: ScenarioProbabilities = Field(
        description=(
            "Your calibrated probability distribution over outcomes at the stated "
            "horizon. Be honest rather than decisive: a 45/35/20 split is a real "
            "answer and is more useful than a false 80/10/10."
        ),
    )
    conviction: Literal["low", "medium", "high"] = Field(
        description=(
            "Conviction in this call. Use 'low' when the fact sheet had material "
            "gaps, when analysts disagreed on facts rather than interpretation, "
            "or when the thesis rests mainly on Reported rather than Verified "
            "figures. 'high' requires Verified evidence on both the fundamental "
            "and the technical side."
        ),
    )
    key_facts: list[Claim] = Field(
        default_factory=list,
        description=(
            "The three to six load-bearing claims behind this decision, each "
            "tagged with its provenance. At least one must be Verified. If the "
            "thesis rests mainly on Reported figures, say so here and lower your "
            "conviction accordingly."
        ),
    )
    invalidation_triggers: list[str] = Field(
        default_factory=list,
        description=(
            "Specific, observable events that would falsify this thesis — a price "
            "level breaking, a metric missing a threshold, a catalyst failing to "
            "land. Each must be checkable without further interpretation. Vague "
            "triggers like 'if sentiment worsens' are not acceptable."
        ),
    )
    data_gaps: list[str] = Field(
        default_factory=list,
        description=(
            "What you could not verify this run — unavailable sources, missing "
            "sessions, absent peer data. State them plainly; an empty list "
            "asserts that nothing material was missing."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences. Every "
            "price level must come from the computed risk levels in the fact "
            "sheet — never invent or round one."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description=(
            "Optional target price in the instrument's quote currency. Use the "
            "computed target from the fact sheet's risk levels when one is given."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Stop-loss price. Must equal the computed stop in the fact sheet's "
            "risk levels when one is available; leave empty if none was computed."
        ),
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )

    @field_validator("price_target", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @property
    def rating_probability_conflict(self) -> str | None:
        """Describe any contradiction between the rating and the distribution.

        A bullish label sitting on a bearish distribution is the exact shape of
        the failure this schema exists to catch, and it is worth surfacing in the
        report rather than silently rendering both.
        """
        bullish = {PortfolioRating.BUY, PortfolioRating.OVERWEIGHT}
        bearish = {PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT}
        edge = self.probabilities.edge
        if self.rating in bullish and edge < 0:
            return (
                f"Rating is {self.rating.value} but the distribution favours "
                f"downside ({edge:+.0%} edge). Treat this call as unresolved."
            )
        if self.rating in bearish and edge > 0:
            return (
                f"Rating is {self.rating.value} but the distribution favours "
                f"upside ({edge:+.0%} edge). Treat this call as unresolved."
            )
        return None


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Conviction**: {decision.conviction.capitalize()}",
        "",
        decision.probabilities.rendered(),
    ]

    conflict = decision.rating_probability_conflict
    if conflict:
        parts.extend(["", f"> **RATING / PROBABILITY CONFLICT.** {conflict}"])

    parts.extend([
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ])

    if decision.key_facts:
        verified = sum(1 for c in decision.key_facts if c.provenance is Provenance.VERIFIED)
        reported = sum(1 for c in decision.key_facts if c.provenance is Provenance.REPORTED)
        inferred = sum(1 for c in decision.key_facts if c.provenance is Provenance.INFERRED)
        parts.extend([
            "",
            "**Evidence base** "
            f"({verified} verified · {reported} reported · {inferred} inferred)",
            "",
            *[claim.rendered() for claim in decision.key_facts],
        ])
        if verified == 0:
            parts.extend([
                "",
                "> **NO VERIFIED EVIDENCE.** Every load-bearing claim is reported "
                "or inferred. This decision is not grounded in measured data.",
            ])

    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {decision.stop_loss}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])

    if decision.invalidation_triggers:
        parts.extend([
            "",
            "**Invalidation triggers**",
            "",
            *[f"- {trigger}" for trigger in decision.invalidation_triggers],
        ])

    if decision.data_gaps:
        parts.extend([
            "",
            "**Data gaps this run**",
            "",
            *[f"- {gap}" for gap in decision.data_gaps],
        ])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    sources_with_data: list[str] = Field(
        default_factory=list,
        description=(
            "Which of 'news', 'stocktwits', 'reddit' actually returned usable "
            "content. List only sources with real data — a placeholder or an "
            "empty result does not count. This drives whether the sentiment read "
            "carries any weight downstream, so it must be accurate."
        ),
    )

    @property
    def is_independent_signal(self) -> bool:
        """Whether this read adds information beyond what the news analyst saw.

        Observed on Indian equities: StockTwits indexes US tickers and Reddit's
        investing subs barely discuss NSE names, so both return empty for a
        ticker like ``RELIANCE.NS``. The analyst then re-read the news analyst's
        own data and emitted a directional band — which entered the debate as a
        second independent opinion when it was the same input counted twice.
        A sentiment read sourced only from news is not an independent signal.
        """
        social = {"stocktwits", "reddit"}
        return bool(social & {s.strip().lower() for s in self.sources_with_data})
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    sources = ", ".join(report.sources_with_data) if report.sources_with_data else "none"
    lines = [
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        f"**Sources with data:** {sources}",
    ]

    if not report.is_independent_signal:
        lines += [
            "",
            "> **NOT AN INDEPENDENT SIGNAL.** No social source returned usable "
            "data, so this read derives from the same news the News Analyst "
            "already covered. Downstream agents must not treat it as a second "
            "confirming opinion, and must not let it tilt the decision "
            "directionally.",
        ]

    lines += ["", report.narrative]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Valuation Analyst
# ---------------------------------------------------------------------------


class ValuationStance(str, Enum):
    """Where the subject trades relative to its peer group."""

    DEEP_DISCOUNT = "Deep Discount"
    DISCOUNT = "Discount"
    IN_LINE = "In Line"
    PREMIUM = "Premium"
    RICH_PREMIUM = "Rich Premium"


class ValuationAssessment(BaseModel):
    """Structured relative-valuation read built on the computed peer table.

    The multiples and peer medians are computed deterministically in
    ``dataflows/peers.py`` and handed to the agent finished. The agent's job is
    interpretation only — whether a discount is deserved — which is exactly the
    boundary between measurement and judgement this pipeline is built to keep
    visible.
    """

    stance: ValuationStance = Field(
        description=(
            "Where the subject trades against the peer median on the multiples "
            "in the computed peer table. Exactly one of Deep Discount / Discount "
            "/ In Line / Premium / Rich Premium."
        ),
    )
    discount_is_justified: bool = Field(
        description=(
            "Whether the discount or premium is warranted by fundamentals "
            "(growth, margins, leverage, returns) rather than being a "
            "mispricing. This is the whole question — a cheap stock that "
            "deserves to be cheap is not an opportunity."
        ),
    )
    justification: str = Field(
        description=(
            "Why the discount or premium is or is not warranted, citing specific "
            "rows from the computed peer table and the verified fundamentals. "
            "Name the metrics you relied on."
        ),
    )
    peer_context: str = Field(
        description=(
            "How the subject compares on each material multiple, and which peers "
            "are the closest true comparables versus which are in the group only "
            "by sector classification."
        ),
    )
    key_multiples: list[Claim] = Field(
        default_factory=list,
        description=(
            "The specific multiples driving your stance, each tagged with "
            "provenance. Figures from the computed peer table are Verified."
        ),
    )
    caveats: str | None = Field(
        default=None,
        description=(
            "Where this comparison breaks down — conglomerates whose segments "
            "warrant different multiples, peers with distorted trailing "
            "earnings, missing data in the table."
        ),
    )


def render_valuation_assessment(assessment: ValuationAssessment) -> str:
    """Render a ValuationAssessment to the markdown the report tree consumes."""
    verdict = "justified by fundamentals" if assessment.discount_is_justified else (
        "NOT justified by fundamentals — potential mispricing"
    )
    parts = [
        f"**Relative Valuation:** **{assessment.stance.value}**",
        f"**Verdict:** {verdict}",
        "",
        f"**Justification**: {assessment.justification}",
        "",
        f"**Peer Context**: {assessment.peer_context}",
    ]
    if assessment.key_multiples:
        parts.extend([
            "",
            "**Key multiples**",
            "",
            *[claim.rendered() for claim in assessment.key_multiples],
        ])
    if assessment.caveats:
        parts.extend(["", f"**Caveats**: {assessment.caveats}"])
    return "\n".join(parts)
