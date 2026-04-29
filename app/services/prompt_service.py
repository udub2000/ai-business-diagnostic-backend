
from app.core.config import settings
from app.schemas.assessment import ScoreRequest, ScoreResponse


SYSTEM_PROMPT = """
You are an expert small business operations consultant specializing in service-based businesses.

The diagnosis has already been computed.
Do not re-score the business.
Do not replace the primary constraint.
Do not invent unsupported problems.

Your job is to turn the provided diagnosis, evidence, contradictions, assumptions, confidence, and approved recommendation pool into a polished client-facing consulting report.

Style rules:
- Write like a strong consultant, not a chatbot
- Be direct, clear, and specific
- Avoid filler and vague language
- Tie observations to business impact
- Focus on the single primary constraint
- Mention the secondary constraint only if it genuinely matters
- Use clean markdown with headers, bold labels, and bullet points
- Do not use tables
- Do not use the "|" pipe character as a separator
- Do not output JSON

Action-plan rules:
- Give exactly 3 to 5 recommendations
- For each recommendation, use this exact structure:
  **Action Title**
  - What to do: one complete sentence.
  - Why it matters: one complete sentence.
  - Expected impact: one complete sentence.
- Keep each line complete and readable
- Do not leave partial phrases such as "You"
- Do not repeat the same recommendation in different wording

Evidence rules:
- Reference the supplied scores and evidence where useful
- If contradictions exist, surface them clearly
- If assumptions exist, explain how they affect confidence
- If confidence is moderate or low, say why in plain language

Output structure:
# Business Snapshot
- 4 to 6 bullets

# Primary Constraint
- One short, decisive paragraph

# Why This Was Identified
- 3 to 5 bullets
- Each bullet should reference a score, input, contradiction, or pattern and explain why it matters

# What This Means Operationally
- One short paragraph

# Risks of Inaction
- 2 to 4 bullets

# Recommended Actions
- Use the required action-plan format exactly

# Expected Impact
- 3 to 5 bullets focused on revenue, margin, throughput, utilization, customer experience, or owner leverage

# Assumptions and Confidence
- Briefly explain assumptions
- Briefly explain confidence level
- If contradictions exist, mention them here as well

Goal:
The report should feel like something a consultant could review live with a client.
""".strip()


def build_evidence_block(payload: dict) -> str:
    parts = []
    for key, value in payload.items():
        if value is not None and value != "":
            parts.append(f"- {key}: {value}")
    return "\n".join(parts) if parts else "- No detailed evidence supplied"


def build_confidence_explanation(score: ScoreResponse) -> str:
    explanations: list[str] = []

    if score.assumptions:
        explanations.append(
            f"{len(score.assumptions)} assumption(s) were needed because some inputs were estimated or missing"
        )

    if score.contradictions:
        explanations.append(
            f"{len(score.contradictions)} contradiction(s) reduced confidence in the internal consistency of the intake"
        )

    if not explanations:
        return "Confidence is supported by relatively complete and internally consistent inputs."

    return "Confidence is influenced by " + "; ".join(explanations) + "."


def build_report_prompt(request: ScoreRequest, score: ScoreResponse) -> tuple[str, str]:
    evidence_block = build_evidence_block(request.intake.model_dump())
    contradictions_block = (
        "\n".join("- " + c for c in score.contradictions)
        if score.contradictions
        else "- None"
    )
    assumptions_block = (
        "\n".join("- " + a for a in score.assumptions)
        if score.assumptions
        else "- None"
    )
    recommendation_block = (
        "\n".join("- " + a for a in score.recommended_action_pool)
        if score.recommended_action_pool
        else "- None"
    )

    user_prompt = f"""
## BUSINESS DATA
Business Name: {request.business.name}
Business Type: {request.business.business_type}
Location: {request.business.location or 'Not provided'}

## SCORES (0-100)
- Demand: {score.scores['Demand']}
- Conversion: {score.scores['Conversion']}
- Capacity: {score.scores['Capacity']}
- Operations: {score.scores['Operations']}
- Financials: {score.scores['Financials']}

## CATEGORY CONFIDENCE
- Demand: {score.category_confidence['Demand']}
- Conversion: {score.category_confidence['Conversion']}
- Capacity: {score.category_confidence['Capacity']}
- Operations: {score.category_confidence['Operations']}
- Financials: {score.category_confidence['Financials']}

## PRIMARY CONSTRAINT
{score.primary_constraint}

## SECONDARY CONSTRAINT
{score.secondary_constraint or 'None'}

## CONSTRAINT PATTERN
{score.constraint_pattern}

## OVERALL CONFIDENCE
{score.overall_confidence_label} ({score.overall_confidence_score}/100)

## CONFIDENCE EXPLANATION
{build_confidence_explanation(score)}

## SUPPORTING EVIDENCE
{evidence_block}

## CONTRADICTIONS
{contradictions_block}

## ASSUMPTIONS
{assumptions_block}

## APPROVED RECOMMENDATION POOL
{recommendation_block}

## INSTRUCTIONS
Use the approved recommendation pool as the basis for the Recommended Actions section.
You may sharpen the wording, sequence, and business rationale, but do not invent unrelated recommendations.
The report must stay consistent with the provided diagnosis and evidence.
""".strip()

    return SYSTEM_PROMPT, user_prompt
