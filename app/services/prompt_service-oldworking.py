from app.core.config import settings
from app.schemas.assessment import ScoreRequest, ScoreResponse


SYSTEM_PROMPT = """
You are an expert small business operations consultant specializing in service-based businesses.
The diagnosis is already provided. Do not re-score or replace it.
Use the provided scores, evidence, constraint pattern, recommendations, confidence, and assumptions to generate a detailed client-facing consulting report.
Focus on the single primary constraint. Mention the secondary constraint only if provided.
Explain the diagnosis, business meaning, risks of inaction, and why each action matters.
Avoid generic advice and do not introduce unsupported problems.
Return markdown.
""".strip()


def build_evidence_block(payload: dict) -> str:
    parts = []
    for key, value in payload.items():
        if value is not None and value != "":
            parts.append(f"- {key}: {value}")
    return "\n".join(parts) if parts else "- No detailed evidence supplied"



def build_report_prompt(request: ScoreRequest, score: ScoreResponse) -> tuple[str, str]:
    evidence_block = build_evidence_block(request.intake.model_dump())
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

## SUPPORTING EVIDENCE
{evidence_block}

## CONTRADICTIONS
{chr(10).join('- ' + c for c in score.contradictions) if score.contradictions else '- None'}

## ASSUMPTIONS
{chr(10).join('- ' + a for a in score.assumptions) if score.assumptions else '- None'}

## APPROVED RECOMMENDATION POOL
{chr(10).join('- ' + a for a in score.recommended_action_pool)}

## INSTRUCTIONS
Write a detailed client-facing diagnostic report with these sections:
1. Business Snapshot
2. Primary Constraint
3. Why the Model Identified This Constraint
4. What This Means Operationally
5. Risks of Inaction
6. Recommended Actions
7. Expected Impact
8. Assumptions and Confidence
""".strip()
    return SYSTEM_PROMPT, user_prompt
