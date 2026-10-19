"""Staging-only integrated assessment: questionnaire + period-aware financial evidence.

Financials are attached as provenance-bearing context, never promoted into a
constraint or recommendation without approved metric-to-condition logic.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .staging_api import AssessmentInput, internal_review_draft
from .financial_evidence_bridge import FinancialEvidenceInput, review_financial_evidence
from .financial_observations import derive_financial_observations

router = APIRouter(prefix='/staging', tags=['integrated-assessment'])

class IntegratedAssessmentInput(BaseModel):
    assessment: AssessmentInput
    financial: FinancialEvidenceInput | None = None

@router.post('/integrated-assessment-review')
def integrated_assessment_review(data: IntegratedAssessmentInput):
    report = internal_review_draft(data.assessment)
    financial = review_financial_evidence(data.financial) if data.financial is not None else None
    derived = derive_financial_observations(data.financial) if data.financial is not None else None
    checks = []
    if financial is None:
        checks.append('No financial statement supplied; financial performance not evaluated.')
    else:
        provenance = financial['provenance']
        if not provenance['period_confirmed'] or not provenance['period_label']:
            checks.append('Confirm the financial reporting period and label before interpreting comparisons.')
        checks.extend(financial['warnings'])
        checks.extend(derived['checks'])
        checks.append('Reconcile parsed financial metrics to the original statements before drawing conclusions.')
    # Financial context is not passed into the decision engine. This is an
    # explicit safety boundary until metric semantics and source periods are validated.
    return {
        'assessment': report,
        'financial_evidence': financial,
        'financial_observations': derived,
        'integration_checks': checks,
        'financial_evidence_applied_to_diagnosis': False,
        'primary_constraint_confirmed': False,
        'report_ready': False,
        'customer_delivery_enabled': False,
        'production_connected': False,
    }
