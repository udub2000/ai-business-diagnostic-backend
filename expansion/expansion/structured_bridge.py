"""Typed intake bridge for explicit condition codes; no narrative semantic inference.

Condition codes must come from a controlled question/selection or a reviewed
extractor, never from hidden test labels. All outputs remain unverified claims.
"""
from dataclasses import dataclass
from typing import Sequence
from .evidence import Observation

CONDITIONS = frozenset({
    'travel_productivity_loss','billing_receivables_delay','project_scope_overrun',
    'organic_client_growth_gap','client_approval_delay','staffing_cash_cycle',
    'peak_slot_mismatch','claims_administration','ticket_escalation_process',
    'event_deposit_cash_timing','unbilled_work','financial_records_missing',
    'conversion_denominator_missing','credential_labor_shortage',
    'conflicting_evidence','positive_financial_health','healthy_reliability',
})

@dataclass(frozen=True)
class ConditionClaim:
    condition: str
    question_id: str
    source_text: str
    asserted: bool
    provenance: str = 'owner_report'


def normalize_claims(claims: Sequence[ConditionClaim]) -> list[Observation]:
    result=[]
    for c in claims:
        if c.condition not in CONDITIONS:
            raise ValueError('Unsupported structured condition')
        if not c.question_id.strip() or not c.source_text.strip():
            raise ValueError('Condition requires question ID and source text')
        if type(c.asserted) is not bool:
            raise TypeError('Condition assertion must be boolean')
        if not c.asserted:
            continue
        polarity=('positive' if c.condition in {'positive_financial_health','healthy_reliability'}
                  else 'data_gap' if c.condition in {'financial_records_missing','conversion_denominator_missing'}
                  else 'conflicted' if c.condition=='conflicting_evidence'
                  else 'external_concern' if c.condition=='credential_labor_shortage'
                  else 'concern')
        result.append(Observation(c.condition,polarity,(c.question_id,),
                                  (c.source_text,),'structured_report_unverified'))
    return result
