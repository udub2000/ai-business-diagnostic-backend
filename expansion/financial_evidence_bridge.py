"""Conservative financial evidence review. No automatic constraint claims or scoring."""
from __future__ import annotations
from math import isfinite
from typing import Literal
from pydantic import BaseModel, Field, field_validator

class FinancialEvidenceInput(BaseModel):
    metrics: dict[str, float | None] = Field(default_factory=dict)
    period_type: Literal['annual','quarterly','monthly','unknown'] = 'unknown'
    period_label: str | None = Field(default=None, max_length=120)
    period_confirmed: bool = False
    statement_type: Literal['income_statement','balance_sheet','cash_flow_statement','mixed'] = 'income_statement'

    @field_validator('metrics')
    @classmethod
    def valid_metrics(cls, metrics):
        if len(metrics)>100:
            raise ValueError('Too many financial metrics')
        for name,value in metrics.items():
            if not name or len(name)>100:
                raise ValueError('Invalid metric name')
            if value is not None and not isfinite(value):
                raise ValueError('Non-finite metric')
        return metrics

def review_financial_evidence(data: FinancialEvidenceInput) -> dict:
    """Return traceable observations; do not assert financial health or primary constraint."""
    metrics=dict(data.metrics)
    warnings=[]
    # The legacy parser divides revenue by 12 regardless of reporting period.
    # Suppress that derivative unless the period is independently confirmed annual.
    metrics.pop('monthly_revenue',None)
    if data.period_type=='unknown' or not data.period_confirmed:
        warnings.append('Statement reporting period is not independently confirmed; monthly revenue is withheld.')
    if data.period_type=='annual' and data.period_confirmed and metrics.get('revenue') is not None:
        metrics['monthly_revenue']=round(metrics['revenue']/12,2)
    elif data.period_type in ('quarterly','monthly') and data.period_confirmed:
        warnings.append('Nonannual statement: no annualization or monthly revenue inference is performed.')
    if data.period_type!='unknown' and not data.period_label:
        warnings.append('Reporting period label missing; year/date attribution needs review.')
    if data.statement_type=='mixed':
        warnings.append('Mixed statement periods may differ; verify each source period before comparisons.')
    return {
        'status':'evidence_review_only',
        'metrics':metrics,
        'provenance':{'source':'uploaded_financial_statement','statement_type':data.statement_type,
                      'period_type':data.period_type,'period_label':data.period_label,
                      'period_confirmed':data.period_confirmed,'independently_verified':False},
        'warnings':warnings,
        'primary_constraint_confirmed':False,'report_ready':False,
        'financial_evidence_applied_to_diagnosis':False,
    }
