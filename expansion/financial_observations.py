"""Narrow, review-only financial arithmetic, never causal constraint scoring.

Only uses same-statement metrics with an explicitly confirmed, labeled period.
The resulting observations are arithmetic checks, not diagnoses or independent verification.
"""
from __future__ import annotations
from math import isclose
from .financial_evidence_bridge import FinancialEvidenceInput


def derive_financial_observations(data: FinancialEvidenceInput) -> dict:
    checks: list[str] = []
    observations: list[dict] = []
    if not (data.period_confirmed and data.period_type != 'unknown' and data.period_label):
        return {'observations': [], 'checks': ['Confirm statement period type and label before deriving financial ratios.'],
                'applied_to_constraint_selection': False}
    if data.statement_type != 'income_statement':
        return {'observations': [], 'checks': ['Ratios require a single-period income statement; mixed or other statement types are excluded.'],
                'applied_to_constraint_selection': False}
    m = data.metrics
    revenue = m.get('revenue')
    cogs = m.get('cost_of_goods_sold')
    gross_profit = m.get('gross_profit')
    if revenue is None or revenue <= 0:
        checks.append('A positive revenue denominator is required for margin analysis.')
    else:
        if gross_profit is not None and cogs is not None:
            if not isclose(revenue - cogs, gross_profit, rel_tol=0.01, abs_tol=1.0):
                checks.append('Gross profit does not reconcile to revenue minus cost of goods sold; margin withheld.')
            else:
                observations.append({'code': 'gross_margin_arithmetic', 'value_pct': round(100 * gross_profit / revenue, 2),
                    'metric_keys': ['revenue', 'cost_of_goods_sold', 'gross_profit'],
                    'period_label': data.period_label, 'status': 'unverified_arithmetic_only'})
        elif gross_profit is not None:
            observations.append({'code': 'gross_margin_arithmetic', 'value_pct': round(100 * gross_profit / revenue, 2),
                'metric_keys': ['revenue', 'gross_profit'], 'period_label': data.period_label,
                'status': 'unverified_arithmetic_only'})
        else:
            checks.append('Gross profit missing; no gross margin calculated.')
    checks.append('Reconcile values to the original statement; confirm metric definitions and same-period comparability.')
    return {'observations': observations, 'checks': checks, 'applied_to_constraint_selection': False}
