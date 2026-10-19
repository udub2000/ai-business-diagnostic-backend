"""Isolated structured observation normalization; never consumes fixture expected labels.

Use explicit question semantics and measured fields, not case-name matching. This
is an adjunct to (not a replacement for) text interpretation and not a diagnosis.
"""
from dataclasses import dataclass
from typing import Mapping, Sequence
from .evidence import Observation

@dataclass(frozen=True)
class StructuredSignal:
    field: str
    value: object
    question_id: str
    source: str = 'owner_report'
    period: str | None = None

# Explicitly whitelisted semantic input fields. Unsupported fields are ignored,
# preventing arbitrary user strings from masquerading as recognized evidence.
FIELD_CONDITIONS = {
    'financial_records_available': ('financial_records_missing', 'data_gap'),
    'conversion_denominator_available': ('conversion_denominator_missing', 'data_gap'),
    'external_credential_shortage_reported': ('credential_labor_shortage', 'external_concern'),
    'customer_log_conflict_reported': ('conflicting_evidence', 'conflicted'),
    'unbilled_extra_work_reported': ('unbilled_work', 'concern'),
}

def normalize_structured(signals: Sequence[StructuredSignal]) -> list[Observation]:
    by_field: dict[str,list[StructuredSignal]] = {}
    for s in signals:
        if s.field in FIELD_CONDITIONS or s.field in {'peak_fill_pct','offpeak_fill_pct'}:
            by_field.setdefault(s.field, []).append(s)
    results: list[Observation] = []
    for field,(condition,polarity) in FIELD_CONDITIONS.items():
        values=by_field.get(field,[])
        # A missing record/denominator is indicated by False, other signals by True.
        target = field not in {'financial_records_available','conversion_denominator_available'}
        positives=[s for s in values if type(s.value) is bool and s.value is target]
        opposites=[s for s in values if type(s.value) is bool and s.value is not target]
        if positives and opposites:
            results.append(Observation('conflicting_evidence','conflicted',
                tuple(s.question_id for s in values),tuple(f'{field}={s.value}' for s in values),'conflicting_structured_sources'))
        elif positives:
            results.append(Observation(condition,polarity,tuple(s.question_id for s in positives),
                tuple(f'{field}={s.value}' for s in positives),'structured_report_unverified'))
    peaks=by_field.get('peak_fill_pct',[])
    offpeaks=by_field.get('offpeak_fill_pct',[])
    if len(peaks)==1 and len(offpeaks)==1:
        a,b=peaks[0],offpeaks[0]
        if all(type(s.value) in (int,float) and 0<=s.value<=100 for s in (a,b)) and (a.period==b.period or not a.period or not b.period):
            if a.value>=90 and b.value<=50 and a.value-b.value>=40:
                results.append(Observation('peak_slot_mismatch','concern',(a.question_id,b.question_id),
                    (f'peak_fill_pct={a.value}',f'offpeak_fill_pct={b.value}'),'structured_report_unverified'))
    return results
