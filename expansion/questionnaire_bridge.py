"""Controlled questionnaire-to-condition bridge (development only).

A questionnaire control must explicitly submit its condition code and assertion;
ordinary narrative answers are never silently converted to a condition claim.
"""
from dataclasses import dataclass
from typing import Sequence
from .routing import EvidenceUnit
from .structured_bridge import ConditionClaim, CONDITIONS

@dataclass(frozen=True)
class ConditionSelection:
    question_id: str
    condition_code: str
    asserted: bool
    selected_by: str = 'respondent'


def questionnaire_condition_claims(answers: Sequence[dict], selections: Sequence[ConditionSelection]) -> list[ConditionClaim]:
    units = [EvidenceUnit.from_questionnaire(a) for a in answers]
    by_id = {}
    for unit in units:
        if unit.question_id in by_id:
            raise ValueError(f'Duplicate question ID: {unit.question_id}')
        by_id[unit.question_id] = unit
    seen=set(); result=[]
    for sel in selections:
        if sel.condition_code not in CONDITIONS:
            raise ValueError('Unsupported condition code')
        if type(sel.asserted) is not bool:
            raise TypeError('Selection assertion must be boolean')
        if sel.selected_by not in ('respondent','reviewer'):
            raise ValueError('Condition selection must have explicit respondent or reviewer provenance')
        if sel.question_id not in by_id:
            raise ValueError('Condition selection must refer to an existing answer')
        key=(sel.question_id,sel.condition_code)
        if key in seen:
            raise ValueError('Duplicate condition selection')
        seen.add(key)
        unit=by_id[sel.question_id]
        if not unit.response.strip() or unit.response.strip().casefold().startswith(('unknown','not supplied','not reported','not available')):
            if sel.asserted:
                raise ValueError('Cannot assert condition using an empty or unknown response')
            continue
        result.append(ConditionClaim(sel.condition_code,unit.question_id,unit.response,sel.asserted,
                                     provenance=f'{sel.selected_by}_selection'))
    return result
