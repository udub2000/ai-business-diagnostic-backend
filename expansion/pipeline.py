"""Isolated end-to-end component pipeline; NOT production integration or final report engine."""
from dataclasses import dataclass, asdict
from typing import Sequence
from .routing import route_business, EvidenceUnit
from .evidence import interpret_evidence, Observation
from .normalized import normalize_structured, StructuredSignal
from .decision import triage
from .domain_evidence import extract_operating_observations
from .structured_bridge import ConditionClaim, normalize_claims
from .questionnaire_bridge import ConditionSelection, questionnaire_condition_claims

@dataclass(frozen=True)
class PipelineResult:
    category: str
    decision: dict
    evidence_count: int
    observations: list[dict]
    preview: dict

def run_pipeline(business_type: str, answers: Sequence[dict], *, owner_goal: str | None = None,
                 structured_signals: Sequence[StructuredSignal] = (),
                 condition_claims: Sequence[ConditionClaim] = (),
                 condition_selections: Sequence[ConditionSelection] = ()) -> PipelineResult:
    category=route_business(business_type)
    units=[EvidenceUnit.from_questionnaire(x) for x in answers]
    observations=interpret_evidence(units)+extract_operating_observations(units)+normalize_structured(structured_signals)+normalize_claims(tuple(condition_claims)+tuple(questionnaire_condition_claims(answers, condition_selections)))
    # Merge overlapping observations without double-counting supporting questions.
    merged={}
    for o in observations:
        key=(o.condition,o.polarity)
        if key not in merged: merged[key]=o
        else:
            prev=merged[key]
            merged[key]=Observation(o.condition,o.polarity,
                tuple(dict.fromkeys(prev.evidence_ids+o.evidence_ids)),
                tuple(dict.fromkeys(prev.excerpts+o.excerpts)),
                'mixed' if prev.status!=o.status else o.status)
    observations=list(merged.values())
    decision=triage(observations,owner_goal=owner_goal)
    # Deliberately NOT a final six-section diagnostic: no invented impact estimates/actions.
    preview={'business_category':category,'assessment_status':decision.decision_type,
             'candidate_area':decision.candidate_area,'primary_constraint_confirmed':False,
             'why':decision.rationale,'evidence_question_ids':list(decision.supporting_question_ids),
             'verification_needed':list(decision.required_checks),
             'report_ready':False}
    return PipelineResult(category,asdict(decision),len(units),[asdict(o) for o in observations],preview)
