"""Conservative decision triage; NOT a validated primary-constraint reasoner.

No hidden fixture expectations are inputs. Rules only classify candidate review paths;
reported observations do not establish causal attribution.
"""
from dataclasses import dataclass
from typing import Iterable
from .evidence import Observation, review_flags

@dataclass(frozen=True)
class TriageDecision:
    decision_type: str
    candidate_area: str | None
    rationale: str
    supporting_question_ids: tuple[str, ...]
    required_checks: tuple[str, ...]
    primary_constraint_confirmed: bool = False

AREAS = {
    'unbilled_work': 'billing_and_scope_governance',
    'peak_slot_mismatch': 'time_segmented_capacity',
    'credential_labor_shortage': 'external_labor_supply',
    'cash_pressure': 'cash_flow',
    'travel_productivity_loss': 'route_productivity',
    'billing_receivables_delay': 'billing_cash_conversion',
    'project_scope_overrun': 'scope_governance',
    'organic_client_growth_gap': 'organic_growth',
    'client_approval_delay': 'client_approval_workflow',
    'staffing_cash_cycle': 'staffing_working_capital',
    'claims_administration': 'claims_administration',
    'ticket_escalation_process': 'ticket_escalation_process',
    'event_deposit_cash_timing': 'event_cash_timing',
}

def triage(observations: Iterable[Observation], *, owner_goal: str | None = None) -> TriageDecision:
    obs = list(observations)
    by_name = {o.condition: o for o in obs}
    flags = list(review_flags(obs))
    def ids(*names):
        return tuple(dict.fromkeys(q for n in names if n in by_name for q in by_name[n].evidence_ids))
    if 'conflicting_evidence' in by_name:
        return TriageDecision('LEADING_HYPOTHESIS_REVIEW_REQUIRED', None,
            'Source disagreement prevents a supported causal conclusion.', ids('conflicting_evidence'),
            tuple(dict.fromkeys(flags + ['reconcile_sources_before_prioritizing'])))
    if 'financial_records_missing' in by_name:
        return TriageDecision('FINANCIAL_VISIBILITY_PRIORITY', 'financial_data_quality',
            'Missing records block a defensible financial root-cause finding.',
            ids('financial_records_missing','cash_pressure'),
            tuple(dict.fromkeys(flags + ['obtain_reliable_financial_records'])))
    if 'conversion_denominator_missing' in by_name:
        return TriageDecision('LEADING_HYPOTHESIS_REVIEW_REQUIRED', 'conversion_measurement',
            'Conversion cannot be assessed without its denominator.', ids('conversion_denominator_missing'),
            tuple(dict.fromkeys(flags + ['establish_conversion_denominator'])))
    if 'credential_labor_shortage' in by_name:
        return TriageDecision('EXTERNAL_CONSTRAINT_WITH_INTERNAL_ADAPTATION', 'external_labor_supply',
            'Reported market-wide labor shortage may constrain delivery; internal adaptation remains untested.',
            ids('credential_labor_shortage'), tuple(dict.fromkeys(flags + ['verify_market_shortage_and_internal_options'])))
    # Domain observations take precedence over generic cash symptoms when only one
    # supported operational mechanism is reported. Multiple mechanisms remain reviewable.
    domain_names = ('travel_productivity_loss','billing_receivables_delay','project_scope_overrun',
                    'organic_client_growth_gap','client_approval_delay','staffing_cash_cycle',
                    'peak_slot_mismatch','claims_administration','ticket_escalation_process',
                    'event_deposit_cash_timing','unbilled_work')
    candidates = [n for n in domain_names if n in by_name]
    if not candidates and 'cash_pressure' in by_name:candidates=['cash_pressure']
    if len(candidates) > 1:
        return TriageDecision('LEADING_HYPOTHESIS_REVIEW_REQUIRED', None,
            'Multiple reported concerns require relative magnitude and causal evaluation.',
            ids(*candidates),tuple(dict.fromkeys(flags + ['compare_competing_mechanisms'])))
    if candidates:
        name=candidates[0]
        return TriageDecision('PROVISIONAL_PRIORITY_REVIEW', AREAS[name],
            'A candidate condition is reported, but its size, cause and priority remain unverified.',
            ids(name),tuple(dict.fromkeys(flags + ['verify_magnitude_and_causal_mechanism'])))
    healthy = {'positive_financial_health','healthy_reliability'} <= by_name.keys()
    stable_goal = owner_goal and any(x in owner_goal.casefold() for x in ('maintain','remain same size','stay stable','stability'))
    if healthy and stable_goal:
        return TriageDecision('NO_MATERIAL_CONSTRAINT_INDICATED',None,
            'Positive financial and operating reports align with the stated stability goal; not independently verified.',
            ids('positive_financial_health','healthy_reliability'),('verify_health_claims_and_goal',))
    return TriageDecision('INSUFFICIENT_EVIDENCE',None,
        'Available observations do not support a defensible primary priority.',
        ids(*by_name),tuple(dict.fromkeys(flags + ['request_objective_and_quantified_evidence'])))
