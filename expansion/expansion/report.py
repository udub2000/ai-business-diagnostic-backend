"""Source-grounded SIX SECTION REVIEW DRAFT, not customer-ready diagnostic report.

Never asserts a confirmed causal constraint or quantifies unverified impact.
"""
from dataclasses import dataclass, asdict
from .pipeline import PipelineResult

ACTIONS = {
 'route_productivity': ('Compare travel time, stops, and overtime by route over the same weeks.', 'Check whether routing or staffing explains the observed capacity pressure.', 'Pilot route clustering and measure travel hours and jobs per crew-day.'),
 'billing_cash_conversion': ('Reconcile completed jobs, invoice dates, and outstanding receivables.', 'Identify approval, documentation, and collection handoff delays.', 'Pilot a billing checklist and monitor days from completion to payment.'),
 'scope_governance': ('Compare estimated and actual hours by engagement and change request.', 'Review who approves out-of-scope work and when clients are notified.', 'Pilot scope-change signoff and track unbilled hours and project margin.'),
 'organic_growth': ('Separate referrals, retained clients, and other leads by source and period.', 'Compare qualified inquiries and conversion before concluding demand is the cause.', 'Pilot a referral follow-up process and measure qualified introductions.'),
 'client_approval_workflow': ('Measure time waiting for client approvals at each project milestone.', 'Separate approval delays from internal production bottlenecks.', 'Pilot milestone reminders and track approval turnaround and project cycle time.'),
 'staffing_working_capital': ('Map payroll timing against customer receipts and placement start dates.', 'Separate cash-cycle pressure from placement margin or recruitment throughput.', 'Test invoicing or funding-timing adjustments and monitor cash coverage.'),
 'claims_administration': ('Measure claim-processing age, rework, and missing documentation.', 'Separate payer response time from internal submission and follow-up delays.', 'Pilot a claim follow-up queue and track aged receivables and first-pass acceptance.'),
 'ticket_escalation_process': ('Measure ticket aging, escalation handoffs, and repeat contacts.', 'Compare ticket complexity and workload before attributing delays to staffing.', 'Pilot escalation criteria and monitor resolution time and reopen rates.'),
 'event_cash_timing': ('Map deposit due dates, vendor commitments, and customer payments by event.', 'Compare event profitability with short-term cash timing requirements.', 'Pilot revised deposit milestones and monitor cash gaps before events.'),
 'billing_and_scope_governance': ('Compare completed work with issued invoices for the same period.', 'Review scope-change approval and billing handoffs.', 'Measure unbilled work by project before choosing an intervention.'),
 'time_segmented_capacity': ('Validate appointment demand and fill by time segment.', 'Review options to shift suitable demand into underused slots.', 'Pilot one schedule change and measure waitlists and utilization.'),
 'external_labor_supply': ('Verify credential requirements and the reported regional supply constraint.', 'Identify work that can be reassigned without violating credentials.', 'Evaluate training, recruitment, and scheduling alternatives.'),
 'financial_data_quality': ('Reconstruct current income, costs, receivables, and cash balances.', 'Reconcile owner estimates to bank and accounting records.', 'Reassess cash causes only after reliable records are available.'),
 'conversion_measurement': ('Define the eligible opportunity population and measurement period.', 'Reconcile numerator and denominator across the same cohort.', 'Assess conversion only after the baseline is verified.'),
 'cash_flow': ('Reconcile cash balances, receivables, and payment commitments.', 'Compare cash timing against earned revenue and costs.', 'Identify whether timing or unit economics explains the pressure.'),
}
GENERIC = ('Verify the reported condition against source records.', 'Compare plausible causes and their relative magnitude.', 'Select an intervention only after the evidence supports it.')

@dataclass(frozen=True)
class ReportDraft:
    business_snapshot: str
    primary_constraint: str
    why_evidence: str
    recommended_actions: tuple[str, ...]
    expected_impact: str
    secondary_observations: str
    report_status: str = 'INTERNAL_REVIEW_DRAFT_NOT_CUSTOMER_READY'

def build_report_draft(result: PipelineResult, *, business_type: str, owner_goal: str | None = None) -> ReportDraft:
    decision=result.decision
    dtype=decision['decision_type']
    area=decision['candidate_area']
    evidence=decision['supporting_question_ids']
    if dtype=='NO_MATERIAL_CONSTRAINT_INDICATED':
        primary='No material primary constraint indicated by the available reports; verification remains necessary.'
    elif dtype=='INSUFFICIENT_EVIDENCE':
        primary='No defensible primary constraint established; further evidence is required.'
    elif dtype=='LEADING_HYPOTHESIS_REVIEW_REQUIRED':
        primary='Competing or unresolved hypotheses require review; no primary constraint confirmed.'
    else:
        primary=f'Candidate priority: {area or "undetermined"} ({dtype.lower().replace("_"," ")}); not a confirmed root cause.'
    actions=ACTIONS.get(area,GENERIC)
    if dtype=='NO_MATERIAL_CONSTRAINT_INDICATED':
        actions=('Confirm the owner’s stability objective and review operating results.', 'Monitor cash, service reliability, and capacity at an appropriate interval.', 'Investigate only meaningful deviations from the agreed objectives.')
    elif dtype=='LEADING_HYPOTHESIS_REVIEW_REQUIRED':
        actions=('Reconcile contradictory or incomplete source records.', 'Define and compare plausible mechanisms using comparable periods.', 'Defer causal intervention until one explanation is sufficiently supported.')
    elif dtype=='INSUFFICIENT_EVIDENCE':
        actions=('Confirm the owner’s primary business objective.', 'Collect current operating and financial measures with periods and sources.', 'Reassess priorities once sufficient comparable evidence is available.')
    snapshot=f'{business_type} | Category: {result.category} | Goal: {owner_goal or "not supplied"} | Questionnaire responses: {result.evidence_count}. No independent verification performed.'
    why=f'{decision["rationale"]} Source question IDs: {", ".join(evidence) if evidence else "none available"}. Evidence is reported or preliminary; causation is not established.'
    impact='Potential impact is directional only: better decision confidence and prioritization after verification. No monetary or percentage estimate is supported by the current inputs.'
    secondary='Review checks: '+(', '.join(decision['required_checks']) if decision['required_checks'] else 'none recorded')+'. Other constraints have not been systematically ruled out.'
    return ReportDraft(snapshot,primary,why,actions,impact,secondary)

def as_sections(report: ReportDraft) -> dict:
    return {k:v for k,v in asdict(report).items() if k!='report_status'}
