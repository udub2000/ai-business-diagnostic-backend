"""Cross-business observation extraction from explicit reported operational evidence.

No fixture labels, business IDs, or expected outcomes are used. Candidate observations
are not causal conclusions; require reviewer checks. A small transparent rule set.
"""
import re
from .evidence import Observation

def extract_operating_observations(units):
    found={}
    def add(kind, unit):
        if kind not in found:
            found[kind]=Observation(kind,'concern',(unit.question_id,),(unit.response,), 'reported_unverified')
        else:
            old=found[kind]
            found[kind]=Observation(kind,'concern',tuple(dict.fromkeys(old.evidence_ids+(unit.question_id,))),tuple(dict.fromkeys(old.excerpts+(unit.response,))), 'reported_unverified')
    for u in units:
        s=u.response.lower()
        if not s or s.startswith(('unknown','not supplied','not reported','not available')):continue
        patterns={
          'travel_productivity_loss': [r'\b(?:travel|driving|waiting|route)\b.{0,85}\b(?:35%|inefficien\w*|loss|significant|overtime|paid\s+(?:crew\s+)?hours)\b',r'\b(?:\d{2}%\s+of\s+paid\s+crew\s+hours)\s+(?:travel|or wait)'],
          'billing_receivables_delay':[r'\binvoic\w*\b.{0,80}\b(?:\d+\s+days\s+after|delay\w*|late|overdue)\b',r'\b(?:receivables|ar)\b.{0,35}\b(?:>\s*60|over\s*60|overdue)\b',r'\b(?:\d+%\s+of\s+(?:ar|receivables))\b'],
          'project_scope_overrun':[r'\b(?:projects?|work)\b.{0,90}\b(?:exceed\w*|over\s+scope|out.of.scope)\b',r'\b(?:scope\s+changes?|scope\s+control)\b.{0,80}\b(?:overrun|fee|hours|delivery)\b'],
          'organic_client_growth_gap':[r'\baum\b.{0,85}\b(?:market|flows?)\b',r'\bnet\s+(?:client\s+)?flows?\b.{0,30}\b(?:negative|−|-)\s*\d'],
          'client_approval_delay':[r'\b(?:offers?|placements?|requisitions?)\b.{0,85}\b(?:delayed|approval\w*)\b',r'\bclient\s+approvals?\b'],
          'staffing_cash_cycle':[r'\b(?:staffing|payroll)\b.{0,100}\b(?:weekly|net.?60|payment)\b',r'\bweekly\s+payroll\b.{0,75}\b(?:net.?60|client\s+payment)\b'],
          'peak_slot_mismatch':[r'\b(?:evening|peak)\b.{0,75}\b(?:98%|waitlist\w*|full)\b',r'\b(?:midday|morning)\b.{0,55}\b(?:22%|empty|low|unused)\b'],
          'claims_administration':[r'\b(?:claims?|denials?)\b.{0,75}\b(?:denied|denials?|16%|administration|payment)\b'],
          'ticket_escalation_process':[r'\b(?:sla|ticket\w*|escalation\w*)\b.{0,100}\b(?:backlog|bounce|owner\w*|82%|95%)\b',r'\b(?:34%\s+ticket|escalations?\s+lack)\b'],
          'event_deposit_cash_timing':[r'\b(?:deposits?|vendors?|client\s+balances?)\b.{0,100}\b(?:outstanding|before|paid|due)\b'],
        }
        for kind,ps in patterns.items():
            if any(re.search(p,s) for p in ps):add(kind,u)
    return list(found.values())
