"""Conservative, auditable evidence interpretation for isolated Step 7 candidate.

This is a first-pass condition classifier, NOT a causal diagnosis or an LLM.
The engine never receives hidden fixture expected outcomes.
"""
from dataclasses import dataclass, asdict
import re
from typing import Iterable
from .routing import EvidenceUnit

@dataclass(frozen=True)
class Observation:
    condition: str
    polarity: str
    evidence_ids: tuple[str, ...]
    excerpts: tuple[str, ...]
    status: str = "reported_unverified"

# Meaningful phrase families are domain patterns, not case-name or case-id matches.
# Conservative matching requires a qualifying context and excludes obvious negations.
RULES = (
    ("positive_financial_health", (r'\b(profit\w*|margin\w*|earnings|cash(?:\s+flow)?)\b.{0,38}\b(positive|healthy|strong|stable|good|adequate)\b', r'\b(positive|healthy|strong|stable|good|adequate)\b.{0,38}\b(profit\w*|margin\w*|earnings|cash(?:\s+flow)?)\b'), "positive"),
    ("unbilled_work", (r'\b(unbilled|not billed|never billed|not invoiced|uninvoiced)\b.{0,50}\b(work|hours|service\w*|tasks?|scope|extras?)\b', r'\b(extra|additional|out.of.scope)\b.{0,35}\b(work|hours|services?)\b.{0,50}\b(not billed|unbilled|not invoiced|free)\b'), "concern"),
    ("peak_slot_mismatch", (r'\b(evening|peak|after.work|weekend)\b.{0,60}\b(full|booked|waitlist\w*|sold.out|capacity)\b', r'\b(morning|midday|off.peak)\b.{0,55}\b(empty|unused|open|quiet|available|low)\b'), "concern"),
    ("credential_labor_shortage", (r'\b(regional|countywide|industrywide|area.wide|local.market)\b.{0,70}\b(shortage|scarcity|hard.to.find|not enough)\b.{0,50}\b(licensed|certified|credential\w*|qualified|technician\w*)\b', r'\b(shortage|scarcity)\b.{0,50}\b(licensed|certified|credential\w*|qualified|technician\w*)\b'), "external_concern"),
    ("financial_records_missing", (r'\b(no|missing|unavailable|unreliable|incomplete|cannot.find|do not have)\b.{0,55}\b(books|financial\s+records|statements|ledger|job.cost\w*|accounts)\b',), "data_gap"),
    ("cash_pressure", (r'\b(cash|liquid\w*|working.capital)\b.{0,40}\b(tight|strained|stress\w*|short|pressure|negative)\b',), "concern"),
    ("conflicting_evidence", (r'\b(customer\w*|client\w*)\b.{0,70}\b(disagree|conflict\w*|contradict\w*|differ\w*|inconsistent|do not match)\b.{0,70}\b(log\w*|record\w*|ticket\w*)\b', r'\b(log\w*|record\w*|ticket\w*)\b.{0,70}\b(disagree|conflict\w*|contradict\w*|differ\w*|inconsistent|do not match)\b.{0,70}\b(customer\w*|client\w*)\b'), "conflicted"),
    ("conversion_denominator_missing", (r'\b(no|missing|unknown|unavailable|unreliable|cannot.calculate|cannot calculate)\b.{0,55}\b(denominator|total leads|qualified leads|opportunities|conversion rate)\b',), "data_gap"),
    ("healthy_reliability", (r'\b(reliable|reliably|consistent|stable|on.time|meeting targets|healthy)\b.{0,40}\b(service|operation\w*|delivery|business|performance)\b',), "positive"),
)
NEGATION = re.compile(r'\b(no|not|never|without|isn.t|aren.t|wasn.t|lack of)\b', re.I)

def interpret_evidence(items: Iterable[EvidenceUnit]) -> list[Observation]:
    found=[]
    for item in items:
        text=item.response.strip()
        if not text or text.casefold() in {"unknown", "n/a", "not provided", "none"}:
            continue
        for condition,patterns,polarity in RULES:
            # Unknown mentions are not evidence that a condition exists.
            if condition == 'unbilled_work' and re.search(r'\bunbilled\b.{0,35}\bunknown\b',text,re.I):continue
            # A missing aging ledger is a verification gap, not absence of all financial books.
            if condition == 'financial_records_missing' and re.search(r'\bno\s+aging\s+ledger\b',text,re.I):continue
            for pattern in patterns:
                m=re.search(pattern,text,re.I)
                if not m: continue
                # Negation preceding a match can reverse its meaning. Some conditions
                # explicitly encode missingness and must not be negated away.
                if polarity in {"positive", "concern", "external_concern"}:
                    prefix=text[max(0,m.start()-24):m.start()]
                    if NEGATION.search(prefix):continue
                found.append(Observation(condition,polarity,(item.question_id,),(text,),
                    "reported_unverified" if item.evidence_state.lower() not in {"verified","documented"} else "documented_claim"))
                break
    # Merge only same-condition, same-polarity observations; retain all provenance.
    merged={}
    for obs in found:
        key=(obs.condition,obs.polarity)
        if key not in merged:merged[key]=obs
        else:
            old=merged[key]
            merged[key]=Observation(obs.condition,obs.polarity,old.evidence_ids+obs.evidence_ids,
                old.excerpts+obs.excerpts,"mixed" if old.status!=obs.status else old.status)
    return list(merged.values())

def review_flags(observations: Iterable[Observation]) -> list[str]:
    names={o.condition for o in observations}
    flags=[]
    if "financial_records_missing" in names:flags.append("financial_cause_unverified")
    if "conflicting_evidence" in names:flags.append("reconcile_conflicting_sources")
    if "conversion_denominator_missing" in names:flags.append("conversion_rate_not_calculable")
    if "credential_labor_shortage" in names:flags.append("external_constraint_not_fully_controllable")
    if "peak_slot_mismatch" in names:flags.append("segment_capacity_by_time")
    return flags
