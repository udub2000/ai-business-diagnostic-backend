"""Step 7 category routing and typed evidence intake. Development only.

No causal diagnosis or report generation is claimed by this module.
"""
from dataclasses import dataclass
from typing import Optional

CATEGORIES = (
    "Home & Field Services",
    "Professional Services",
    "Relationship & Transaction Services",
    "Other Client-Based Business",
)

# Explicit routing, not keyword heuristics. Review new business types before addition.
BUSINESS_ROUTES = {
    "lawn care": CATEGORIES[0], "roofing": CATEGORIES[0],
    "cleaning": CATEGORIES[0], "property services": CATEGORIES[0],
    "field installation": CATEGORIES[0], "installer": CATEGORIES[0],
    "landscaping": CATEGORIES[0],
    "fixed-fee consulting": CATEGORIES[1], "wealth advisory": CATEGORIES[1],
    "financial advisory": CATEGORIES[1], "accounting": CATEGORIES[1],
    "professional-services feedback loop": CATEGORIES[1],
    "recruiting": CATEGORIES[2], "brokerage and staffing": CATEGORIES[2],
    "brokerage": CATEGORIES[2],
    "fitness studio": CATEGORIES[3], "outpatient rehabilitation": CATEGORIES[3],
    "managed it": CATEGORIES[3], "it services": CATEGORIES[3],
    "event management": CATEGORIES[3], "therapy": CATEGORIES[3],
    "salon": CATEGORIES[3],
}

class RoutingReviewRequired(ValueError):
    pass

def route_business(business_type: str) -> str:
    if not isinstance(business_type, str) or not business_type.strip():
        raise RoutingReviewRequired("Business type is required")
    key = " ".join(business_type.casefold().replace("&", "and").split())
    result = BUSINESS_ROUTES.get(key)
    if result is None:
        raise RoutingReviewRequired(f"Unmapped business type requires review: {business_type}")
    return result

@dataclass(frozen=True)
class EvidenceUnit:
    question_id: str
    response: str
    evidence_state: str
    source: Optional[str] = None
    period: Optional[str] = None
    units: Optional[str] = None
    confidence: Optional[str] = None

    @classmethod
    def from_questionnaire(cls, item: dict):
        if not isinstance(item, dict):
            raise TypeError("Questionnaire answer must be an object")
        qid = item.get("question_id") or item.get("id")
        answer = item.get("response") if "response" in item else item.get("answer")
        state = item.get("evidence_state") or item.get("status") or "unspecified"
        if not qid:
            raise ValueError("Question identifier required")
        if answer is None:
            answer = ""
        if not isinstance(answer, (str, int, float, bool)):
            raise ValueError("Non-scalar response must be separately represented")
        return cls(str(qid), str(answer), str(state),
                   source=str(item.get("provenance")) if item.get("provenance") is not None else None,
                   period=str(item.get("period")) if item.get("period") is not None else None,
                   units=str(item.get("units")) if item.get("units") is not None else None,
                   confidence=str(item.get("confidence")) if item.get("confidence") is not None else None)
