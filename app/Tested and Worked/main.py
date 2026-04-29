from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Business Diagnostic API",
    version="v2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MOCK_REPORT = {
    "assessment_id": "test123",
    "scores": {
        "demand": 80,
        "conversion": 70,
        "capacity": 40,
        "operations": 60,
        "financials": 55
    },
    "category_confidence": {
        "demand": 82,
        "conversion": 75,
        "capacity": 88,
        "operations": 70,
        "financials": 68
    },
    "primary_constraint": "Capacity",
    "secondary_constraint": "Financials",
    "constraint_pattern": "Demand exceeds fulfillment capacity",
    "overall_confidence_score": 78,
    "overall_confidence_label": "Medium",
    "assumptions": ["Test mode active"],
    "contradictions": [],
    "report_markdown": """
# Business Snapshot

- Strong demand profile
- Healthy close rate
- Crews appear fully utilized
- Backlog suggests capacity strain
- Financial performance is positive but somewhat tight

# Primary Constraint

Your business is currently constrained by **Capacity**, meaning you have enough demand to grow, but not enough fulfillment bandwidth to absorb more work efficiently.

# Why This Was Identified

The lowest category score is Capacity. This suggests the business is not primarily struggling with demand generation or conversion, but with the ability to fulfill the volume of work coming in.

# What This Means for the Business

In practical terms, this often shows up as long booking windows, scheduling pressure, difficulty fitting in new clients, and missed revenue opportunities.

# Risks of Inaction

If this continues, growth will stall because additional demand will not convert into completed jobs at the same rate.

# Recommended Actions

1. Add labor capacity  
   → Increase the number of crews or available labor hours.

2. Improve routing and scheduling  
   → Reduce wasted time so current crews can complete more work.

3. Raise pricing selectively  
   → Shape demand and improve margin while capacity remains tight.

# Expected Impact

If addressed, this should shorten backlog, improve customer responsiveness, and increase revenue capture from existing demand.

# Assumptions and Confidence

This is a test-mode report with mocked values, so treat it as a connectivity check rather than a final diagnostic.
"""
}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": "AI Business Diagnostic API",
        "model_version": "v2"
    }


@app.post("/api/v1/assessments/report")
def create_report():
    return MOCK_REPORT


@app.get("/api/v1/assessments/{assessment_id}")
def get_assessment(assessment_id: str):
    if assessment_id == MOCK_REPORT["assessment_id"]:
        return MOCK_REPORT
    return {
        "assessment_id": assessment_id,
        "scores": {
            "demand": 75,
            "conversion": 65,
            "capacity": 45,
            "operations": 58,
            "financials": 60
        },
        "category_confidence": {
            "demand": 70,
            "conversion": 68,
            "capacity": 80,
            "operations": 66,
            "financials": 64
        },
        "primary_constraint": "Capacity",
        "secondary_constraint": "Operations",
        "constraint_pattern": "Fulfillment bandwidth is limiting growth",
        "overall_confidence_score": 72,
        "overall_confidence_label": "Medium",
        "assumptions": ["Fallback mock response"],
        "contradictions": [],
        "report_markdown": """
# Business Snapshot

This is a fallback mock report used to confirm the frontend can load report pages.

# Primary Constraint

Capacity is the current limiting factor.

# Recommendations

1. Add labor
2. Improve routing
3. Adjust pricing
"""
    }