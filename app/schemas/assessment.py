from typing import Any

from pydantic import BaseModel, Field

from app.schemas.business import BusinessCreate, BusinessRead


class IntakePayload(BaseModel):
    leads_per_week: float | None = None
    conversion_rate: float | None = None
    avg_job_value: float | None = None
    crews: int | None = None
    jobs_per_crew_per_day: float | None = None
    backlog_days: float | None = None
    utilization_rate: float | None = None
    labor_cost_pct: float | None = None
    gross_margin_pct: float | None = None
    travel_time_minutes: float | None = None
    demand_consistency: str | None = None
    schedule_fullness: str | None = None
    profitability_feeling: str | None = None
    response_speed: str | None = None
    turned_away_work: str | None = None
    can_take_work_tomorrow: str | None = None
    notes: str | None = None


class ScoreRequest(BaseModel):
    business: BusinessCreate
    intake: IntakePayload


class ScoreResponse(BaseModel):
    scores: dict[str, int]
    category_confidence: dict[str, int]
    primary_constraint: str
    secondary_constraint: str | None
    constraint_pattern: str
    overall_confidence_score: int
    overall_confidence_label: str
    contradictions: list[str]
    assumptions: list[str]
    recommended_action_pool: list[str]


class ReportRequest(BaseModel):
    business: BusinessCreate
    intake: IntakePayload
    generate_report: bool = True


class AssessmentRead(BaseModel):
    id: int
    business: BusinessRead
    score_payload: dict[str, Any]
    report_markdown: str | None = None

    class Config:
        from_attributes = True
