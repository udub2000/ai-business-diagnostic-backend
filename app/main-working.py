from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Generator, Optional
from uuid import uuid4

import anthropic
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import Base, SessionLocal, engine
from app.db.repository import create_assessment, get_assessment_by_id, list_recent_assessments

load_dotenv()
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Business Diagnostic API", version="v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AssessmentInput(BaseModel):
    business_type: str
    location: str
    owner_name: Optional[str] = None
    seasonality_context: Optional[str] = None
    analysis_mode: str = "auto"
    mode: str = "guided"
    leads_per_week: Optional[float] = None
    leads_estimate: Optional[str] = None
    schedule_fullness: Optional[str] = None
    backlog_days: Optional[float] = None
    backlog_bucket: Optional[str] = None
    conversion_rate: Optional[float] = None
    quotes_to_jobs_bucket: Optional[str] = None
    sales_behavior: Optional[str] = None
    response_speed: Optional[str] = None
    crews: Optional[int] = None
    jobs_per_crew_per_day: Optional[float] = None
    busy_day_description: Optional[str] = None
    turned_down_work: Optional[str] = None
    utilization_rate: Optional[float] = None
    can_take_work_tomorrow: Optional[str] = None
    average_job_time_hours: Optional[float] = None
    travel_time_minutes: Optional[float] = None
    scheduling_efficiency: Optional[str] = None
    time_loss_source: Optional[str] = None
    admin_drag: Optional[str] = None
    monthly_revenue: Optional[float] = None
    monthly_revenue_range: Optional[str] = None
    labor_cost_pct: Optional[float] = None
    profitability_feeling: Optional[str] = None
    gross_margin_pct: Optional[float] = None
    pricing_adequacy: Optional[str] = None
    cash_stress: Optional[str] = None
    notes: Optional[str] = None


def get_structured_input_completeness(payload: AssessmentInput) -> float:
    structured_fields = [
        payload.leads_per_week, payload.leads_estimate, payload.schedule_fullness,
        payload.backlog_days, payload.backlog_bucket,
        payload.conversion_rate, payload.quotes_to_jobs_bucket, payload.sales_behavior,
        payload.response_speed,
        payload.crews, payload.jobs_per_crew_per_day, payload.busy_day_description,
        payload.turned_down_work, payload.utilization_rate, payload.can_take_work_tomorrow,
        payload.average_job_time_hours, payload.travel_time_minutes, payload.scheduling_efficiency,
        payload.time_loss_source, payload.admin_drag,
        payload.monthly_revenue, payload.monthly_revenue_range, payload.labor_cost_pct,
        payload.profitability_feeling, payload.gross_margin_pct, payload.pricing_adequacy,
        payload.cash_stress,
    ]
    filled = sum(1 for x in structured_fields if x not in (None, "", 0))
    return filled / len(structured_fields)


def resolve_analysis_mode(payload: AssessmentInput) -> str:
    requested_mode = (payload.analysis_mode or "auto").strip().lower()
    notes = (payload.notes or "").strip()
    notes_len = len(notes)
    completeness = get_structured_input_completeness(payload)

    if requested_mode in {"notes_led", "hybrid", "structured"}:
        return requested_mode

    if notes_len >= 250 and completeness < 0.35:
        return "notes_led"
    if completeness >= 0.60:
        return "structured"
    return "hybrid"


def is_notes_dominant_mode(payload: AssessmentInput) -> bool:
    requested_mode = (payload.analysis_mode or "auto").strip().lower()
    notes = (payload.notes or "").strip()
    notes_len = len(notes)
    completeness = get_structured_input_completeness(payload)
    resolved_mode = resolve_analysis_mode(payload)

    if resolved_mode == "notes_led":
        return bool(notes)

    # Only auto mode is allowed to infer notes-led behavior.
    # Manual structured/hybrid overrides must be respected.
    if requested_mode == "auto":
        return notes_len >= 250 and completeness < 0.35

    return False


def should_ignore_blank_structured_fields(payload: AssessmentInput) -> bool:
    return resolve_analysis_mode(payload) == "notes_led"


def count_present(values: list[object]) -> int:
    return sum(1 for x in values if x not in (None, "", 0))


def build_score_statuses(payload: AssessmentInput) -> Dict[str, str]:
    if should_ignore_blank_structured_fields(payload) or is_notes_dominant_mode(payload):
        return {
            "demand": "insufficient_data",
            "conversion": "insufficient_data",
            "capacity": "insufficient_data",
            "operations": "insufficient_data",
            "financials": "insufficient_data",
        }

    category_inputs = {
        "demand": {
            "all": [
                payload.leads_per_week, payload.leads_estimate, payload.schedule_fullness,
                payload.backlog_days, payload.backlog_bucket,
            ],
            "direct": [payload.leads_per_week, payload.backlog_days],
        },
        "conversion": {
            "all": [
                payload.conversion_rate, payload.quotes_to_jobs_bucket,
                payload.sales_behavior, payload.response_speed,
            ],
            "direct": [payload.conversion_rate],
        },
        "capacity": {
            "all": [
                payload.crews, payload.jobs_per_crew_per_day, payload.busy_day_description,
                payload.turned_down_work, payload.utilization_rate,
                payload.can_take_work_tomorrow, payload.backlog_days, payload.backlog_bucket,
            ],
            "direct": [payload.crews, payload.jobs_per_crew_per_day, payload.utilization_rate],
        },
        "operations": {
            "all": [
                payload.jobs_per_crew_per_day, payload.travel_time_minutes,
                payload.scheduling_efficiency, payload.time_loss_source,
                payload.admin_drag, payload.average_job_time_hours,
            ],
            "direct": [payload.jobs_per_crew_per_day, payload.travel_time_minutes, payload.average_job_time_hours],
        },
        "financials": {
            "all": [
                payload.monthly_revenue, payload.monthly_revenue_range, payload.labor_cost_pct,
                payload.profitability_feeling, payload.gross_margin_pct,
                payload.pricing_adequacy, payload.cash_stress,
            ],
            "direct": [payload.monthly_revenue, payload.labor_cost_pct, payload.gross_margin_pct],
        },
    }

    statuses: Dict[str, str] = {}
    for category, parts in category_inputs.items():
        all_count = count_present(parts["all"])
        direct_count = count_present(parts["direct"])

        if all_count < 2:
            statuses[category] = "insufficient_data"
        elif direct_count >= 1:
            statuses[category] = "measured"
        else:
            statuses[category] = "estimated"

    return statuses


def sanitize_scores_for_display(scores: Dict[str, int], score_statuses: Dict[str, str]) -> Dict[str, Optional[int]]:
    return {
        category: (score if score_statuses.get(category) != "insufficient_data" else None)
        for category, score in scores.items()
    }


def build_score_summary_note(payload: AssessmentInput, score_statuses: Dict[str, str]) -> str:
    if should_ignore_blank_structured_fields(payload):
        return "This assessment is running in notes-led mode, so blank structured fields are being ignored and category scores are being treated as unavailable unless directly supported elsewhere."
    if is_notes_dominant_mode(payload):
        return "Limited structured input was provided, so category scores are being treated as unavailable while the report relies primarily on the notes field."

    insufficient = sum(1 for status in score_statuses.values() if status == "insufficient_data")
    estimated = sum(1 for status in score_statuses.values() if status == "estimated")

    if insufficient:
        return f"{insufficient} category score(s) are unavailable because there was not enough structured data to support them."
    if estimated:
        return f"{estimated} category score(s) are estimated from partial structured inputs rather than directly measured."
    return "Category scores are supported by enough structured input to be treated as reasonably grounded."


def score_from_bucket(value: Optional[str], mapping: Dict[str, int], default: int = 50) -> int:
    if not value:
        return default
    return mapping.get(value, default)


def clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))


def score_leads(payload: AssessmentInput) -> int:
    if payload.leads_per_week is not None:
        x = payload.leads_per_week
        if x < 5:
            return 20
        if x < 15:
            return 40
        if x < 30:
            return 65
        if x < 50:
            return 82
        return 95
    return score_from_bucket(payload.leads_estimate, {"0-5": 20, "5-15": 40, "15-30": 65, "30+": 90})


def score_backlog(payload: AssessmentInput) -> int:
    if payload.backlog_days is not None:
        x = payload.backlog_days
        if x <= 1:
            return 25
        if x <= 3:
            return 45
        if x <= 7:
            return 65
        if x <= 14:
            return 82
        return 95
    return score_from_bucket(payload.backlog_bucket, {"same_day": 25, "2_3_days": 45, "4_7_days": 65, "1_2_weeks": 82, "2_plus_weeks": 95})


def score_schedule_fullness(payload: AssessmentInput) -> int:
    return score_from_bucket(payload.schedule_fullness, {"empty_slots": 25, "mixed": 55, "mostly_full": 80, "overbooked": 95})


def score_conversion(payload: AssessmentInput) -> int:
    if payload.conversion_rate is not None:
        x = payload.conversion_rate
        if x < 20:
            return 20
        if x < 30:
            return 40
        if x < 50:
            return 70
        if x < 70:
            return 90
        return 100
    bucket_score = score_from_bucket(payload.quotes_to_jobs_bucket, {"1_2": 25, "3_4": 45, "5_7": 75, "8_10": 95}, 60)
    behavior_score = score_from_bucket(payload.sales_behavior, {"lose_most": 25, "win_half": 55, "win_most": 80, "almost_everyone": 95}, 60)
    return round((bucket_score * 0.6) + (behavior_score * 0.4))


def score_response_speed(payload: AssessmentInput) -> int:
    return score_from_bucket(payload.response_speed, {"same_hour": 95, "same_day": 80, "next_day": 55, "slow": 30})


def score_utilization(payload: AssessmentInput) -> int:
    if payload.utilization_rate is not None:
        x = payload.utilization_rate
        if x < 50:
            return 25
        if x < 65:
            return 45
        if x < 80:
            return 70
        if x < 90:
            return 88
        return 95
    return score_from_bucket(payload.busy_day_description, {"fully_booked": 90, "mostly_busy": 70, "frequent_downtime": 40, "inconsistent": 30})


def score_turned_down_work(payload: AssessmentInput) -> int:
    return score_from_bucket(payload.turned_down_work, {"never": 25, "occasionally": 55, "frequently": 80, "all_the_time": 95})


def score_can_take_work_tomorrow(payload: AssessmentInput) -> int:
    return score_from_bucket(payload.can_take_work_tomorrow, {"yes_easily": 20, "somewhat": 45, "tight": 75, "no": 95})


def score_jobs_per_day(payload: AssessmentInput) -> int:
    if payload.jobs_per_crew_per_day is None:
        return 55
    x = payload.jobs_per_crew_per_day
    if x < 2:
        return 20
    if x < 4:
        return 45
    if x < 6:
        return 70
    if x < 8:
        return 85
    return 95


def score_travel_time(payload: AssessmentInput) -> int:
    if payload.travel_time_minutes is None:
        return score_from_bucket(payload.time_loss_source, {"driving": 30, "jobs_run_long": 45, "waiting": 40, "admin": 50}, 55)
    x = payload.travel_time_minutes
    if x > 45:
        return 20
    if x > 30:
        return 40
    if x > 20:
        return 60
    if x > 10:
        return 80
    return 95


def score_scheduling(payload: AssessmentInput) -> int:
    return score_from_bucket(payload.scheduling_efficiency, {"very_smooth": 95, "mostly_smooth": 75, "some_issues": 50, "chaotic": 25}, 60)


def score_admin_drag(payload: AssessmentInput) -> int:
    return score_from_bucket(payload.admin_drag, {"low": 90, "moderate": 60, "high": 30}, 60)


def score_labor_cost(payload: AssessmentInput) -> int:
    if payload.labor_cost_pct is not None:
        x = payload.labor_cost_pct
        if x < 35:
            return 90
        if x < 45:
            return 75
        if x < 55:
            return 60
        if x < 65:
            return 40
        return 20
    return 50


def score_margin(payload: AssessmentInput) -> int:
    if payload.gross_margin_pct is not None:
        x = payload.gross_margin_pct
        if x < 15:
            return 20
        if x < 25:
            return 40
        if x < 35:
            return 60
        if x < 50:
            return 80
        return 95
    return 50


def score_profitability_feeling(payload: AssessmentInput) -> int:
    return score_from_bucket(payload.profitability_feeling, {"comfortable": 90, "tight": 60, "breaking_even": 35, "struggling": 20}, 55)


def score_pricing(payload: AssessmentInput) -> int:
    return score_from_bucket(payload.pricing_adequacy, {"strong": 90, "okay": 65, "low": 30, "unclear": 50}, 50)


def score_cash_stress(payload: AssessmentInput) -> int:
    return score_from_bucket(payload.cash_stress, {"low": 90, "moderate": 60, "high": 25}, 55)


def detect_assumptions(payload: AssessmentInput) -> list[str]:
    assumptions = []
    if payload.conversion_rate is None and not payload.quotes_to_jobs_bucket and not payload.sales_behavior:
        assumptions.append("Conversion signal is limited and partially inferred.")
    if payload.labor_cost_pct is None:
        assumptions.append("Labor cost was not provided directly.")
    if payload.gross_margin_pct is None:
        assumptions.append("Gross margin was not provided directly.")
    if payload.backlog_days is None and not payload.backlog_bucket:
        assumptions.append("Backlog was estimated from qualitative inputs.")

    # Intelligence-layer upgrades
    if payload.mode in {"guided", "hybrid"}:
        assumptions.append("Some inputs were estimated due to limited tracking systems.")
    if payload.leads_estimate and payload.leads_per_week is None:
        assumptions.append("Lead volume was estimated instead of measured directly.")
    if payload.backlog_bucket and payload.backlog_days is None:
        assumptions.append("Backlog timing was estimated from a booked-out range.")
    if payload.quotes_to_jobs_bucket and payload.conversion_rate is None:
        assumptions.append("Conversion was estimated from inquiry-to-booking buckets.")
    if payload.sales_behavior and payload.conversion_rate is None:
        assumptions.append("Conversion was inferred from sales behavior rather than a measured conversion rate.")
    if payload.busy_day_description and payload.utilization_rate is None:
        assumptions.append("Crew utilization was inferred from workload descriptions.")
    if payload.busy_day_description and payload.jobs_per_crew_per_day is None:
        assumptions.append("Capacity was estimated without exact jobs-per-day data.")
    if payload.monthly_revenue_range and payload.monthly_revenue is None:
        assumptions.append("Revenue was estimated from a range instead of exact figures.")
    if payload.time_loss_source and payload.travel_time_minutes is None:
        assumptions.append("Operational drag was inferred from the stated main time-loss source.")
    if payload.pricing_adequacy in {"unclear", None}:
        assumptions.append("Pricing strength is uncertain and may need direct validation.")

    deduped = []
    for item in assumptions:
        if item not in deduped:
            deduped.append(item)
    return deduped


def confidence_label(score: int) -> str:
    if score >= 85:
        return "High"
    if score >= 65:
        return "Medium"
    if score >= 45:
        return "Low"
    return "Exploratory"


def build_confidence_explanation(scores: Dict[str, int], assumptions: list[str], contradictions: list[str], no_constraint: bool, notes_dominant_mode: bool = False) -> str:
    reasons = []
    numeric_scores = {
        key: value
        for key, value in scores.items()
        if isinstance(value, (int, float))
    }

    if notes_dominant_mode:
        reasons.append("the report relies primarily on narrative notes because structured intake data was sparse")
    if assumptions:
        reasons.append(f"{len(assumptions)} assumption(s) were needed because some inputs were estimated or missing")
    if contradictions:
        reasons.append(f"{len(contradictions)} contradiction(s) reduced confidence in the consistency of the inputs")
    if no_constraint and not notes_dominant_mode and numeric_scores:
        reasons.append("scores were strong and tightly grouped across categories")
    elif not no_constraint and not notes_dominant_mode and len(numeric_scores) > 1:
        ordered = sorted(numeric_scores.items(), key=lambda x: x[1])
        gap = ordered[1][1] - ordered[0][1]
        reasons.append(f"the gap between the weakest and next-weakest category was {gap} point(s)")
    elif not numeric_scores:
        reasons.append("category scores were unavailable or intentionally suppressed for this assessment")
    if not reasons:
        return "Confidence is supported by fairly complete and internally consistent inputs."
    return "Confidence is influenced by " + "; ".join(reasons) + "."


def detect_contradictions(payload: AssessmentInput, scores: Dict[str, int]) -> list[str]:
    contradictions = []
    if payload.schedule_fullness == "overbooked" and payload.busy_day_description in {"frequent_downtime", "inconsistent"}:
        contradictions.append("Schedule appears overbooked, but daily capacity usage sounds inconsistent.")
    if payload.backlog_bucket in {"1_2_weeks", "2_plus_weeks"} and payload.can_take_work_tomorrow == "yes_easily":
        contradictions.append("Long backlog conflicts with ability to take work immediately.")
    if scores["demand"] > 75 and scores["capacity"] > 75 and scores["operations"] < 40:
        contradictions.append("Strong demand and capacity signals conflict with weak operational efficiency.")

    # Intelligence-layer upgrades
    if payload.turned_down_work in {"frequently", "all_the_time"} and payload.can_take_work_tomorrow == "yes_easily":
        contradictions.append("The business is turning away work despite reporting available capacity tomorrow.")
    if payload.sales_behavior in {"win_most", "almost_everyone"} and payload.conversion_rate is not None and payload.conversion_rate < 40:
        contradictions.append("Sales behavior suggests strong close rates, but the reported conversion rate is weak.")
    if payload.profitability_feeling == "comfortable" and payload.cash_stress == "high":
        contradictions.append("Profitability is described as comfortable, but cash stress is reported as high.")
    if payload.pricing_adequacy == "strong" and payload.profitability_feeling in {"breaking_even", "struggling"}:
        contradictions.append("Pricing is described as strong, yet profitability appears weak.")
    if payload.schedule_fullness == "empty_slots" and payload.turned_down_work in {"frequently", "all_the_time"}:
        contradictions.append("The schedule has open capacity, but the business also reports turning away work.")

    deduped = []
    for item in contradictions:
        if item not in deduped:
            deduped.append(item)
    return deduped


def calculate_scores(payload: AssessmentInput) -> Dict[str, Any]:
    demand = clamp_score((score_leads(payload) * 0.30) + (score_backlog(payload) * 0.35) + (score_schedule_fullness(payload) * 0.35))
    conversion = clamp_score((score_conversion(payload) * 0.80) + (score_response_speed(payload) * 0.20))
    capacity = clamp_score((score_utilization(payload) * 0.35) + (score_backlog(payload) * 0.25) + (score_turned_down_work(payload) * 0.20) + (score_can_take_work_tomorrow(payload) * 0.20))
    operations = clamp_score((score_jobs_per_day(payload) * 0.30) + (score_travel_time(payload) * 0.25) + (score_scheduling(payload) * 0.25) + (score_admin_drag(payload) * 0.20))
    financials = clamp_score((score_labor_cost(payload) * 0.25) + (score_margin(payload) * 0.25) + (score_profitability_feeling(payload) * 0.20) + (score_pricing(payload) * 0.15) + (score_cash_stress(payload) * 0.15))

    scores = {"demand": demand, "conversion": conversion, "capacity": capacity, "operations": operations, "financials": financials}
    sorted_scores = sorted(scores.items(), key=lambda item: item[1])

    min_score = sorted_scores[0][1]
    max_score = sorted_scores[-1][1]
    score_spread = max_score - min_score
    notes_dominant_mode = is_notes_dominant_mode(payload)
    notes_led_mode = should_ignore_blank_structured_fields(payload)
    no_constraint = min_score >= 75 and score_spread <= 15

    if notes_led_mode or notes_dominant_mode:
        primary_constraint = "None"
        secondary_constraint = None
        no_constraint = True
    elif no_constraint:
        primary_constraint = "None"
        secondary_constraint = None
    else:
        primary_constraint = sorted_scores[0][0].capitalize()
        secondary_constraint = None
        if (sorted_scores[1][1] - sorted_scores[0][1]) <= 10:
            secondary_constraint = sorted_scores[1][0].capitalize()

    contradictions = detect_contradictions(payload, scores)
    assumptions = detect_assumptions(payload)

    completeness_fields = [
        payload.business_type, payload.location,
        payload.leads_per_week or payload.leads_estimate or payload.schedule_fullness,
        payload.conversion_rate or payload.quotes_to_jobs_bucket or payload.sales_behavior,
        payload.crews or payload.busy_day_description,
        payload.scheduling_efficiency or payload.travel_time_minutes,
        payload.monthly_revenue or payload.monthly_revenue_range or payload.profitability_feeling,
    ]
    completeness_ratio = sum(1 for x in completeness_fields if x not in (None, "", 0)) / len(completeness_fields)
    completeness_score = round(completeness_ratio * 100)

    if len(sorted_scores) > 1:
        score_gap = sorted_scores[1][1] - sorted_scores[0][1]
    else:
        score_gap = 25
    separation_score = 95 if score_gap >= 15 else 80 if score_gap >= 10 else 65 if score_gap >= 6 else 45 if score_gap >= 3 else 25
    contradiction_penalty = min(len(contradictions) * 10, 30)
    overall_confidence_score = max(20, round((completeness_score * 0.5) + (separation_score * 0.5) - contradiction_penalty))

    if notes_led_mode:
        strategic_state = "NOTES_LED"
        constraint_pattern = "The assessment is intentionally running in notes-led mode, so blank structured fields are being ignored and the narrative notes are the primary source of truth."
    elif notes_dominant_mode:
        strategic_state = "NOTES_LED"
        constraint_pattern = "The assessment is being driven primarily by the narrative notes rather than a complete structured intake, so any diagnosis should be treated as notes-led and directional."
    elif no_constraint:
        if financials >= 85:
            strategic_state = "EXIT_READY"
            constraint_pattern = "The business appears healthy, stable, and financially attractive enough to evaluate exit or acquisition opportunities."
        else:
            strategic_state = "SCALE_READY"
            constraint_pattern = "The business appears healthy across core dimensions and may be better served by scale, expansion, or strategic optimization than bottleneck fixing."
    else:
        strategic_state = "OPTIMIZATION"
        patterns = {
            "Demand": "Inconsistent lead flow limiting utilization",
            "Conversion": "Healthy inquiries but weak close effectiveness",
            "Capacity": "Demand exceeds fulfillment capacity",
            "Operations": "Scheduling and delivery inefficiency reducing output",
            "Financials": "Margin pressure limiting healthy growth",
        }
        constraint_pattern = patterns[primary_constraint]

    category_confidence = {"demand": 70, "conversion": 72, "capacity": 75, "operations": 68, "financials": 65}
    score_statuses = build_score_statuses(payload)
    display_scores = sanitize_scores_for_display(scores, score_statuses)
    score_summary_note = build_score_summary_note(payload, score_statuses)
    confidence_explanation = build_confidence_explanation(scores, assumptions, contradictions, no_constraint, notes_dominant_mode)

    return {
        "scores": display_scores,
        "raw_scores": scores,
        "score_statuses": score_statuses,
        "score_summary_note": score_summary_note,
        "category_confidence": category_confidence,
        "primary_constraint": primary_constraint,
        "secondary_constraint": secondary_constraint,
        "constraint_pattern": constraint_pattern,
        "overall_confidence_score": overall_confidence_score,
        "overall_confidence_label": confidence_label(overall_confidence_score),
        "confidence_explanation": confidence_explanation,
        "assumptions": assumptions,
        "contradictions": contradictions,
        "no_constraint": no_constraint,
        "strategic_state": strategic_state,
    }


def build_business_type_guidance(business_type: str) -> str:
    bt = (business_type or "").strip().lower()

    keyword_groups = {
        "cleaning": ["cleaning", "janitorial", "maid", "housekeeping"],
        "lawn care": ["lawn", "landscaping", "landscape", "mowing", "yard"],
        "glass": ["glass", "mirror", "glazing", "window", "shower door"],
        "home services": ["hvac", "plumbing", "electric", "electrical", "roofing", "painting", "remodel", "garage door", "pest", "appliance", "flooring"],
        "professional services": ["consulting", "agency", "accounting", "bookkeeping", "legal", "marketing", "design", "it services", "financial advisory"],
    }

    detected = None
    for label, keywords in keyword_groups.items():
        if any(k in bt for k in keywords):
            detected = label
            break

    guidance = {
        "cleaning": [
            "Tailor actions to route density, recurring schedules, crew utilization, travel gaps, upsells, and quote follow-up.",
            "Prefer recommendations involving recurring revenue retention, zone-based scheduling, add-on services, and cleaner productivity.",
        ],
        "lawn care": [
            "Tailor actions to route density, seasonal demand swings, recurring maintenance plans, crew productivity, and weather disruption.",
            "Prefer recommendations involving route clustering, maintenance contract retention, upsells, and capacity planning during peak season.",
        ],
        "glass": [
            "Tailor actions to estimating speed, site measurement quality, install scheduling, job handoff quality, material coordination, and margin control.",
            "Prefer recommendations involving separating measurement from installation bottlenecks, tighter quote-to-install workflow, and job profitability discipline.",
        ],
        "home services": [
            "Tailor actions to dispatching, technician utilization, callback reduction, service-to-sale conversion, travel efficiency, and price discipline.",
            "Prefer recommendations involving route efficiency, faster lead response, diagnostic consistency, field productivity, and ticket-value expansion.",
        ],
        "professional services": [
            "Tailor actions to pipeline conversion, proposal follow-up, utilization, project scoping, delivery efficiency, and pricing discipline.",
            "Prefer recommendations involving tightening the sales process, reducing scope creep, improving utilization, and packaging services more profitably.",
        ],
        None: [
            "Tailor actions to the stated business type rather than giving generic service-business advice.",
            "Make the recommendations feel operationally realistic for that business model.",
        ],
    }

    lines = guidance[detected]
    detected_label = detected.title() if detected else "General Service Business"
    return "\n".join([
        f"Business-Type Context: {detected_label}",
        f"- {lines[0]}",
        f"- {lines[1]}",
    ])


SYSTEM_PROMPT = """
You are a high-level business consultant specializing in service businesses.

You are given:
- structured diagnostic scores
- identified constraint OR strategic state

Your job is to:
1. Clearly explain the situation
2. Justify it using the data
3. Translate it into business impact
4. Provide SPECIFIC, ACTIONABLE steps

CRITICAL RULES:
- If a constraint exists, focus only on that constraint
- If there is no constraint, shift to strategy: scale, optimize, or exit
- Recommendations must be practical and implementable
- Avoid vague advice like 'improve marketing'
- Tie every recommendation to a direct outcome
- Prioritize actions clearly
- Think like a consultant, not a chatbot

REPORT QUALITY RULES:
- Preserve the numbered section structure
- Preserve the separated report header
- Write in a polished, consultant-grade style
- Use stronger reasoning tied to the evidence
- Complete every section fully
- Do not end mid-sentence
""".strip()


def build_claude_user_prompt(payload: AssessmentInput, result: dict[str, Any]) -> str:
    evidence = [
        f"Business Type: {payload.business_type}",
        f"Location: {payload.location}",
        f"Mode: {payload.mode}",
        f"Requested Analysis Mode: {payload.analysis_mode}",
        f"Resolved Analysis Mode: {resolve_analysis_mode(payload)}",
        f"Notes-Led Mode: {should_ignore_blank_structured_fields(payload)}",
        "",
        "SCORES:",
        f"- Demand: {result['scores']['demand']} ({result.get('score_statuses', {}).get('demand', 'unknown')})",
        f"- Conversion: {result['scores']['conversion']} ({result.get('score_statuses', {}).get('conversion', 'unknown')})",
        f"- Capacity: {result['scores']['capacity']} ({result.get('score_statuses', {}).get('capacity', 'unknown')})",
        f"- Operations: {result['scores']['operations']} ({result.get('score_statuses', {}).get('operations', 'unknown')})",
        f"- Financials: {result['scores']['financials']} ({result.get('score_statuses', {}).get('financials', 'unknown')})",
        f"- Score Summary Note: {result.get('score_summary_note', '')}",
        "",
        f"No Constraint: {result.get('no_constraint')}",
        f"Strategic State: {result.get('strategic_state')}",
        f"Primary Constraint: {result['primary_constraint']}",
        f"Secondary Constraint: {result.get('secondary_constraint') or 'None'}",
        f"Constraint Pattern: {result.get('constraint_pattern') or 'None'}",
        f"Confidence: {result.get('overall_confidence_label', 'Unknown')} ({result.get('overall_confidence_score', 0)}/100)",
        f"Confidence Explanation: {result.get('confidence_explanation', '')}",
        "",
        "Narrative Notes:",
        payload.notes or "None provided",
        "",
        "Key Evidence:",
        f"- Schedule fullness: {payload.schedule_fullness}",
        f"- Backlog bucket: {payload.backlog_bucket}",
        f"- Conversion rate: {payload.conversion_rate}",
        f"- Sales behavior: {payload.sales_behavior}",
        f"- Crews: {payload.crews}",
        f"- Busy day description: {payload.busy_day_description}",
        f"- Turned down work: {payload.turned_down_work}",
        f"- Scheduling efficiency: {payload.scheduling_efficiency}",
        f"- Monthly revenue range: {payload.monthly_revenue_range}",
        f"- Profitability feeling: {payload.profitability_feeling}",
        f"- Pricing adequacy: {payload.pricing_adequacy}",
        f"- Cash stress: {payload.cash_stress}",
        "",
        "Assumptions:",
    ]
    assumptions = result.get("assumptions", [])
    evidence.extend([f"- {a}" for a in assumptions] if assumptions else ["- None"])
    evidence.extend([
        "",
        build_business_type_guidance(payload.business_type),
        "",
        "Write a client-facing consulting report in plain markdown with this exact high-level shape:",
        "Start with a divided header block:",
        "Line 1: Business name line like '<Business Type> Business Consulting Report'",
        "Line 2: Location and mode line like '<Location> | <Mode label>'",
        "",
        "Then write these numbered sections exactly in this order:",
        "1. Business Snapshot",
        "2. Diagnosis: <Primary Constraint or Strategic State>",
        "3. Why This Was Identified",
        "4. What This Means for the Business",
        "5. Risk of Inaction",
        "6. Action Plan",
        "",
        "Section rules:",
        "- Keep the numbering format exactly",
        "- Keep the tone executive and consultant-grade",
        "- Use short paragraphs and readable spacing",
        "- In section 3, use strong reasoning tied to scores and evidence",
        "- In section 6, write 3-5 action items with these sublabels:",
        "  Action X: <short title>",
        "  Time to Impact: Immediate | 30 Days | 90 Days",
        "  Effort Level: Low | Medium | High",
        "  Expected Impact: High | Medium | Foundational",
        "  What to do:",
        "  Why it matters:",
        "  Expected outcome:",
        "- Make the actions specific enough that an owner could begin execution this week",
        "- Sequence the actions from highest leverage to lowest leverage",
        "- Avoid generic advice like improve operations, improve marketing, or optimize pricing",
        "- Use concrete actions tied directly to the diagnosed constraint or strategic state",
        "- Make the action plan feel tailored to the business type, not interchangeable with other industries",
        "- If hard financial metrics are missing, do not guess ROI or dollar returns",
        "- When a category score is marked insufficient_data, treat it as unavailable rather than average or weak",
        "- Do not interpret blank categories as 50s or as evidence of mediocre performance",
        "- When data is qualitative or limited, still include Time to Impact and Effort Level, but keep Expected Impact directional rather than overly precise",
        "- When data is stronger, you may make Expected Impact more confident, but do not fabricate numbers",
        "- Keep sections 1 through 5 concise so section 6 has enough room for complete action plans",
        "- In section 5 or section 6, you may reference contradictions when they materially affect the recommendation",
        "- In the final lines of the report, briefly explain confidence using the provided confidence explanation",
        "- When Resolved Analysis Mode is notes_led, use the notes field as the primary source of truth for diagnosis and recommendations",
        "- When Resolved Analysis Mode is notes_led, ignore blank structured fields rather than treating them as missing, average, or weak business signals",
        "- When Resolved Analysis Mode is notes_led, do not infer a primary constraint from absent structured inputs",
        "- When Resolved Analysis Mode is notes_led, build the action plan from the narrative notes and any explicitly stated facts only",
        "",
        "Do not add new numbered sections.",
        "Do not use tables.",
        "Do not use JSON.",
        "Do not end mid-sentence.",
    ])
    return "\n".join(evidence)


def fallback_report_markdown(payload: AssessmentInput, result: Dict[str, Any], reason: str = "") -> str:
    reason_line = f"\n\n_Fallback reason: {reason}_" if reason else ""
    diagnosis = result["primary_constraint"] if not result.get("no_constraint") else result.get("strategic_state", "SCALE_READY")
    notes_led_mode = should_ignore_blank_structured_fields(payload)
    location_line = payload.location or "Location not provided"
    mode_line = f"{payload.mode.title()} Operations Model"

    return f"""
{payload.business_type} Business Consulting Report

{location_line} | {mode_line}

1. Business Snapshot

{"This assessment is intentionally running in notes-led mode, so the report is based primarily on the narrative notes and blank structured fields are being ignored." if notes_led_mode else f"This business is currently in a **{diagnosis}** state. The available inputs suggest that the most important issue should be addressed before broader growth efforts continue."}

2. Diagnosis: {diagnosis}

{"No formal structured-data constraint is being asserted because this assessment is running in notes-led mode. The diagnosis should be read as a narrative interpretation of the notes provided." if notes_led_mode else f"The model identified **{diagnosis}** as the main business issue based on the submitted operating and financial signals. The current pattern is: {result['constraint_pattern']}"}

3. Why This Was Identified

The scoring model reviewed demand, conversion, capacity, operations, and financial performance together. The lowest area or strategic-state logic determined the current diagnosis. The available evidence was strong enough to produce a directional recommendation, though confidence depends on the quality and completeness of the intake.

4. What This Means for the Business

This means the business is likely underperforming not because every part of the company is broken, but because one core issue is limiting broader performance. Until that issue is addressed, growth, profitability, or operational stability may remain constrained.

5. Risk of Inaction

If this issue is not addressed, the business may continue operating below its potential, absorb unnecessary strain, and miss the window to improve performance before the next growth cycle or seasonal shift.

6. Action Plan

Action 1: Stabilize the primary constraint  
Time to Impact: Immediate  
Effort Level: Medium  
Expected Impact: High  
What to do: Make one direct operating change within the next 7 days that targets the diagnosed bottleneck, such as tightening scheduling rules, raising price floors, or enforcing faster lead follow-up.  
Why it matters: Focused intervention on the true bottleneck creates faster results than spreading effort across multiple lower-value initiatives.  
Expected outcome: Clearer movement in throughput, close rate, booked-out time, or margin within the next two weeks.

Action 2: Convert the fix into a repeatable system  
Time to Impact: 30 Days  
Effort Level: Medium  
Expected Impact: High  
What to do: Turn the chosen fix into a documented workflow, rule, script, pricing standard, or operating routine the team can execute consistently.  
Why it matters: One-time effort creates temporary relief, but repeatable systems create sustained business improvement.  
Expected outcome: More stable execution with less owner dependence and fewer performance swings.

Action 3: Track one leading indicator every week  
Time to Impact: 30 Days  
Effort Level: Low  
Expected Impact: Foundational  
What to do: Choose the one metric most tied to the diagnosis, such as conversion rate, jobs per day, booked-out time, or revenue per job, and review it weekly.  
Why it matters: Measurement shows whether the intervention is working and helps the business correct course before issues compound.  
Expected outcome: Better decisions, faster iteration, and improved confidence in what is actually driving performance.

Assumptions: {", ".join(result["assumptions"]) if result["assumptions"] else "None"}  
Contradictions: {", ".join(result["contradictions"]) if result["contradictions"] else "None"}  
Confidence: {result["overall_confidence_label"]}  
Confidence Explanation: {result.get("confidence_explanation", "Confidence is supported by the available inputs.")}{reason_line}
""".strip()


def report_looks_incomplete(report: str) -> bool:
    if not report:
        return True

    stripped = report.strip()

    if len(stripped) < 700:
        return True

    bad_endings = (
        "You",
        "What to do:",
        "Why it matters:",
        "Expected outcome:",
        "Action 1:",
        "Action 2:",
        "Action 3:",
        "Action 4:",
        "Action 5:",
    )

    if any(stripped.endswith(x) for x in bad_endings):
        return True

    return False


def is_claude_overloaded_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "overloaded_error" in text or "overloaded" in text


def calculate_retry_delay(attempt: int, overloaded: bool = False) -> float:
    # Exponential backoff with jitter. Overload errors need longer spacing.
    base = 3.0 if overloaded else 1.75
    delay = min(30.0, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0.25, 1.25)
    return delay + jitter



def build_report_markdown(payload: AssessmentInput, result: Dict[str, Any]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return fallback_report_markdown(payload, result, "Missing ANTHROPIC_API_KEY")

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    client = anthropic.Anthropic(api_key=api_key)

    prompt = build_claude_user_prompt(payload, result)
    last_error = ""

    max_attempts = 7

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"=== CLAUDE CALL START attempt {attempt}/{max_attempts} ===")
            with client.messages.stream(
                model=model,
                max_tokens=3000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()

            parts = []
            for block in message.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)

            report = "\n\n".join(parts).strip()
            print(f"=== CLAUDE RESPONSE LENGTH attempt {attempt}/{max_attempts}: {len(report)} ===")
            print("=== REPORT LENGTH ===", len(report))

            if report and not report_looks_incomplete(report):
                print("=== USING CLAUDE LIVE OUTPUT ===")
                return f"# CLAUDE LIVE\n\n{report}"

            last_error = f"Incomplete Claude response on attempt {attempt}"
            print(f"=== RETRYING: {last_error} ===")
            if attempt < max_attempts:
                delay = calculate_retry_delay(attempt, overloaded=False)
                print(f"=== WAITING {delay:.1f}s BEFORE NEXT CLAUDE ATTEMPT ===")
                time.sleep(delay)

        except Exception as exc:
            overloaded = is_claude_overloaded_error(exc)
            error_type = "Claude overloaded" if overloaded else "Claude error"
            last_error = f"{error_type} on attempt {attempt}: {exc}"
            print(f"=== RETRYING: {last_error} ===")
            if attempt < max_attempts:
                delay = calculate_retry_delay(attempt, overloaded=overloaded)
                print(f"=== WAITING {delay:.1f}s BEFORE NEXT CLAUDE ATTEMPT ===")
                time.sleep(delay)

    return fallback_report_markdown(payload, result, last_error or "Claude retries exhausted")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": "AI Business Diagnostic API",
        "model_version": "v2",
        "storage": "sqlite_enabled",
        "claude_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "anthropic_model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "prompt_version": os.getenv("PROMPT_VERSION", "report_v2_strategic"),
        "analysis_modes_supported": ["auto", "structured", "hybrid", "notes_led"],
    }


@app.post("/api/v1/assessments/score")
def score_assessment(payload: AssessmentInput) -> dict[str, Any]:
    result = calculate_scores(payload)
    return {"assessment_id": f"preview-{uuid4()}", **result, "report_markdown": ""}


@app.post("/api/v1/assessments/report")
def create_report_route(payload: AssessmentInput, db: Session = Depends(get_db)) -> dict[str, Any]:
    result = calculate_scores(payload)
    report_markdown = build_report_markdown(payload, result)
    assessment = create_assessment(
        db,
        payload_dict=payload.model_dump(),
        result=result,
        report_markdown=report_markdown,
        scoring_model_version="v2",
        prompt_version=os.getenv("PROMPT_VERSION", "report_v2_strategic"),
    )
    latest_report = assessment.reports[-1]
    return {"assessment_id": assessment.id, **result, "report_markdown": latest_report.report_markdown}


@app.get("/api/v1/assessments/{assessment_id}")
def get_assessment_route(assessment_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    assessment = get_assessment_by_id(db, assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Not Found")
    latest_report = assessment.reports[-1] if assessment.reports else None

    no_constraint = assessment.primary_constraint == "None"
    strategic_state = "OPTIMIZATION"
    if no_constraint:
        if assessment.constraint_pattern and "notes-led mode" in assessment.constraint_pattern.lower():
            strategic_state = "NOTES_LED"
        elif assessment.constraint_pattern and "notes" in assessment.constraint_pattern.lower():
            strategic_state = "NOTES_LED"
        elif assessment.constraint_pattern and "exit" in assessment.constraint_pattern.lower():
            strategic_state = "EXIT_READY"
        else:
            strategic_state = "SCALE_READY"

    raw_scores = json.loads(assessment.scores_json)
    assumptions = json.loads(assessment.assumptions_json)
    contradictions = json.loads(assessment.contradictions_json)
    notes_dominant_mode = "notes" in (assessment.constraint_pattern or "").lower() or strategic_state == "NOTES_LED"
    score_statuses = {k: ("insufficient_data" if notes_dominant_mode else "estimated") for k in raw_scores.keys()}
    display_scores = {k: (None if v == "insufficient_data" else raw_scores[k]) for k, v in score_statuses.items()}
    score_summary_note = "Limited structured input was provided, so category scores are being treated as unavailable while the report relies primarily on the notes field." if notes_dominant_mode else "Stored category scores may include estimated values from partial structured inputs."
    confidence_explanation = build_confidence_explanation(raw_scores, assumptions, contradictions, no_constraint, notes_dominant_mode)

    return {
        "assessment_id": assessment.id,
        "scores": display_scores,
        "raw_scores": raw_scores,
        "score_statuses": score_statuses,
        "score_summary_note": score_summary_note,
        "category_confidence": json.loads(assessment.category_confidence_json),
        "primary_constraint": assessment.primary_constraint,
        "secondary_constraint": assessment.secondary_constraint,
        "constraint_pattern": assessment.constraint_pattern,
        "overall_confidence_score": assessment.overall_confidence_score,
        "overall_confidence_label": assessment.overall_confidence_label,
        "confidence_explanation": confidence_explanation,
        "assumptions": assumptions,
        "contradictions": contradictions,
        "no_constraint": no_constraint,
        "strategic_state": strategic_state,
        "report_markdown": latest_report.report_markdown if latest_report else "",
    }


@app.get("/api/v1/assessments")
def list_assessments_route(limit: int = Query(default=20, ge=1, le=100), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = list_recent_assessments(db, limit=limit)
    return [
        {
            "assessment_id": row.id,
            "business_type": row.business_type,
            "location": row.location,
            "primary_constraint": row.primary_constraint,
            "secondary_constraint": row.secondary_constraint,
            "overall_confidence_label": row.overall_confidence_label,
            "overall_confidence_score": row.overall_confidence_score,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]
