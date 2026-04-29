from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.schemas.assessment import IntakePayload, ScoreResponse


RECOMMENDATION_LIBRARY = {
    "Demand": [
        "Increase local lead generation",
        "Improve referral engine",
        "Strengthen recurring service plans",
    ],
    "Conversion": [
        "Improve lead response speed",
        "Tighten quoting and follow-up",
        "Improve trust signals and qualification",
    ],
    "Capacity": [
        "Add crew or labor capacity",
        "Rebalance schedule and remove low-value jobs",
        "Use pricing to shape demand",
    ],
    "Operations": [
        "Optimize routing",
        "Reduce idle time and scheduling friction",
        "Standardize field execution",
    ],
    "Financials": [
        "Raise prices where justified",
        "Cut low-margin work",
        "Track gross profit by job",
    ],
}

CONSTRAINT_PATTERNS = {
    "Demand": "Inconsistent lead flow limiting utilization",
    "Conversion": "Healthy inquiries but weak close effectiveness",
    "Capacity": "Healthy demand but insufficient fulfillment bandwidth",
    "Operations": "Routing and scheduling inefficiency reducing output",
    "Financials": "Margin compression from high labor burden or underpricing",
}


@dataclass
class MetricResult:
    score: int
    reliability: float
    assumed: bool = False


def clamp(value: float, min_value: int = 0, max_value: int = 100) -> int:
    return int(max(min_value, min(max_value, round(value))))


def band(value: float | None, thresholds: list[tuple[float, int]], default: int, reliability: float = 1.0) -> MetricResult:
    if value is None:
        return MetricResult(score=default, reliability=0.25, assumed=True)
    for upper, score in thresholds:
        if value <= upper:
            return MetricResult(score=score, reliability=reliability)
    return MetricResult(score=thresholds[-1][1], reliability=reliability)


def map_choice(value: str | None, mapping: dict[str, int], default: int = 50, reliability: float = 0.55) -> MetricResult:
    if not value:
        return MetricResult(score=default, reliability=0.25, assumed=True)
    return MetricResult(score=mapping.get(value, default), reliability=reliability)


def weighted_score(metrics: list[tuple[MetricResult, float]]) -> tuple[int, int, list[str]]:
    score = sum(metric.score * weight for metric, weight in metrics)
    confidence = sum(metric.reliability * weight for metric, weight in metrics) * 100
    assumptions = ["Assumed one or more missing inputs" for metric, _ in metrics if metric.assumed]
    return clamp(score), clamp(confidence), assumptions


def detect_contradictions(intake: IntakePayload) -> list[str]:
    contradictions: list[str] = []
    if (intake.backlog_days is not None and intake.backlog_days >= 7) and (intake.utilization_rate is not None and intake.utilization_rate <= 55):
        contradictions.append("High backlog paired with low utilization may indicate scheduling inefficiency or unreliable inputs.")
    if (intake.leads_per_week is not None and intake.leads_per_week <= 10) and (intake.backlog_days is not None and intake.backlog_days >= 10):
        contradictions.append("Low leads paired with high backlog may indicate a very small team, heavy repeat business, or inconsistent inputs.")
    if intake.profitability_feeling == "struggling" and (intake.gross_margin_pct is not None and intake.gross_margin_pct >= 45):
        contradictions.append("Healthy reported margin paired with financial stress may indicate cash flow pressure or hidden costs.")
    return contradictions


def score_assessment(intake: IntakePayload) -> ScoreResponse:
    assumptions: list[str] = []

    demand_metrics = [
        (band(intake.leads_per_week, [(5, 20), (15, 45), (30, 70), (9999, 90)], default=50), 0.30),
        (band(intake.backlog_days, [(1, 25), (3, 45), (7, 65), (14, 82), (9999, 95)], default=50, reliability=0.8), 0.30),
        (map_choice(intake.schedule_fullness, {
            "frequent_empty_slots": 25,
            "some_gaps": 45,
            "mostly_full": 75,
            "overbooked": 90,
        }), 0.20),
        (map_choice(intake.demand_consistency, {
            "inconsistent": 35,
            "mixed": 55,
            "stable": 80,
        }, reliability=0.55), 0.20),
    ]
    demand_score, demand_conf, demand_assumptions = weighted_score(demand_metrics)
    assumptions += demand_assumptions

    conversion_metrics = [
        (band(intake.conversion_rate, [(20, 20), (30, 40), (50, 70), (70, 90), (9999, 100)], default=50), 0.45),
        (map_choice(intake.response_speed, {
            "slow": 35,
            "average": 60,
            "fast": 85,
        }, reliability=0.55), 0.15),
        (map_choice(intake.notes.lower() if intake.notes else None, {
            "wrong_leads": 40,
        }, default=60, reliability=0.45), 0.15),
        (map_choice(None, {}, default=60, reliability=0.25), 0.25),
    ]
    conversion_score, conversion_conf, conversion_assumptions = weighted_score(conversion_metrics)
    assumptions += conversion_assumptions

    capacity_metrics = [
        (band(intake.utilization_rate, [(50, 25), (65, 45), (80, 70), (90, 88), (9999, 95)], default=50), 0.35),
        (band(intake.backlog_days, [(1, 25), (3, 45), (7, 65), (14, 82), (9999, 95)], default=50, reliability=0.8), 0.30),
        (map_choice(intake.turned_away_work, {
            "never": 30,
            "occasionally": 60,
            "frequently": 80,
            "always": 95,
        }), 0.20),
        (map_choice(intake.can_take_work_tomorrow, {
            "yes_easily": 30,
            "maybe": 55,
            "unlikely": 80,
            "no": 95,
        }), 0.15),
    ]
    capacity_score, capacity_conf, capacity_assumptions = weighted_score(capacity_metrics)
    assumptions += capacity_assumptions

    operations_metrics = [
        (band(intake.jobs_per_crew_per_day, [(2, 25), (4, 50), (6, 75), (9999, 90)], default=50), 0.25),
        (band(intake.travel_time_minutes, [(10, 90), (20, 70), (35, 50), (9999, 25)], default=50), 0.20),
        (map_choice(intake.notes.lower() if intake.notes else None, {
            "chaotic": 25,
        }, default=60, reliability=0.45), 0.20),
        (map_choice(intake.schedule_fullness, {
            "frequent_empty_slots": 45,
            "some_gaps": 55,
            "mostly_full": 70,
            "overbooked": 60,
        }, reliability=0.45), 0.15),
        (map_choice(None, {}, default=60, reliability=0.25), 0.10),
        (map_choice(None, {}, default=60, reliability=0.25), 0.10),
    ]
    operations_score, operations_conf, operations_assumptions = weighted_score(operations_metrics)
    assumptions += operations_assumptions

    financial_metrics = [
        (band(intake.gross_margin_pct, [(20, 20), (30, 40), (45, 65), (60, 85), (9999, 95)], default=50), 0.25),
        (band(intake.labor_cost_pct, [(35, 90), (45, 75), (55, 60), (65, 40), (9999, 20)], default=50), 0.25),
        (map_choice(intake.profitability_feeling, {
            "comfortable": 90,
            "tight": 55,
            "breaking_even": 40,
            "struggling": 20,
        }), 0.15),
        (map_choice(None, {}, default=60, reliability=0.25), 0.15),
        (map_choice(None, {}, default=60, reliability=0.25), 0.10),
        (map_choice(None, {}, default=60, reliability=0.25), 0.10),
    ]
    financial_score, financial_conf, financial_assumptions = weighted_score(financial_metrics)
    assumptions += financial_assumptions

    scores = {
        "Demand": demand_score,
        "Conversion": conversion_score,
        "Capacity": capacity_score,
        "Operations": operations_score,
        "Financials": financial_score,
    }
    category_confidence = {
        "Demand": demand_conf,
        "Conversion": conversion_conf,
        "Capacity": capacity_conf,
        "Operations": operations_conf,
        "Financials": financial_conf,
    }

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
    primary_constraint = sorted_scores[0][0]
    secondary_constraint = sorted_scores[1][0] if (sorted_scores[1][1] - sorted_scores[0][1]) <= 10 else None

    contradictions = detect_contradictions(intake)
    input_quality = sum(category_confidence.values()) / len(category_confidence)
    score_gap = sorted_scores[1][1] - sorted_scores[0][1]
    score_separation = 95 if score_gap >= 15 else 80 if score_gap >= 10 else 65 if score_gap >= 6 else 45 if score_gap >= 3 else 25
    answered_fields = sum(1 for v in intake.model_dump().values() if v not in (None, ""))
    total_fields = len(intake.model_dump())
    completeness_ratio = answered_fields / total_fields
    completeness = 95 if completeness_ratio >= 0.85 else 80 if completeness_ratio >= 0.70 else 60 if completeness_ratio >= 0.50 else 40 if completeness_ratio >= 0.30 else 20
    contradiction_component = max(0, 100 - (10 * len(contradictions)))
    overall_confidence_score = clamp((input_quality * 0.40) + (score_separation * 0.25) + (completeness * 0.20) + (contradiction_component * 0.15))
    overall_confidence_label = "High" if overall_confidence_score >= 85 else "Medium" if overall_confidence_score >= 65 else "Low" if overall_confidence_score >= 45 else "Exploratory"

    assumptions = sorted(set(assumptions))

    return ScoreResponse(
        scores={k: int(v) for k, v in scores.items()},
        category_confidence={k: int(v) for k, v in category_confidence.items()},
        primary_constraint=primary_constraint,
        secondary_constraint=secondary_constraint,
        constraint_pattern=CONSTRAINT_PATTERNS[primary_constraint],
        overall_confidence_score=overall_confidence_score,
        overall_confidence_label=overall_confidence_label,
        contradictions=contradictions,
        assumptions=assumptions,
        recommended_action_pool=RECOMMENDATION_LIBRARY[primary_constraint],
    )
