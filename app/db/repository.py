from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Assessment, AssessmentReport


def create_assessment(
    db: Session,
    *,
    payload_dict: dict[str, Any],
    result: dict[str, Any],
    report_markdown: str,
    scoring_model_version: str = "v2",
    prompt_version: str = "report_v1",
) -> Assessment:
    assessment = Assessment(
        business_type=payload_dict.get("business_type", ""),
        location=payload_dict.get("location", ""),
        owner_name=payload_dict.get("owner_name"),
        seasonality_context=payload_dict.get("seasonality_context"),
        mode=payload_dict.get("mode", "guided"),
        raw_input_json=json.dumps(payload_dict),
        scores_json=json.dumps(result["scores"]),
        category_confidence_json=json.dumps(result.get("category_confidence", {})),
        primary_constraint=result["primary_constraint"],
        secondary_constraint=result.get("secondary_constraint"),
        constraint_pattern=result.get("constraint_pattern"),
        overall_confidence_score=result.get("overall_confidence_score", 0),
        overall_confidence_label=result.get("overall_confidence_label", "Exploratory"),
        assumptions_json=json.dumps(result.get("assumptions", [])),
        contradictions_json=json.dumps(result.get("contradictions", [])),
        scoring_model_version=scoring_model_version,
        prompt_version=prompt_version,
    )

    db.add(assessment)
    db.flush()

    report = AssessmentReport(
        assessment_id=assessment.id,
        report_markdown=report_markdown,
        report_type="client_report",
    )
    db.add(report)
    db.commit()
    db.refresh(assessment)
    return assessment


def get_assessment_by_id(db: Session, assessment_id: str) -> Assessment | None:
    return db.query(Assessment).filter(Assessment.id == assessment_id).first()


def list_recent_assessments(db: Session, limit: int = 20) -> list[Assessment]:
    return (
        db.query(Assessment)
        .order_by(Assessment.created_at.desc())
        .limit(limit)
        .all()
    )
