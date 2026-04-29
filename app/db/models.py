from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.database import Base


def uuid_str() -> str:
    return str(uuid.uuid4())


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(String, primary_key=True, default=uuid_str)
    business_type = Column(String, nullable=False)
    location = Column(String, nullable=False)
    owner_name = Column(String, nullable=True)
    seasonality_context = Column(String, nullable=True)
    mode = Column(String, nullable=False)

    raw_input_json = Column(Text, nullable=False)
    scores_json = Column(Text, nullable=False)
    category_confidence_json = Column(Text, nullable=False)

    primary_constraint = Column(String, nullable=False)
    secondary_constraint = Column(String, nullable=True)
    constraint_pattern = Column(String, nullable=True)

    overall_confidence_score = Column(Integer, nullable=False)
    overall_confidence_label = Column(String, nullable=False)

    assumptions_json = Column(Text, nullable=False)
    contradictions_json = Column(Text, nullable=False)

    scoring_model_version = Column(String, nullable=False, default="v2")
    prompt_version = Column(String, nullable=False, default="report_v1")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    reports = relationship(
        "AssessmentReport",
        back_populates="assessment",
        cascade="all, delete-orphan",
        order_by="AssessmentReport.created_at",
    )


class AssessmentReport(Base):
    __tablename__ = "assessment_reports"

    id = Column(String, primary_key=True, default=uuid_str)
    assessment_id = Column(String, ForeignKey("assessments.id"), nullable=False)
    report_markdown = Column(Text, nullable=False)
    report_type = Column(String, nullable=False, default="client_report")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    assessment = relationship("Assessment", back_populates="reports")
