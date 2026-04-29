from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    business_id: Mapped[int] = mapped_column(ForeignKey("businesses.id"), index=True)
    scoring_model_version: Mapped[str] = mapped_column(String(50), default="v2")
    prompt_version: Mapped[str] = mapped_column(String(50), default="client_report_v1")
    intake_payload: Mapped[dict] = mapped_column(JSON)
    score_payload: Mapped[dict] = mapped_column(JSON)
    report_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    business = relationship("Business", back_populates="assessments")
