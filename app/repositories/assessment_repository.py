from sqlalchemy.orm import Session

from app.models.assessment import Assessment


class AssessmentRepository:
    @staticmethod
    def create(
        db: Session,
        *,
        business_id: int,
        scoring_model_version: str,
        prompt_version: str,
        intake_payload: dict,
        score_payload: dict,
        report_markdown: str | None,
    ) -> Assessment:
        assessment = Assessment(
            business_id=business_id,
            scoring_model_version=scoring_model_version,
            prompt_version=prompt_version,
            intake_payload=intake_payload,
            score_payload=score_payload,
            report_markdown=report_markdown,
        )
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
        return assessment

    @staticmethod
    def get(db: Session, assessment_id: int) -> Assessment | None:
        return db.query(Assessment).filter(Assessment.id == assessment_id).first()
