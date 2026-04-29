from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.repositories.assessment_repository import AssessmentRepository
from app.repositories.business_repository import BusinessRepository
from app.schemas.assessment import AssessmentRead, ReportRequest, ScoreRequest, ScoreResponse
from app.schemas.business import BusinessRead
from app.services.anthropic_service import AnthropicService
from app.services.prompt_service import build_report_prompt
from app.services.scoring_service import score_assessment

router = APIRouter(prefix="/api/v1")


@router.post("/assessments/score", response_model=ScoreResponse)
def score_only(request: ScoreRequest) -> ScoreResponse:
    return score_assessment(request.intake)


@router.post("/assessments/report", response_model=AssessmentRead)
def generate_report(request: ReportRequest, db: Session = Depends(get_db)) -> AssessmentRead:
    business = BusinessRepository.create(db, request.business)
    score = score_assessment(request.intake)

    report_markdown = None
    if request.generate_report:
        system_prompt, user_prompt = build_report_prompt(ScoreRequest(business=request.business, intake=request.intake), score)
        report_markdown = AnthropicService().generate_markdown_report(system_prompt, user_prompt)

    assessment = AssessmentRepository.create(
        db,
        business_id=business.id,
        scoring_model_version=settings.scoring_model_version,
        prompt_version=settings.prompt_version,
        intake_payload=request.intake.model_dump(),
        score_payload=score.model_dump(),
        report_markdown=report_markdown,
    )

    return AssessmentRead(
        id=assessment.id,
        business=BusinessRead.model_validate(business),
        score_payload=assessment.score_payload,
        report_markdown=assessment.report_markdown,
    )


@router.get("/assessments/{assessment_id}", response_model=AssessmentRead)
def get_assessment(assessment_id: int, db: Session = Depends(get_db)) -> AssessmentRead:
    assessment = AssessmentRepository.get(db, assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    return AssessmentRead(
        id=assessment.id,
        business=BusinessRead.model_validate(assessment.business),
        score_payload=assessment.score_payload,
        report_markdown=assessment.report_markdown,
    )
