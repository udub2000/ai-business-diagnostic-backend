"""LOCAL-ONLY development API. Never deploy as a replacement for production."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from .pipeline import run_pipeline
from .questionnaire_bridge import ConditionSelection
from .routing import RoutingReviewRequired

app = FastAPI(title='Diagnostic Expansion — Isolated Staging', version='0.1.0')

class SelectionInput(BaseModel):
    question_id: str = Field(min_length=1)
    condition_code: str = Field(min_length=1)
    asserted: bool
    selected_by: Literal['respondent', 'reviewer'] = 'respondent'

class AssessmentInput(BaseModel):
    business_type: str = Field(min_length=1)
    owner_goal: str | None = None
    answers: list[dict] = Field(min_length=1)
    condition_selections: list[SelectionInput] = Field(default_factory=list)

@app.get('/staging/health')
def health():
    return {'status': 'ok', 'environment': 'isolated', 'production_connected': False}

@app.post('/staging/assessment-preview')
def preview(data: AssessmentInput):
    try:
        result = run_pipeline(data.business_type, data.answers, owner_goal=data.owner_goal,
            condition_selections=[ConditionSelection(**s.model_dump()) for s in data.condition_selections])
    except (ValueError, TypeError, RoutingReviewRequired) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {'category': result.category, 'evidence_count': result.evidence_count,
            'observations': result.observations, 'preview': result.preview,
            'not_final_report': True, 'production_connected': False}

# Isolated same-origin demo; no external assets or production endpoints.
from pathlib import Path
from fastapi.responses import FileResponse

@app.get('/staging/demo', include_in_schema=False)
def demo():
    return FileResponse(Path(__file__).parent / 'static' / 'demo.html', media_type='text/html')

# Fixture-derived development questionnaire; not an approved production questionnaire.
import json
from fastapi.responses import JSONResponse

@app.get('/staging/questionnaire-catalog')
def questionnaire_catalog():
    catalog_path = Path(__file__).parent / 'questionnaire_catalog.json'
    return JSONResponse(json.loads(catalog_path.read_text(encoding='utf-8')))

@app.get('/staging/questionnaire', include_in_schema=False)
def questionnaire():
    return FileResponse(Path(__file__).parent / 'static' / 'questionnaire.html', media_type='text/html')

# Explicitly internal six-section review draft; not a customer deliverable.
from .report import build_report_draft, as_sections

@app.post('/staging/internal-review-draft')
def internal_review_draft(data: AssessmentInput):
    try:
        result = run_pipeline(data.business_type, data.answers, owner_goal=data.owner_goal,
            condition_selections=[ConditionSelection(**s.model_dump()) for s in data.condition_selections])
        draft = build_report_draft(result, business_type=data.business_type, owner_goal=data.owner_goal)
    except (ValueError, TypeError, RoutingReviewRequired) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {'sections': as_sections(draft), 'report_status': draft.report_status,
            'decision': result.decision, 'evidence_count': result.evidence_count,
            'not_final_report': True, 'production_connected': False}

@app.get('/staging/frontend-v2', include_in_schema=False)
def frontend_v2():
    return FileResponse(Path(__file__).parent / 'static' / 'frontend_v2_staging.html', media_type='text/html')
