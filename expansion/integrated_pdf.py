"""Internal-only PDF rendering of the *combined* assessment and financial review."""
from io import BytesIO
from xml.sax.saxutils import escape
from fastapi import APIRouter
from fastapi.responses import Response
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from .integrated_assessment import IntegratedAssessmentInput, integrated_assessment_review

router = APIRouter(prefix='/staging', tags=['integrated-review'])

@router.post('/integrated-review-pdf')
def integrated_review_pdf(data: IntegratedAssessmentInput):
    review = integrated_assessment_review(data)
    styles = getSampleStyleSheet()
    story = [Paragraph('INTERNAL INTEGRATED REVIEW — NOT CUSTOMER READY', styles['Heading1']),
             Paragraph('Provisional diagnostic; financial evidence does not determine the primary constraint.', styles['Normal']),
             Spacer(1, 12)]
    for section, content in review['assessment']['sections'].items():
        story.append(Paragraph(escape(section.replace('_', ' ').title()), styles['Heading2']))
        values = content if isinstance(content, (list, tuple)) else [content]
        for value in values:
            story.append(Paragraph(escape(str(value)), styles['Normal']))
            story.append(Spacer(1, 5))
    story.append(Paragraph('Financial Evidence (Unverified)', styles['Heading2']))
    evidence = review['financial_evidence']
    if evidence is None:
        story.append(Paragraph('No financial statement supplied.', styles['Normal']))
    else:
        provenance = evidence['provenance']
        for key in ('statement_type', 'period_type', 'period_label', 'period_confirmed'):
            story.append(Paragraph(escape(f'{key}: {provenance.get(key)}'), styles['Normal']))
        for key, value in sorted(evidence['metrics'].items()):
            story.append(Paragraph(escape(f'{key}: {value}'), styles['Normal']))
        derived = review['financial_observations'] or {}
        story.append(Paragraph('Derived observations (review only)', styles['Heading3']))
        story.append(Paragraph(escape(str(derived)), styles['Normal']))
    story.append(Paragraph('Required Verification', styles['Heading2']))
    for check in review['integration_checks']:
        story.append(Paragraph(escape(str(check)), styles['Normal']))
        story.append(Spacer(1, 5))
    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=letter).build(story)
    return Response(content=buf.getvalue(), media_type='application/pdf', headers={
        'Content-Disposition': 'attachment; filename="integrated-internal-review.pdf"',
        'Cache-Control': 'no-store', 'X-Report-Status': 'internal-review-only'})
