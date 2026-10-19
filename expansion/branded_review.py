"""Staging-only branded report visual parity prototype; never a customer report."""
from io import BytesIO
from xml.sax.saxutils import escape
from fastapi import APIRouter
from fastapi.responses import Response
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, KeepTogether
from .integrated_assessment import IntegratedAssessmentInput, integrated_assessment_review
from .report_layout import SECTION_LABELS

router = APIRouter(prefix='/staging', tags=['branded-review'])
NAVY=colors.HexColor('#071126')
INK=colors.HexColor('#0F172A')
MUTED=colors.HexColor('#64748B')
BORDER=colors.HexColor('#DBE3EF')
PALE=colors.HexColor('#F7F9FC')


def build_branded_review(data: IntegratedAssessmentInput) -> bytes:
    review=integrated_assessment_review(data)
    sections=review['assessment']['sections']
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BrandTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle(name='BrandSection', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle(name='BrandBody', parent=styles['BodyText'], fontSize=9, leading=13, textColor=INK, spaceAfter=6))
    styles.add(ParagraphStyle(name='BrandWhite', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=14, textColor=colors.white))
    story=[Paragraph('AI BUSINESS DIAGNOSTIC',styles['BrandBody']),Paragraph('Report Overview',styles['BrandTitle'])]
    warning=Table([[Paragraph('INTERNAL REVIEW — NOT CUSTOMER READY. No confirmed primary constraint or validated customer recommendations.',styles['BrandWhite'])]],colWidths=[495])
    warning.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),NAVY),('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14),('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12)]))
    story.extend([warning,Spacer(1,16)])
    for key,label in SECTION_LABELS:
        val=sections.get(key,'')
        lines=val if isinstance(val,(list,tuple)) else [val]
        contents=[Paragraph(escape(label),styles['BrandSection'])]
        contents.extend(Paragraph(escape(str(x)),styles['BrandBody']) for x in lines)
        story.append(KeepTogether(contents))
        story.append(Spacer(1,10))
    story.append(Paragraph('Financial Evidence — Unverified',styles['BrandSection']))
    financial=review['financial_evidence']
    if financial is None:
        story.append(Paragraph('No financial statement supplied.',styles['BrandBody']))
    else:
        provenance=financial['provenance']
        story.append(Paragraph(escape('Source: '+str(provenance.get('statement_type'))+'; period: '+str(provenance.get('period_label'))+'; confirmed: '+str(provenance.get('period_confirmed'))),styles['BrandBody']))
        for k,v in sorted(financial.get('metrics',{}).items()):
            story.append(Paragraph(escape(str(k)+': '+str(v)),styles['BrandBody']))
    story.append(Paragraph('Required Verification',styles['BrandSection']))
    for check in review['integration_checks']:
        story.append(Paragraph(escape(str(check)),styles['BrandBody']))
    buf=BytesIO()
    def footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.line(40,32,572,32)
        canvas.setFillColor(MUTED)
        canvas.setFont('Helvetica',7.5)
        canvas.drawString(40,20,'The Uniikye Truth | INTERNAL STAGING REVIEW ONLY')
        canvas.drawRightString(572,20,f'Page {doc.page}')
        canvas.restoreState()
    SimpleDocTemplate(buf,pagesize=letter,leftMargin=52,rightMargin=52,topMargin=44,bottomMargin=48,title='Internal Diagnostic Report Layout Review',author='The Uniikye Truth',pageCompression=0).build(story,onFirstPage=footer,onLaterPages=footer)
    return buf.getvalue()

@router.post('/branded-review-pdf')
def branded_review_pdf(data: IntegratedAssessmentInput):
    return Response(content=build_branded_review(data),media_type='application/pdf',headers={'Content-Disposition':'attachment; filename="staging-branded-internal-review.pdf"','Cache-Control':'no-store','X-Report-Status':'internal-review-only','X-Content-Type-Options':'nosniff'})
