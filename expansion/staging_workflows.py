"""Isolated, no-email/no-database staging workflow adapters.

Document previews are evidence discovery, NOT parsed financial metrics or scoring.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO

from app.document_intelligence.preview import MAX_FILE_BYTES, MAX_FILES, SUPPORTED, preview_document
from .staging_api import AssessmentInput, internal_review_draft

router = APIRouter(prefix='/staging', tags=['isolated-workflows'])

@router.post('/document-preview')
async def document_preview(documents: list[UploadFile] = File(...)):
    if not 1 <= len(documents) <= MAX_FILES:
        raise HTTPException(400, f'Provide 1–{MAX_FILES} documents')
    results=[]
    for document in documents:
        name=Path(document.filename or '').name
        ext=Path(name).suffix.lower()
        if ext not in SUPPORTED:
            raise HTTPException(400, 'Unsupported document format')
        with tempfile.TemporaryDirectory(prefix='diag_staging_') as directory:
            path=Path(directory)/('document'+ext)
            total=0
            with path.open('wb') as output:
                while block := await document.read(256*1024):
                    total += len(block)
                    if total > MAX_FILE_BYTES:
                        raise HTTPException(413, 'Document exceeds 8 MB')
                    output.write(block)
            try:
                result=preview_document(path,name)
                # Only report metadata and evidence category labels. Do not echo raw
                # financial values or document excerpts to staging clients/logs.
                results.append({'file_name':name, 'status':result.get('status','previewed'),
                                'categories':result.get('categories',[]),
                                'preview_only':True})
            except Exception:
                results.append({'file_name':name, 'status':'manual_review_required',
                                'categories':[], 'preview_only':True})
    return {'documents':results,'stored':False,'financial_metrics_extracted':False,
            'email_sent':False,'production_connected':False}

@router.post('/internal-review-pdf')
def internal_review_pdf(data: AssessmentInput):
    review=internal_review_draft(data)
    sections=review['sections']
    styles=getSampleStyleSheet()
    buf=BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=letter)
    story=[Paragraph('INTERNAL REVIEW DRAFT — NOT CUSTOMER READY',styles['Heading1']),
           Paragraph('Synthetic testing only. No verified primary constraint.',styles['Normal']),Spacer(1,12)]
    for title,value in sections.items():
        story.append(Paragraph(escape(title.replace('_',' ').title()),styles['Heading2']))
        values=value if isinstance(value,(list,tuple)) else [value]
        for item in values:
            story.append(Paragraph(escape(str(item)),styles['Normal']))
            story.append(Spacer(1,7))
    doc.build(story)
    return Response(content=buf.getvalue(),media_type='application/pdf',
        headers={'Content-Disposition':'attachment; filename="internal-review-draft.pdf"',
                 'Cache-Control':'no-store', 'X-Report-Status':'internal-review-only'})

@router.post('/financial-extraction-preview')
async def financial_extraction_preview(
    income_statement: UploadFile | None = File(default=None),
    balance_sheet: UploadFile | None = File(default=None),
    cash_flow_statement: UploadFile | None = File(default=None),
):
    """Synthetic-only financial parsing preview, without database/email or persistent files.

    This is not a validated business diagnosis. Financial figures are returned to the
    caller only; do not use this endpoint with real customer documents.
    """
    from .legacy_financial_parser import parse_financial_uploads

    uploads = {'income_statement': income_statement, 'balance_sheet': balance_sheet,
               'cash_flow_statement': cash_flow_statement}
    if not any(uploads.values()):
        raise HTTPException(400, 'Provide at least one financial statement')
    allowed = {'.csv', '.xlsx', '.pdf'}
    with tempfile.TemporaryDirectory(prefix='diag_fin_staging_') as directory:
        metadata = {}
        for kind, upload in uploads.items():
            if upload is None:
                continue
            filename = Path(upload.filename or '').name
            extension = Path(filename).suffix.lower()
            if extension not in allowed:
                raise HTTPException(400, f'Unsupported {kind} format')
            path = Path(directory) / (kind + extension)
            size = 0
            with path.open('wb') as dest:
                while chunk := await upload.read(256 * 1024):
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        raise HTTPException(413, f'{kind} exceeds 8 MB')
                    dest.write(chunk)
            metadata[kind] = {'file_name': filename, 'file_path': str(path)}
        parsed = parse_financial_uploads(metadata)
        # Parse failures are not success; no silent loss of document evidence.
        if parsed['parse_errors'] or any(not entry.get('row_count') for entry in parsed['statements'].values()) or not parsed['combined_metrics']:
            return {'status': 'manual_review_required', 'parse_warnings': [
                'One or more financial statements could not be parsed'],
                'metrics': {}, 'statement_types': list(metadata), 'stored': False,
                'email_sent': False, 'production_connected': False}
        return {'status': 'parsed_preview', 'metrics': parsed['combined_metrics'],
                'statement_types': list(metadata),
                'unavailable_metric_count': len(parsed['unavailable_metrics']),
                'stored': False, 'email_sent': False, 'production_connected': False,
                'diagnostic_verified': False}

# Review parsed metrics with explicit period provenance before allowing any
# financial evidence to influence the diagnostic. This endpoint does not score.
from .financial_evidence_bridge import FinancialEvidenceInput, review_financial_evidence

@router.post('/financial-evidence-review')
def financial_evidence_review(data: FinancialEvidenceInput):
    return review_financial_evidence(data)
