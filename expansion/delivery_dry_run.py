"""Offline MIME packaging verification. No SMTP, Gmail, network or credentials."""
from email.message import EmailMessage
from email.policy import SMTP
from hashlib import sha256
from fastapi import APIRouter
from pydantic import BaseModel, Field
from .integrated_assessment import IntegratedAssessmentInput
from .integrated_pdf import integrated_review_pdf
from .report_layout import render_review_html
from .integrated_assessment import integrated_assessment_review

router = APIRouter(prefix='/staging', tags=['offline-delivery'])

class DeliveryDryRunInput(BaseModel):
    review: IntegratedAssessmentInput
    # No actual addresses: this is an offline content/attachment contract test.
    recipient_roles: list[str] = Field(default_factory=lambda: ['owner', 'operator'], min_length=1, max_length=2)


def build_dry_run(data: DeliveryDryRunInput):
    roles = data.recipient_roles
    if len(set(roles)) != len(roles) or any(role not in {'owner', 'operator'} for role in roles):
        raise ValueError('Only distinct owner/operator roles are supported')
    review = integrated_assessment_review(data.review)
    html = render_review_html(review)
    pdf = integrated_review_pdf(data.review).body
    msg = EmailMessage()
    msg['Subject'] = '[STAGING ONLY] Internal diagnostic review — not customer ready'
    msg['From'] = 'staging@example.invalid'
    msg['To'] = ', '.join(f'{role}@example.invalid' for role in roles)
    msg.set_content('Internal synthetic test only. Findings are provisional; do not forward to customers.')
    msg.add_alternative(html, subtype='html')
    msg.add_attachment(pdf, maintype='application', subtype='pdf', filename='internal-review.pdf')
    wire = msg.as_bytes(policy=SMTP)
    return {
        'status': 'offline_mime_built', 'recipient_roles': roles,
        'attachment_count': len(list(msg.iter_attachments())),
        'attachment_names': ['internal-review.pdf'],
        'pdf_signature_valid': pdf.startswith(b'%PDF'),
        'pdf_size_bytes': len(pdf), 'mime_sha256': sha256(wire).hexdigest(),
        'report_ready': False, 'customer_delivery_enabled': False,
        'email_sent': False, 'network_used': False, 'stored': False,
        'note': 'No SMTP or Gmail connection; MIME is constructed in memory and discarded.'
    }

@router.post('/delivery-dry-run')
def delivery_dry_run(data: DeliveryDryRunInput):
    from fastapi import HTTPException
    try:
        return build_dry_run(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
