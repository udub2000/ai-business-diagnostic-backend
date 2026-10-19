"""Safe, presentation-focused six-section staging report, never customer deliverable."""
from html import escape
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from .integrated_assessment import IntegratedAssessmentInput, integrated_assessment_review

router = APIRouter(prefix='/staging', tags=['report-layout-review'])
SECTION_LABELS = (
    ('business_snapshot', 'Business Snapshot'),
    ('primary_constraint', 'Primary Constraint'),
    ('why_evidence', 'Why — Supporting Evidence'),
    ('recommended_actions', 'Recommended Actions'),
    ('expected_impact', 'Expected Impact'),
    ('secondary_observations', 'Secondary Observations'),
)

def render_review_html(review: dict) -> str:
    sections = review['assessment']['sections']
    cards = []
    for key, label in SECTION_LABELS:
        value = sections.get(key, '')
        if isinstance(value, (list, tuple)):
            body = '<ol>' + ''.join(f'<li>{escape(str(item))}</li>' for item in value) + '</ol>'
        else:
            body = f'<p>{escape(str(value))}</p>'
        cards.append(f'<section class="panel"><h2>{escape(label)}</h2>{body}</section>')
    financial = review['financial_evidence']
    if financial is None:
        financial_html = '<p>No financial statement provided; financial performance has not been evaluated.</p>'
    else:
        provenance = financial['provenance']
        metrics = financial.get('metrics') or {}
        rows = ''.join(f'<tr><th scope="row">{escape(str(k))}</th><td>{escape(str(v))}</td></tr>' for k, v in sorted(metrics.items()))
        financial_html = (f'<p>Source: {escape(str(provenance.get("statement_type")))}; '
                          f'period: {escape(str(provenance.get("period_label")))}; '
                          f'confirmed: {escape(str(provenance.get("period_confirmed")))}</p>'
                          f'<table><tbody>{rows}</tbody></table>'
                          '<p>Financial figures are not used to select the primary constraint.</p>')
    checks = ''.join(f'<li>{escape(str(item))}</li>' for item in review['integration_checks'])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diagnostic — Internal Report Layout Review</title>
<style>
:root{{color-scheme:light}}body{{margin:0;background:#f4f6f9;color:#172334;font:16px/1.55 system-ui,Arial,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:30px 20px 80px}}
header{{background:#142c47;color:#fff;padding:30px;border-radius:12px;margin-bottom:20px}}
header p{{color:#e3ebf4}}h1{{margin:0;font-size:1.9rem}}h2{{font-size:1.2rem;margin:0 0 12px}}
.panel{{background:#fff;border:1px solid #dce3ec;border-radius:12px;padding:22px;margin:16px 0;overflow-wrap:anywhere}}
.notice{{border-left:5px solid #a36209;background:#fff6e6;color:#603b0c;padding:12px 16px;margin-top:16px}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #dce3ec;padding:8px}}
@media print{{body{{background:white}}main{{padding:0}}.panel{{break-inside:avoid}}}}
</style></head><body><main>
<header><h1>Business Diagnostic — Report Layout Review</h1><p>Six-section assessment and financial context</p></header>
<div class="notice" role="status"><strong>INTERNAL REVIEW — NOT CUSTOMER READY.</strong>
Provisional findings only. No verified primary constraint or customer delivery.</div>
{''.join(cards)}
<section class="panel"><h2>Financial Evidence — Unverified</h2>{financial_html}</section>
<section class="panel"><h2>Verification Required</h2><ul>{checks}</ul></section>
</main></body></html>'''

@router.post('/report-layout-review', response_class=HTMLResponse)
def report_layout_review(data: IntegratedAssessmentInput):
    review = integrated_assessment_review(data)
    return HTMLResponse(render_review_html(review), headers={
        'Cache-Control': 'no-store', 'X-Report-Status': 'internal-review-only',
        'Content-Security-Policy': "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        'X-Content-Type-Options': 'nosniff',
    })
