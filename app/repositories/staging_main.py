"""Isolated Step 7 preview only. Legacy app is not imported to avoid DB/email/AI side effects."""
import os

if os.getenv("STAGING_MODE") != "isolated":
    raise RuntimeError("Refusing startup: set STAGING_MODE=isolated")
if os.getenv("ANTHROPIC_API_KEY") or any(os.getenv(x) for x in ("GMAIL_OAUTH_REFRESH_TOKEN", "GMAIL_OAUTH_CLIENT_SECRET", "GMAIL_SMTP_PASSWORD")):
    raise RuntimeError("Refusing startup: live AI/email credentials present")

from expansion.staging_api import app

from expansion.staging_workflows import router as staging_workflow_router
app.include_router(staging_workflow_router)

from expansion.integrated_assessment import router as integrated_router
app.include_router(integrated_router)

from expansion.integrated_pdf import router as integrated_pdf_router
app.include_router(integrated_pdf_router)

from expansion.report_layout import router as report_layout_router
app.include_router(report_layout_router)

from expansion.delivery_dry_run import router as offline_delivery_router
app.include_router(offline_delivery_router)

from expansion.branded_review import router as branded_review_router
app.include_router(branded_review_router)
