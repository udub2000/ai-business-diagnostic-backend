
from anthropic import Anthropic

from app.core.config import settings


class AnthropicService:
    def __init__(self) -> None:
        self.client = Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None

    def generate_markdown_report(self, system_prompt: str, user_prompt: str) -> str:
        if not self.client:
            return (
                "# Diagnostic Report\n\n"
                "Anthropic API key not configured. This placeholder report confirms the backend pipeline is working.\n\n"
                "## Next Step\n"
                "Add an Anthropic API key in your environment to generate live reports."
            )

        response = self.client.messages.create(
            model=settings.anthropic_model,
            max_tokens=2200,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        parts = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                parts.append(block.text)

        report = "\n".join(parts).strip()

        if not report:
            return (
                "# Diagnostic Report\n\n"
                "The AI response came back empty. Please retry the report generation."
            )

        if len(report) < 900 or not report.rstrip().endswith((".", "!", "?")):
            return (
                "# Diagnostic Report\n\n"
                "The generated report appears incomplete. Please retry the report generation.\n\n"
                "## Diagnostic Note\n"
                "The language model response was shorter than expected or ended mid-thought."
            )

        return report
