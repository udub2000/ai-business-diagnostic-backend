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
                "## Next Step\nAdd an Anthropic API key in your environment to generate live reports."
            )

        response = self.client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1800,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts).strip()
