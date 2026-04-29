from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Business Diagnostic API"
    environment: str = "development"
    database_url: str = "sqlite:///./ai_business_diagnostic.db"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    scoring_model_version: str = "v2"
    prompt_version: str = "client_report_v1"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
