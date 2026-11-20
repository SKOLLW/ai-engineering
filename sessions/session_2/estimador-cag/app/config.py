"""Project settings, loaded from environment variables and the .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM provider. SecretStr hides the key if the settings are ever printed or logged.
    openai_api_key: SecretStr
    anthropic_api_key: SecretStr | None = None  # optional: this project uses OpenAI only
    llm_provider: Literal["openai"] = "openai"
    llm_model: str = "gpt-4o-mini"

    # Application
    app_env: str = "development"
    log_level: str = "DEBUG"

    @field_validator("openai_api_key")
    @classmethod
    def key_must_not_be_empty(cls, value: SecretStr) -> SecretStr:
        # "OPENAI_API_KEY=" in .env gives an empty string, not a missing value: reject it here
        # so the app fails at start-up instead of on the first LLM call.
        if not value.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY is empty. Put your key in the .env file.")
        return value


@lru_cache
def get_settings() -> Settings:
    """Read the settings once and reuse them (changes to .env need a server restart)."""
    return Settings()
