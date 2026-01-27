"""
Configuration settings for Tech Discovery Recorder backend.
Follows Austin's patterns from CLAUDE.md - environment-based config with Pydantic.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # App Configuration
    app_name: str = "Tech Discovery Recorder"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Database
    database_url: str = Field(
        description="PostgreSQL database URL",
        examples=["postgresql://localhost:5432/tech_discovery"]
    )

    # Security
    jwt_secret: str = Field(
        description="JWT secret key (min 32 characters)",
        min_length=32
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(default=1440, description="Token expiry in minutes")  # 24 hours

    # Claude API
    claude_api_key: str = Field(
        description="Anthropic Claude API key",
        examples=["sk-ant-..."]
    )
    claude_model: str = Field(default="claude-3-haiku-20240307", description="Claude model to use")
    claude_max_tokens: int = Field(default=4000, description="Max tokens for Claude responses")

    # OpenAI API (for Whisper transcription)
    openai_api_key: str = Field(
        description="OpenAI API key for Whisper transcription",
        examples=["sk-proj-..."]
    )

    # Storage
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key for S3")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret key for S3")
    aws_region: str = Field(default="us-west-2", description="AWS region")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for audio storage")
    local_storage_path: str = Field(default="./storage", description="Local storage path for dev")

    # Recording Settings
    max_file_size_mb: int = Field(default=50, description="Maximum audio file size in MB")
    retention_days: int = Field(default=90, description="Audio retention period in days")
    supported_formats: list[str] = Field(
        default=["audio/webm", "audio/wav", "audio/mp4", "audio/m4a"],
        description="Supported audio formats"
    )

    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Requests per minute per IP")
    claude_rate_limit_rpm: int = Field(default=50, description="Claude API requests per minute")

    # CORS
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000,https://*.render.com",
        description="Allowed CORS origins (comma-separated)"
    )

    def get_allowed_origins(self) -> list[str]:
        """Get allowed origins as a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return not self.debug and self.log_level.upper() != "DEBUG"

    @property
    def use_s3_storage(self) -> bool:
        """Check if S3 storage is configured."""
        return (
            self.aws_access_key_id is not None
            and self.aws_secret_access_key is not None
            and self.s3_bucket is not None
        )

    def get_database_url(self) -> str:
        """Get the complete database URL."""
        return self.database_url

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return "sqlite" in self.database_url.lower()

    def validate_required_settings(self) -> None:
        """Validate that all required settings are present."""
        required_fields = ["database_url", "jwt_secret", "claude_api_key", "openai_api_key"]
        missing_fields = []

        for field in required_fields:
            if not getattr(self, field, None):
                missing_fields.append(field.upper())

        if missing_fields:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")


# Global settings instance
settings = Settings()

# Validate settings on import (fail fast)
try:
    settings.validate_required_settings()
except ValueError as e:
    print(f"❌ Configuration Error: {e}")
    print("💡 Create a .env file with required variables. See .env.example")
    raise