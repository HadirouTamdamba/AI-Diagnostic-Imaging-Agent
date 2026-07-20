"""
Configuration settings for Medical Imaging Diagnosis Agent
Fixed for Pydantic v2+ compatibility
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation"""

    # API Configuration
    google_api_key: str = ""
    model_id: str = "gemini-flash-latest"
    # Live web search adds references but costs extra requests (free-tier quota);
    # can be toggled off in the UI to conserve the daily quota.
    enable_web_search: bool = True
    # Default UI / report language ("en" or "fr"); switchable in the UI.
    default_language: str = "en"

    # Image Processing
    max_image_size: int = 5 * 1024 * 1024  # 5MB
    supported_formats: list[str] = ["jpg", "jpeg", "png"]
    max_image_width: int = 1024
    max_image_height: int = 1024

    # Analysis Configuration
    max_analysis_time: int = 120  # seconds
    confidence_threshold: float = 0.7

    # UI Configuration
    page_title: str = "🏥 Medical Imaging Diagnosis Agent"
    page_icon: str = "🩻"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }

    @field_validator('supported_formats')
    @classmethod
    def validate_formats(cls, v):
        if not isinstance(v, list) or not v:
            return ["jpg", "jpeg", "png"]
        return v

# Global settings instance
settings = Settings()
