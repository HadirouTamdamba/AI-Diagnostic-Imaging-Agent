"""
Unit tests for application settings
"""
from src.config.settings import Settings


class TestSettings:

    def test_defaults(self):
        settings = Settings(_env_file=None)
        assert settings.model_id == "gemini-2.0-flash"
        assert settings.max_image_size == 5 * 1024 * 1024
        assert settings.supported_formats == ["jpg", "jpeg", "png"]
        assert settings.max_analysis_time == 120

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "gemini-2.5-flash")
        monkeypatch.setenv("MAX_IMAGE_SIZE", "10485760")
        settings = Settings(_env_file=None)
        assert settings.model_id == "gemini-2.5-flash"
        assert settings.max_image_size == 10 * 1024 * 1024

    def test_invalid_formats_fall_back_to_defaults(self):
        settings = Settings(_env_file=None, supported_formats=[])
        assert settings.supported_formats == ["jpg", "jpeg", "png"]
