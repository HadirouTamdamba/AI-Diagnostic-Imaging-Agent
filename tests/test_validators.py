"""
Unit tests for session validation utilities
"""
from src.utils.validators import SessionValidator


class TestApiKeyValidation:

    def setup_method(self):
        self.validator = SessionValidator()

    def test_empty_key_rejected(self):
        result = self.validator.validate_api_key("")
        assert result["valid"] is False
        assert "required" in result["error"]

    def test_none_key_rejected(self):
        result = self.validator.validate_api_key(None)
        assert result["valid"] is False

    def test_short_key_rejected(self):
        result = self.validator.validate_api_key("AIza123")
        assert result["valid"] is False
        assert "short" in result["error"]

    def test_wrong_prefix_rejected(self):
        result = self.validator.validate_api_key("sk-" + "x" * 30)
        assert result["valid"] is False
        assert "format" in result["error"]

    def test_legacy_aiza_key_accepted(self):
        result = self.validator.validate_api_key("AIza" + "x" * 35)
        assert result["valid"] is True

    def test_new_auth_key_accepted(self):
        # 2026 Google AI Studio "auth keys" start with "AQ."
        result = self.validator.validate_api_key("AQ.Ab8" + "x" * 40)
        assert result["valid"] is True

    def test_placeholder_rejected(self):
        result = self.validator.validate_api_key("your_google_api_key_here")
        assert result["valid"] is False
        assert "placeholder" in result["error"]

    def test_surrounding_whitespace_trimmed(self):
        result = self.validator.validate_api_key("  AIza" + "x" * 35 + "  ")
        assert result["valid"] is True
