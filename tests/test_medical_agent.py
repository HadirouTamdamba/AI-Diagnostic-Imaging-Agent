"""
Unit tests for MedicalImagingAgent (Agno agent mocked, no API calls)
"""
from unittest.mock import MagicMock, patch

import pytest
from src.models.medical_agent import MedicalImagingAgent


@pytest.fixture
def mocked_agent():
    """MedicalImagingAgent with the underlying Agno Agent replaced by a mock"""
    with patch("src.models.medical_agent.Agent") as agent_cls, \
         patch("src.models.medical_agent.Gemini"), \
         patch("src.models.medical_agent.DuckDuckGoTools"):
        agent_cls.return_value = MagicMock()
        medical_agent = MedicalImagingAgent("AIza" + "x" * 35, retry_base_delay=0.0)
        yield medical_agent


def make_response(content="Report content", metrics=None):
    response = MagicMock()
    response.content = content
    response.metrics = metrics or {"input_tokens": [100], "output_tokens": [400]}
    return response


class TestAnalyzeImage:

    def test_success_returns_structured_result(self, mocked_agent):
        mocked_agent.agent.run.return_value = make_response()

        result = mocked_agent.analyze_image(MagicMock())

        assert result["success"] is True
        assert result["content"] == "Report content"
        assert result["model_used"] == "gemini-2.0-flash"
        assert result["attempts"] == 1
        assert result["token_usage"]["total_tokens"] == 500

    def test_transient_error_is_retried(self, mocked_agent):
        mocked_agent.agent.run.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED: rate limit exceeded"),
            make_response(),
        ]

        result = mocked_agent.analyze_image(MagicMock())

        assert result["success"] is True
        assert result["attempts"] == 2
        assert mocked_agent.agent.run.call_count == 2

    def test_non_transient_error_fails_fast(self, mocked_agent):
        mocked_agent.agent.run.side_effect = Exception("Invalid API key provided")

        result = mocked_agent.analyze_image(MagicMock())

        assert result["success"] is False
        assert "Invalid API key" in result["error"]
        assert mocked_agent.agent.run.call_count == 1

    def test_exhausted_retries_returns_failure(self, mocked_agent):
        mocked_agent.agent.run.side_effect = Exception("503 Service Unavailable")

        result = mocked_agent.analyze_image(MagicMock())

        assert result["success"] is False
        assert result["attempts"] == mocked_agent.max_retries
        assert mocked_agent.agent.run.call_count == mocked_agent.max_retries

    def test_custom_prompt_is_used(self, mocked_agent):
        mocked_agent.agent.run.return_value = make_response()

        mocked_agent.analyze_image(MagicMock(), custom_prompt="Custom prompt")

        assert mocked_agent.agent.run.call_args[0][0] == "Custom prompt"

    def test_missing_metrics_does_not_fail(self, mocked_agent):
        response = MagicMock()
        response.content = "Report"
        response.metrics = None
        mocked_agent.agent.run.return_value = response

        result = mocked_agent.analyze_image(MagicMock())

        assert result["success"] is True
        assert result["token_usage"]["total_tokens"] == 0


class TestTransientErrorDetection:

    @pytest.mark.parametrize("message", [
        "429 Too Many Requests",
        "quota exceeded for this project",
        "503 Service Unavailable",
        "Deadline exceeded",
        "Connection reset by peer",
    ])
    def test_transient_errors_detected(self, message):
        assert MedicalImagingAgent._is_transient_error(Exception(message)) is True

    @pytest.mark.parametrize("message", [
        "Invalid API key",
        "Permission denied",
        "Image format not supported",
    ])
    def test_permanent_errors_not_retried(self, message):
        assert MedicalImagingAgent._is_transient_error(Exception(message)) is False


class TestAgentInfo:

    def test_get_agent_info(self, mocked_agent):
        info = mocked_agent.get_agent_info()
        assert info["model_id"] == "gemini-2.0-flash"
        assert "DuckDuckGo Search" in info["tools"]
