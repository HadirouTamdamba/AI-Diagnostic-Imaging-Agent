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
        assert result["model_used"] == "gemini-flash-latest"
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

    def test_status_code_takes_precedence_over_opaque_message(self):
        # agno hides the body ("<Response [...]>"); the status code is the real signal
        transient = Exception("<Response [503 Service Unavailable]>")
        transient.status_code = 503
        assert MedicalImagingAgent._is_transient_error(transient) is True

        bad_key = Exception("<Response [400 Bad Request]>")
        bad_key.status_code = 400
        assert MedicalImagingAgent._is_transient_error(bad_key) is False


class TestErrorPayloadInContent:

    def test_503_error_json_content_is_retried(self, mocked_agent):
        # agno 2.x can return a transient 503 as content without raising
        err_json = '{"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}'
        mocked_agent.agent.run.return_value = make_response(content=err_json)

        result = mocked_agent.analyze_image(MagicMock())

        assert result["success"] is False
        assert result["status_code"] == 503
        # 503 is transient -> retried up to max_retries
        assert mocked_agent.agent.run.call_count == mocked_agent.max_retries

    def test_normal_report_content_not_flagged(self, mocked_agent):
        mocked_agent.agent.run.return_value = make_response(content="### 1. Findings\nNormal study.")

        result = mocked_agent.analyze_image(MagicMock())

        assert result["success"] is True
        assert result["content"].startswith("### 1. Findings")

    def test_detector_ignores_plain_json_without_error(self):
        assert MedicalImagingAgent._error_payload_in_content('{"result": "ok"}') is None
        assert MedicalImagingAgent._error_payload_in_content("Normal report") is None


class TestErrorFormatting:

    def test_400_gives_api_key_hint(self):
        err = Exception("<Response [400 Bad Request]>")
        err.status_code = 400
        msg = MedicalImagingAgent._format_error(err)
        assert "AIza" in msg and "400" in msg

    def test_429_gives_quota_hint(self):
        err = Exception("<Response [429]>")
        err.status_code = 429
        assert "quota" in MedicalImagingAgent._format_error(err).lower()

    def test_unknown_status_falls_back_to_str(self):
        assert MedicalImagingAgent._format_error(Exception("boom")) == "boom"

    def test_bad_key_failure_result_is_actionable(self, mocked_agent):
        err = Exception("<Response [400 Bad Request]>")
        err.status_code = 400
        mocked_agent.agent.run.side_effect = err

        result = mocked_agent.analyze_image(MagicMock())

        assert result["success"] is False
        assert result["status_code"] == 400
        assert result["attempts"] == 1  # 400 is not retried
        assert "AIza" in result["error"]


class TestDailyQuota:

    def test_daily_quota_429_not_retried(self, mocked_agent):
        # Per-day free-tier quota won't recover in seconds -> fail fast, no retries
        err = Exception(
            "429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        err.status_code = 429
        mocked_agent.agent.run.side_effect = err

        result = mocked_agent.analyze_image(MagicMock())

        assert result["success"] is False
        assert result["attempts"] == 1
        assert mocked_agent.agent.run.call_count == 1
        assert "Daily free-tier quota" in result["error"]

    def test_per_minute_429_still_retried(self, mocked_agent):
        # A rate-limit 429 without a per-day marker should still retry
        err = Exception("429 Too Many Requests")
        err.status_code = 429
        mocked_agent.agent.run.side_effect = err

        result = mocked_agent.analyze_image(MagicMock())

        assert result["success"] is False
        assert mocked_agent.agent.run.call_count == mocked_agent.max_retries


class TestWebSearchToggle:

    def _agent(self, enable):
        with patch("src.models.medical_agent.Agent") as agent_cls, \
             patch("src.models.medical_agent.Gemini"), \
             patch("src.models.medical_agent.DuckDuckGoTools"):
            agent_cls.return_value = MagicMock()
            return MedicalImagingAgent("AIza" + "x" * 35, enable_web_search=enable)

    def test_enabled_registers_tool_and_prompt(self):
        agent = self._agent(True)
        assert agent.get_agent_info()["tools"] == ["DuckDuckGo Search"]
        assert "DuckDuckGo search tool" in agent.get_analysis_prompt()

    def test_disabled_no_tool_and_prompt_adapts(self):
        agent = self._agent(False)
        assert agent.get_agent_info()["tools"] == []
        assert agent.get_agent_info()["web_search"] is False
        assert "DuckDuckGo search tool" not in agent.get_analysis_prompt()


class TestLanguage:

    def test_english_prompt_default(self, mocked_agent):
        prompt = mocked_agent.get_analysis_prompt("en")
        assert "LANGUE" not in prompt

    def test_french_prompt_directive(self, mocked_agent):
        prompt = mocked_agent.get_analysis_prompt("fr")
        assert "français" in prompt

    def test_analyze_records_language(self, mocked_agent):
        mocked_agent.agent.run.return_value = make_response()
        result = mocked_agent.analyze_image(MagicMock(), language="fr")
        assert result["language"] == "fr"

    def test_translate_report_success(self, mocked_agent):
        mocked_agent.agent.run.return_value = make_response(content="Rapport traduit")
        result = mocked_agent.translate_report("Report", "fr")
        assert result["success"] is True
        assert result["content"] == "Rapport traduit"
        assert result["language"] == "fr"

    def test_translate_report_failure_is_structured(self, mocked_agent):
        err = Exception("<Response [400 Bad Request]>")
        err.status_code = 400
        mocked_agent.agent.run.side_effect = err
        result = mocked_agent.translate_report("Report", "fr")
        assert result["success"] is False
        assert result["status_code"] == 400


@pytest.fixture
def agent_with_fallback():
    """Agent + (primary, fallback) mocks; the patch stays active for the whole test
    so the lazily-built fallback agent is mocked too."""
    with patch("src.models.medical_agent.Agent") as agent_cls, \
         patch("src.models.medical_agent.Gemini"), \
         patch("src.models.medical_agent.DuckDuckGoTools"):
        primary, fallback = MagicMock(), MagicMock()
        agent_cls.side_effect = [primary, fallback]
        agent = MedicalImagingAgent(
            "AIza" + "x" * 35, retry_base_delay=0.0,
            fallback_model_id="gemini-3-flash-preview",
        )
        yield agent, primary, fallback


class TestModelFallback:

    def test_overloaded_primary_falls_back(self, agent_with_fallback):
        agent, primary, fallback = agent_with_fallback
        err = Exception("<Response [503]>")
        err.status_code = 503
        primary.run.side_effect = err
        fallback.run.return_value = make_response(content="Report from fallback")

        result = agent.analyze_image(MagicMock())

        assert result["success"] is True
        assert result["model_used"] == "gemini-3-flash-preview"
        assert result["fallback_used"] is True
        assert result["content"] == "Report from fallback"

    def test_no_fallback_on_config_error(self, agent_with_fallback):
        agent, primary, fallback = agent_with_fallback
        err = Exception("<Response [400]>")
        err.status_code = 400
        primary.run.side_effect = err

        result = agent.analyze_image(MagicMock())

        assert result["success"] is False
        fallback.run.assert_not_called()  # 400 would fail identically on any model

    def test_no_fallback_when_primary_succeeds(self, agent_with_fallback):
        agent, primary, fallback = agent_with_fallback
        primary.run.return_value = make_response()

        result = agent.analyze_image(MagicMock())

        assert result["success"] is True
        assert result["fallback_used"] is False
        fallback.run.assert_not_called()

    def test_fallback_equal_to_primary_is_ignored(self):
        with patch("src.models.medical_agent.Agent"), \
             patch("src.models.medical_agent.Gemini"), \
             patch("src.models.medical_agent.DuckDuckGoTools"):
            agent = MedicalImagingAgent("AIza" + "x" * 35, model_id="m", fallback_model_id="m")
        assert agent.fallback_model_id is None


class TestAgentInfo:

    def test_get_agent_info(self, mocked_agent):
        info = mocked_agent.get_agent_info()
        assert info["model_id"] == "gemini-flash-latest"
        assert "DuckDuckGo Search" in info["tools"]
