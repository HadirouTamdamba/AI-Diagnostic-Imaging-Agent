"""
Medical imaging analysis agent with enhanced capabilities
"""
import json
import logging
import time
from typing import Any

from agno.agent import Agent
from agno.media import Image as AgnoImage
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

logger = logging.getLogger(__name__)


class ProviderResponseError(Exception):
    """Raised when the model returns a provider error payload as its content
    instead of raising (observed with agno 2.x on transient 5xx). Carries the
    HTTP status code so retry/formatting logic treats it like a real API error.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

# Substrings identifying transient API errors worth retrying (rate limits, overload, network)
TRANSIENT_ERROR_MARKERS = (
    "429", "rate limit", "quota", "resource_exhausted",
    "500", "internal", "503", "unavailable", "overloaded",
    "timeout", "deadline", "connection",
)

# HTTP status codes that are safe to retry (server-side / throttling)
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Markers identifying a per-DAY free-tier quota (won't recover within a retry window)
DAILY_QUOTA_MARKERS = ("perday", "per day", "free_tier", "freetier", "free-tier")

# Actionable messages by HTTP status. The provider SDK (agno) hides the raw
# error body, so we map the status code to a message the user can act on.
STATUS_HINTS = {
    400: "Invalid request — most commonly an invalid or placeholder API key, or an "
         "unsupported model/image. Check that GOOGLE_API_KEY is a real Google AI Studio "
         "key (it starts with 'AIza' or 'AQ.').",
    401: "Authentication failed — the API key is missing or invalid.",
    403: "Access denied — the API key lacks permission for this model, or the "
         "Generative Language API is not enabled for the project.",
    404: "Model not found — the configured MODEL_ID may be retired or misspelled.",
    429: "Rate limit or quota exceeded — retry later or check your Google AI Studio quota.",
}

class MedicalImagingAgent:
    """Enhanced medical imaging analysis agent"""

    def __init__(self, api_key: str, model_id: str = "gemini-flash-latest",
                 max_retries: int = 3, retry_base_delay: float = 2.0,
                 enable_web_search: bool = True):
        self.api_key = api_key
        self.model_id = model_id
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.enable_web_search = enable_web_search
        self.agent = self._initialize_agent()

    def _initialize_agent(self) -> Agent:
        """Initialize the Gemini agent, optionally with the web-search tool.

        Web search adds live medical references but costs extra API round-trips
        (each tool call is another request), so it can be disabled to conserve
        the Gemini free-tier daily quota.
        """
        try:
            tools = [DuckDuckGoTools()] if self.enable_web_search else []
            return Agent(
                model=Gemini(
                    id=self.model_id,
                    api_key=self.api_key
                ),
                tools=tools,
                markdown=True,
                debug_mode=False,
                telemetry=False,  # do not phone home from a medical app
            )
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise

    def get_analysis_prompt(self) -> str:
        """Enhanced analysis prompt with structured output.

        Section 5 adapts to whether the web-search tool is available so the model
        is not asked to call a tool that is not registered.
        """
        if self.enable_web_search:
            research_section = """### 5. 📚 Research Context & References
**Important**: Use the DuckDuckGo search tool to find:
- Recent medical literature on similar cases
- Current treatment protocols and guidelines
- Relevant technological advances in imaging
- Patient education resources

Provide 2-3 authoritative medical references with brief summaries."""
        else:
            research_section = """### 5. 📚 Research Context & References
Based on your medical knowledge, provide 2-3 authoritative references
(clinical guidelines, key literature) with brief summaries. Note that these
are from training knowledge and should be verified against current sources."""

        return f"""
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging.
Analyze the provided medical image with the following structured approach:

### 1. 🔍 Image Type & Technical Assessment
- **Imaging Modality**: Specify the type (X-ray, MRI, CT, Ultrasound, etc.)
- **Anatomical Region**: Identify the body part and positioning
- **Image Quality**: Assess technical adequacy, contrast, and diagnostic value
- **Acquisition Parameters**: Note any visible technical details

### 2. 📋 Systematic Findings
- **Primary Observations**: List key anatomical structures visible
- **Abnormalities**: Describe any deviations from normal anatomy
- **Measurements**: Include size, density, and location details where relevant
- **Severity Assessment**: Rate as Normal/Mild/Moderate/Severe with justification

### 3. 🎯 Diagnostic Assessment
- **Primary Diagnosis**: Most likely diagnosis with confidence level (%)
- **Differential Diagnoses**: Alternative possibilities ranked by likelihood
- **Supporting Evidence**: Specific imaging findings supporting each diagnosis
- **Critical Findings**: Any urgent or significant abnormalities requiring immediate attention

### 4. 👥 Patient-Friendly Explanation
- **Simplified Summary**: Explain findings in clear, non-technical language
- **Visual Analogies**: Use relatable comparisons when helpful
- **Common Concerns**: Address typical patient questions about these findings
- **Next Steps**: General guidance on typical follow-up procedures

{research_section}

### 6. ⚠️ Limitations & Recommendations
- **Analysis Limitations**: Acknowledge any constraints of AI analysis
- **Professional Review**: Emphasize need for radiologist confirmation
- **Follow-up**: Suggest appropriate next steps

Format your response with clear markdown headers, bullet points, and professional medical terminology balanced with patient accessibility.
"""

    @staticmethod
    def _is_daily_quota_error(error: Exception) -> bool:
        """True for a per-day free-tier 429, which won't recover within a retry window."""
        if getattr(error, "status_code", None) != 429:
            return False
        message = str(error).lower()
        return any(marker in message for marker in DAILY_QUOTA_MARKERS)

    @staticmethod
    def _is_transient_error(error: Exception) -> bool:
        """Detect retryable API errors via HTTP status code, then message markers.

        The status code is the reliable signal: agno's ModelProviderError exposes
        ``status_code`` but hides the message behind an opaque ``<Response [...]>``.
        A per-day quota 429 is treated as non-transient — retrying in seconds is
        pointless since the daily allowance only resets at midnight Pacific.
        """
        if MedicalImagingAgent._is_daily_quota_error(error):
            return False
        status = getattr(error, "status_code", None)
        if status in TRANSIENT_STATUS_CODES:
            return True
        if status is not None:  # a definite non-transient status (e.g. 400/401/404)
            return False
        message = str(error).lower()
        return any(marker in message for marker in TRANSIENT_ERROR_MARKERS)

    @staticmethod
    def _format_error(error: Exception) -> str:
        """Turn an opaque provider error into an actionable message using its status code."""
        if MedicalImagingAgent._is_daily_quota_error(error):
            return ("Daily free-tier quota exhausted for this model. It resets at midnight "
                    "Pacific Time. To keep testing now, enable billing in Google AI Studio "
                    "for higher limits, or disable web search (fewer requests per analysis). "
                    "(HTTP 429)")
        status = getattr(error, "status_code", None)
        hint = STATUS_HINTS.get(status)
        if hint:
            return f"{hint} (HTTP {status})"
        return str(error)

    @staticmethod
    def _extract_token_usage(response: Any) -> dict[str, int]:
        """Extract token usage from an Agno response for observability (best effort).

        Handles both the agno 2.x ``RunMetrics`` object (int attributes) and the
        older dict-with-lists shape, without ever failing the analysis.
        """
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        metrics = getattr(response, "metrics", None)
        if metrics is None:
            return usage
        try:
            for key in usage:
                if isinstance(metrics, dict):
                    value = metrics.get(key, 0)
                else:  # RunMetrics or similar object
                    value = getattr(metrics, key, 0)
                usage[key] = int(sum(value) if isinstance(value, (list, tuple)) else (value or 0))
            if not usage["total_tokens"]:
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        except Exception:  # metrics format varies across agno versions; never fail the analysis
            pass
        return usage

    @staticmethod
    def _error_payload_in_content(content: Any) -> tuple[int, str] | None:
        """Detect a provider error JSON returned as the response content.

        agno 2.x sometimes places a transient provider error (e.g. 503) in
        ``content`` instead of raising. Returns (code, message) if detected.
        """
        text = (content or "")
        text = text.strip() if isinstance(text, str) else ""
        if not (text.startswith("{") and '"error"' in text and '"code"' in text):
            return None
        try:
            error = (json.loads(text).get("error") or {})
            code = error.get("code")
            if code:
                return int(code), str(error.get("message") or "Provider error")
        except (ValueError, TypeError):
            pass
        return None

    def analyze_image(self, image: AgnoImage, custom_prompt: str | None = None) -> dict[str, Any]:
        """Analyze medical image with retry on transient errors and token telemetry"""
        prompt = custom_prompt or self.get_analysis_prompt()
        start_time = time.time()
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.agent.run(prompt, images=[image])

                # agno 2.x may return a transient provider error as content
                # instead of raising; route it through the retry/format logic.
                payload = self._error_payload_in_content(getattr(response, "content", None))
                if payload is not None:
                    code, msg = payload
                    raise ProviderResponseError(msg, status_code=code)

                analysis_time = time.time() - start_time
                token_usage = self._extract_token_usage(response)

                result = {
                    "content": response.content,
                    "analysis_time": round(analysis_time, 2),
                    "model_used": self.model_id,
                    "token_usage": token_usage,
                    "attempts": attempt,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "success": True
                }

                logger.info(
                    f"Analysis completed in {analysis_time:.2f}s "
                    f"(attempt {attempt}/{self.max_retries}, tokens: {token_usage['total_tokens']})"
                )
                return result

            except Exception as e:
                last_error = e
                if attempt < self.max_retries and self._is_transient_error(e):
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Transient error on attempt {attempt}/{self.max_retries}: {e}. "
                        f"Retrying in {delay:.0f}s..."
                    )
                    time.sleep(delay)
                else:
                    break

        message = self._format_error(last_error)
        logger.error(f"Analysis failed after {attempt} attempt(s): {message}")
        return {
            "content": f"Analysis failed: {message}",
            "error": message,
            "status_code": getattr(last_error, "status_code", None),
            "attempts": attempt,
            "success": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_agent_info(self) -> dict[str, Any]:
        """Get information about the agent configuration"""
        return {
            "model_id": self.model_id,
            "tools": ["DuckDuckGo Search"] if self.enable_web_search else [],
            "web_search": self.enable_web_search,
            "capabilities": [
                "Medical Image Analysis",
                "Radiology Report Generation",
                "Research Literature Search",
                "Patient Education"
            ]
        }
