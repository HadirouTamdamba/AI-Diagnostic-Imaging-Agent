"""
Medical imaging analysis agent with enhanced capabilities
"""
import logging
import time
from typing import Any

from agno.agent import Agent
from agno.media import Image as AgnoImage
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

logger = logging.getLogger(__name__)

# Substrings identifying transient API errors worth retrying (rate limits, overload, network)
TRANSIENT_ERROR_MARKERS = (
    "429", "rate limit", "quota", "resource_exhausted",
    "500", "internal", "503", "unavailable", "overloaded",
    "timeout", "deadline", "connection",
)

class MedicalImagingAgent:
    """Enhanced medical imaging analysis agent"""

    def __init__(self, api_key: str, model_id: str = "gemini-2.0-flash",
                 max_retries: int = 3, retry_base_delay: float = 2.0):
        self.api_key = api_key
        self.model_id = model_id
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.agent = self._initialize_agent()

    def _initialize_agent(self) -> Agent:
        """Initialize the Gemini agent with tools"""
        try:
            return Agent(
                model=Gemini(
                    id=self.model_id,
                    api_key=self.api_key
                ),
                tools=[DuckDuckGoTools()],
                markdown=True,
                show_tool_calls=True,
                debug_mode=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise

    def get_analysis_prompt(self) -> str:
        """Enhanced analysis prompt with structured output"""
        return """
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

### 5. 📚 Research Context & References
**Important**: Use the DuckDuckGo search tool to find:
- Recent medical literature on similar cases
- Current treatment protocols and guidelines
- Relevant technological advances in imaging
- Patient education resources

Provide 2-3 authoritative medical references with brief summaries.

### 6. ⚠️ Limitations & Recommendations
- **Analysis Limitations**: Acknowledge any constraints of AI analysis
- **Professional Review**: Emphasize need for radiologist confirmation
- **Follow-up**: Suggest appropriate next steps

Format your response with clear markdown headers, bullet points, and professional medical terminology balanced with patient accessibility.
"""

    @staticmethod
    def _is_transient_error(error: Exception) -> bool:
        """Heuristic detection of retryable API errors (rate limit, overload, network)"""
        message = str(error).lower()
        return any(marker in message for marker in TRANSIENT_ERROR_MARKERS)

    @staticmethod
    def _extract_token_usage(response: Any) -> dict[str, int]:
        """Extract token usage from an Agno response for observability (best effort)"""
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        try:
            metrics = getattr(response, "metrics", None) or {}
            for key in usage:
                value = metrics.get(key, 0)
                usage[key] = int(sum(value) if isinstance(value, (list, tuple)) else value)
            if not usage["total_tokens"]:
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        except Exception:  # metrics format varies across agno versions; never fail the analysis
            pass
        return usage

    def analyze_image(self, image: AgnoImage, custom_prompt: str | None = None) -> dict[str, Any]:
        """Analyze medical image with retry on transient errors and token telemetry"""
        prompt = custom_prompt or self.get_analysis_prompt()
        start_time = time.time()
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.agent.run(prompt, images=[image])
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

        logger.error(f"Analysis failed after {attempt} attempt(s): {last_error}")
        return {
            "content": f"Analysis failed: {last_error}",
            "error": str(last_error),
            "attempts": attempt,
            "success": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_agent_info(self) -> dict[str, Any]:
        """Get information about the agent configuration"""
        return {
            "model_id": self.model_id,
            "tools": ["DuckDuckGo Search"],
            "capabilities": [
                "Medical Image Analysis",
                "Radiology Report Generation",
                "Research Literature Search",
                "Patient Education"
            ]
        }
