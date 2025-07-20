 """
Medical imaging analysis agent with enhanced capabilities
"""
import logging
from typing import List, Dict, Any, Optional
from agno.agent import Agent
from agno.models.google import Gemini 
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st

logger = logging.getLogger(__name__)

class MedicalImagingAgent:
    """Enhanced medical imaging analysis agent"""
    
    def __init__(self, api_key: str, model_id: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_id = model_id
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
    
    def analyze_image(self, image: AgnoImage, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Analyze medical image with enhanced error handling"""
        try:
            prompt = custom_prompt or self.get_analysis_prompt()
            
            # Add timing and monitoring
            import time
            start_time = time.time()
            
            with st.spinner("🔄 Analyzing medical image... This may take up to 2 minutes."):
                response = self.agent.run(prompt, images=[image])
                
            analysis_time = time.time() - start_time
            
            # Structure the response
            result = {
                "content": response.content,
                "analysis_time": round(analysis_time, 2),
                "model_used": self.model_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
            logger.info(f"Analysis completed in {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "content": f"Analysis failed: {str(e)}",
                "error": str(e),
                "success": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
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