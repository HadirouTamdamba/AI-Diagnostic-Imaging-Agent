"""
Main Streamlit application for Medical Imaging Diagnosis Agent
"""
import streamlit as st
import logging
from typing import Optional
import time
import traceback


# Local imports
from config.settings import settings
from models.medical_agent import MedicalImagingAgent
from utils.image_processor import ImageProcessor
from utils.validators import SessionValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalImagingApp:
    """Main application class with enhanced features"""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.session_validator = SessionValidator()
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=settings.page_title,
            page_icon=settings.page_icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if "analysis_history" not in st.session_state:
            st.session_state.analysis_history = []
        if "GOOGLE_API_KEY" not in st.session_state:
            st.session_state.GOOGLE_API_KEY = None
        if "current_analysis" not in st.session_state:
            st.session_state.current_analysis = None
    
    def render_sidebar(self):
        """Render enhanced sidebar with configuration"""
        with st.sidebar:
            st.title("⚙️ Configuration")
            
            # API Key Configuration
            if not st.session_state.GOOGLE_API_KEY:
                st.subheader("🔑 API Configuration")
                api_key = st.text_input(
                    "Google API Key:",
                    type="password",
                    help="Get your API key from Google AI Studio"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Key", type="primary"):
                        if api_key:
                            st.session_state.GOOGLE_API_KEY = api_key
                            st.success("API Key saved successfully!")
                            st.rerun()
                        else:
                            st.error("Please enter a valid API key")
                
                with col2:
                    st.link_button(
                        "🔗 Get API Key",
                        "https://aistudio.google.com/apikey"
                    )
            else:
                st.success("✅ API Key configured")
                if st.button("🔄 Reset API Key"):
                    st.session_state.GOOGLE_API_KEY = None
                    st.rerun()
            
            st.divider()
            
            # Application Information
            st.subheader("ℹ️ Application Info")
            st.info(
                "**Medical Imaging Diagnosis Agent**\n\n"
                "Advanced AI-powered analysis of medical imaging data using "
                "state-of-the-art computer vision and radiological expertise."
            )
            
            # Model Information
            if st.session_state.GOOGLE_API_KEY:
                st.subheader("🤖 Model Information")
                st.write(f"**Model**: {settings.model_id}")
                st.write("**Capabilities**: Image Analysis, Research, Reporting")
            
            # Analysis History
            if st.session_state.analysis_history:
                st.subheader("📊 Analysis History")
                st.write(f"**Total Analyses**: {len(st.session_state.analysis_history)}")
                
                if st.button("🗑️ Clear History"):
                    st.session_state.analysis_history = []
                    st.rerun()
            
            # Disclaimer
            st.divider()
            st.error(
                "⚠️ **MEDICAL DISCLAIMER**\n\n"
                "This tool is for educational and informational purposes only. "
                "All analyses must be reviewed by qualified healthcare professionals. "
                "Do not make medical decisions based solely on this analysis."
            )
    
    def render_main_content(self):
        """Render main application content"""
        st.title("🏥 Medical Imaging Diagnosis Agent")
        st.markdown("---")
        
        # Check API key
        if not st.session_state.GOOGLE_API_KEY:
            st.warning("⚠️ Please configure your Google API key in the sidebar to continue")
            return
        
        # Image upload section
        st.subheader("📤 Upload Medical Image")
        
        uploaded_file = st.file_uploader(
            "Choose a medical image file",
            type=settings.supported_formats,
            help=f"Supported formats: {', '.join(settings.supported_formats).upper()}"
        )
        
        if uploaded_file is not None:
            self.process_uploaded_image(uploaded_file)
    
    def process_uploaded_image(self, uploaded_file):
        """Process and analyze uploaded image"""
        try:
            # Create columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Uploaded Image")
                
                # Process image
                with st.spinner("🔄 Processing image..."):
                    optimized_image, temp_path = self.image_processor.optimize_for_analysis(uploaded_file)
                
                # Display image
                st.image(
                    optimized_image,
                    caption=f"Optimized Image ({optimized_image.size[0]}x{optimized_image.size[1]})",
                    use_container_width=True
                )
                
                # Image info
                st.info(f"**File**: {uploaded_file.name}\n**Size**: {uploaded_file.size / 1024:.1f} KB")
                
                # Analysis button
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    self.perform_analysis(temp_path, uploaded_file.name)
            
            with col2:
                st.subheader("📋 Analysis Results")
                
                if st.session_state.current_analysis:
                    self.display_analysis_results(st.session_state.current_analysis)
                else:
                    st.info("👈 Click 'Analyze Image' to start analysis")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            logger.error(f"Image processing error: {traceback.format_exc()}")
    
    def perform_analysis(self, image_path: str, filename: str):
        """Perform medical image analysis"""
        try:
            # Initialize agent
            agent = MedicalImagingAgent(st.session_state.GOOGLE_API_KEY)
            
            # Create AgnoImage
            agno_image = self.image_processor.create_agno_image(image_path)
            
            # Perform analysis
            result = agent.analyze_image(agno_image)
            
            # Store result
            result["filename"] = filename
            st.session_state.current_analysis = result
            st.session_state.analysis_history.append(result)
            
            # Cleanup
            self.image_processor.cleanup_temp_files(image_path)
            
            if result["success"]:
                st.success(f"✅ Analysis completed in {result['analysis_time']}s")
                st.rerun()
            else:
                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            logger.error(f"Analysis error: {traceback.format_exc()}")
    
    def display_analysis_results(self, result: dict):
        """Display analysis results with enhanced formatting"""
        if result["success"]:
            st.success("✅ Analysis Complete")
            
            # Analysis metadata
            with st.expander("📊 Analysis Metadata"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Analysis Time**: {result['analysis_time']}s")
                    st.write(f"**Model Used**: {result['model_used']}")
                with col2:
                    st.write(f"**Timestamp**: {result['timestamp']}")
                    st.write(f"**File**: {result.get('filename', 'Unknown')}")
            
            # Main analysis content
            st.markdown("### 📋 Medical Analysis Report")
            st.markdown(result["content"])
            
            # Download report
            st.download_button(
                label="📥 Download Report",
                data=result["content"],
                file_name=f"medical_analysis_{result['timestamp'].replace(' ', '_').replace(':', '-')}.md",
                mime="text/markdown"
            )
        else:
            st.error("❌ Analysis Failed")
            st.error(result.get("error", "Unknown error"))
    
    def run(self):
        """Run the main application"""
        try:
            self.render_sidebar()
            self.render_main_content()
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"Application error: {traceback.format_exc()}")

# Application entry point
if __name__ == "__main__":
    app = MedicalImagingApp()
    app.run()