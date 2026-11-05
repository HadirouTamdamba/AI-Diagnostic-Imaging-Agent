"""Enhanced Main Streamlit application with latest optimizations for Medical Imaging Diagnosis Agent
"""
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

# Add src to path for imports before trying to import streamlit
current_dir = Path(__file__).parent
src_dir = current_dir
sys.path.insert(0, str(src_dir))

# Suppress specific warnings
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning:duckduckgo_search'
warnings.filterwarnings('ignore', category=RuntimeWarning, module='duckduckgo_search')
warnings.filterwarnings('ignore', category=ResourceWarning, message='.*unclosed file.*')

# Configure logging before other imports
_log_handlers = [logging.StreamHandler()]
try:
    # File logging is optional (may fail in read-only/containerized environments)
    _log_handlers.append(logging.FileHandler(os.getenv('LOG_FILE', 'medical_agent.log'), mode='a'))
except OSError:
    pass

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_log_handlers
)
logger = logging.getLogger(__name__)

try:
    import streamlit as st  # type: ignore[import]
except ImportError:
    logger.error("Streamlit is required to run this application.")
    logger.error("Install dependencies with `pip install -r requirements.txt`.")
    raise

try:
    # Local imports with error handling
    from config.i18n import LANGUAGES
    from config.i18n import t as _t
    from config.settings import settings
    from models.medical_agent import MedicalImagingAgent
    from utils.image_processor import ImageProcessor
    from utils.validators import SessionValidator

    logger.info("All modules imported successfully")
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    st.stop()
except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_agent(api_key: str, model_id: str, enable_web_search: bool,
              fallback_model_id: str = "") -> MedicalImagingAgent:
    """Cache the agent across Streamlit reruns to avoid re-initializing the Gemini client"""
    return MedicalImagingAgent(api_key, model_id=model_id, enable_web_search=enable_web_search,
                               fallback_model_id=fallback_model_id or None)


class MedicalImagingApp:
    """Enhanced main application class with improved error handling"""

    def __init__(self):
        """Initialize application with comprehensive error handling"""
        try:
            self.image_processor = ImageProcessor(
                max_size=settings.max_image_size,
                target_size=(settings.max_image_width, settings.max_image_height)
            )
            self.session_validator = SessionValidator()
            self.setup_page_config()
            self.initialize_session_state()
            logger.info("Application initialized successfully")
        except Exception as e:
            logger.error(f"Application initialization failed: {e}")
            st.error(f"Failed to initialize application: {e}")
            raise

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        try:
            st.set_page_config(
                page_title=settings.page_title,
                page_icon=settings.page_icon,
                layout="wide",
                initial_sidebar_state="expanded",
                menu_items={
                    'Get Help': 'https://github.com/HadirouTamdamba/AI-Diagnostic-Imaging-Agent',
                    'Report a bug': 'https://github.com/HadirouTamdamba/AI-Diagnostic-Imaging-Agent/issues',
                    'About': "Medical Imaging Diagnosis Agent - AI-powered Medical Image Analysis"
                }
            )
        except Exception as e:
            logger.warning(f"Page config setup failed: {e}")

    def initialize_session_state(self):
        """Initialize session state variables with validation"""
        try:
            # Initialize core session variables
            # API key can be pre-configured via environment/.env (Docker, cloud deployment).
            # Validate it here so a placeholder/malformed key does not falsely show as
            # "configured" and then fail later with an opaque API error.
            env_key = (settings.google_api_key or "").strip() or None
            if env_key and not self.session_validator.validate_api_key(env_key)["valid"]:
                logger.warning(
                    "Ignoring GOOGLE_API_KEY from environment: failed format validation "
                    "(is it still the .env placeholder 'your_google_api_key_here'?)"
                )
                env_key = None

            session_defaults = {
                "analysis_history": [],
                "GOOGLE_API_KEY": env_key,
                "current_analysis": None,
                "app_initialized": True,
                "error_count": 0,
                "current_page": "main",  # Add page navigation
                "enable_web_search": settings.enable_web_search,
                "ui_lang": settings.default_language,
            }

            for key, default_value in session_defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value

            # Validate session state
            self.session_validator.validate_session_state()

        except Exception as e:
            logger.error(f"Session state initialization failed: {e}")
            st.error("Failed to initialize session. Please refresh the page.")

    def t(self, key: str) -> str:
        """Translate a UI string into the current session language."""
        return _t(key, st.session_state.get("ui_lang", "en"))

    def _render_language_selector(self):
        """Render the EN/FR language selector."""
        codes = list(LANGUAGES.keys())
        current = st.session_state.get("ui_lang", "en")
        choice = st.radio(
            self.t("language"),
            options=codes,
            index=codes.index(current) if current in codes else 0,
            format_func=lambda c: LANGUAGES[c],
            horizontal=True,
            key="ui_lang_selector",
        )
        if choice != current:
            st.session_state.ui_lang = choice
            st.rerun()

    def render_sidebar(self):
        """Enhanced sidebar with better error handling and navigation"""
        try:
            with st.sidebar:
                st.title(self.t("configuration"))

                # Language selector
                self._render_language_selector()

                st.divider()

                # Navigation Menu
                self._render_navigation_menu()

                st.divider()

                # API Key Configuration Section
                self._render_api_config()

                st.divider()

                # Analysis Options (web search toggle)
                self._render_analysis_options()

                st.divider()

                # Application Information
                self._render_app_info()

                # Model Information (if API key is configured)
                if st.session_state.GOOGLE_API_KEY:
                    self._render_model_info()

                # Analysis History
                self._render_analysis_history()

                # System Status
                self._render_system_status()

                # Disclaimer
                st.divider()
                self._render_disclaimer()

        except Exception as e:
            logger.error(f"Sidebar rendering failed: {e}")
            st.error("Sidebar configuration error")

    def _render_navigation_menu(self):
        """Render navigation menu in sidebar"""
        st.subheader(self.t("navigation"))

        # Navigation buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button(self.t("home"), width='stretch'):
                st.session_state.current_page = "main"
                st.rerun()

        with col2:
            if st.button(self.t("about"), width='stretch'):
                st.session_state.current_page = "about"
                st.rerun()

    def _render_api_config(self):
        """Render API configuration section"""
        if not st.session_state.GOOGLE_API_KEY:
            st.subheader(self.t("api_configuration"))

            api_key = st.text_input(
                self.t("api_key_label"),
                type="password",
                help="Get your API key from Google AI Studio",
                placeholder=self.t("api_key_placeholder")
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button(self.t("save_key"), type="primary"):
                    if self._validate_and_save_api_key(api_key):
                        st.success(self.t("api_key_saved"))
                        time.sleep(1)
                        st.rerun()

            with col2:
                st.link_button(
                    self.t("get_api_key"),
                    "https://aistudio.google.com/apikey",
                    help="Get your free Google AI Studio API key"
                )
        else:
            st.success(self.t("api_key_configured"))
            if st.button(self.t("reset_api_key")):
                st.session_state.GOOGLE_API_KEY = None
                st.rerun()

    def _validate_and_save_api_key(self, api_key: str) -> bool:
        """Validate and save API key"""
        try:
            validation_result = self.session_validator.validate_api_key(api_key)
            if validation_result["valid"]:
                st.session_state.GOOGLE_API_KEY = api_key
                return True
            else:
                st.error(f"❌ {validation_result['error']}")
                return False
        except Exception as e:
            st.error(f"API key validation failed: {e}")
            return False

    def _render_analysis_options(self):
        """Render analysis options: live web search toggle."""
        st.subheader(self.t("analysis_options"))
        enabled = st.toggle(
            self.t("web_search_label"),
            value=st.session_state.get("enable_web_search", True),
            help=self.t("web_search_help"),
        )
        if enabled != st.session_state.get("enable_web_search"):
            st.session_state.enable_web_search = enabled
            st.rerun()
        st.caption(self.t("web_search_on") if enabled else self.t("web_search_off"))

    def _render_app_info(self):
        """Render application information"""
        st.subheader(self.t("app_info"))

        with st.expander("📋 Features", expanded=False):
            st.markdown("""
            **Core Capabilities:**
            - 🔍 Multi-modal medical image analysis
            - 📊 Structured diagnostic reporting
            - 🔬 Research literature integration
            - 👥 Patient-friendly explanations
            - 📥 Report export functionality
            """)

        with st.expander("🛠️ Technical Specs", expanded=False):
            st.markdown(f"""
            - **Model**: {settings.model_id}
            - **Max Image Size**: {settings.max_image_size / (1024*1024):.1f}MB
            - **Supported Formats**: {', '.join(settings.supported_formats).upper()}
            - **Analysis Timeout**: {settings.max_analysis_time}s
            """)

    def _render_model_info(self):
        """Render model information"""
        st.subheader(self.t("model_status"))
        try:
            # Cached agent (no re-initialization on every rerun)
            agent = get_agent(
                st.session_state.GOOGLE_API_KEY, settings.model_id,
                st.session_state.enable_web_search, settings.fallback_model_id,
            )
            agent_info = agent.get_agent_info()

            st.success(self.t("model_ready"))
            st.write(f"**{self.t('model')}**: {agent_info['model_id']}")
            st.write(f"**Tools**: {', '.join(agent_info['tools']) or '—'}")

        except Exception as e:
            st.error(self.t("model_error"))
            st.error(f"Failed to initialize: {str(e)}")

    def _render_analysis_history(self):
        """Render analysis history section"""
        if st.session_state.analysis_history:
            st.subheader(self.t("analysis_history"))

            history_count = len(st.session_state.analysis_history)
            successful_analyses = sum(1 for analysis in st.session_state.analysis_history if analysis.get("success", False))

            col1, col2 = st.columns(2)
            with col1:
                st.metric(self.t("total"), history_count)
            with col2:
                st.metric(self.t("successful"), successful_analyses)

            if st.button(self.t("clear_history")):
                st.session_state.analysis_history = []
                st.session_state.current_analysis = None
                st.rerun()

    def _render_system_status(self):
        """Render system status information"""
        st.subheader(self.t("system_status"))

        # Check system health
        status_items = {
            "Streamlit": "🟢 Running",
            "Dependencies": "🟢 Loaded" if 'agno' in sys.modules else "🔴 Missing",
            "Session": "🟢 Active" if st.session_state.get("app_initialized") else "🔴 Error"
        }

        for item, status in status_items.items():
            st.write(f"**{item}**: {status}")

    def _render_disclaimer(self):
        """Render medical disclaimer"""
        st.error(self.t("disclaimer"))

    def render_about_page(self):
        """Render the About page: what the platform is and what it does"""
        st.title(self.t("about_title"))
        st.markdown(self.t("about_intro"))
        st.markdown("---")

        st.subheader(self.t("about_what_h"))
        st.markdown(self.t("about_what"))

        st.subheader(self.t("about_how_h"))
        for i, col in enumerate(st.columns(4), start=1):
            with col.container(border=True):
                st.markdown(f"**{i}. {self.t(f'about_step{i}_h')}**")
                st.caption(self.t(f"about_step{i}_b"))

        left, right = st.columns(2)
        with left:
            st.subheader(self.t("about_tech_h"))
            st.markdown(self.t("about_tech"))
        with right:
            st.subheader(self.t("about_limits_h"))
            st.markdown(self.t("about_limits"))

        st.subheader(self.t("about_links_h"))
        repo = "https://github.com/HadirouTamdamba/AI-Diagnostic-Imaging-Agent"
        c1, c2, c3 = st.columns(3)
        with c1:
            st.link_button(self.t("about_repo"), repo, width='stretch')
        with c2:
            st.link_button(self.t("about_issues"), f"{repo}/issues", width='stretch')
        with c3:
            st.link_button(
                self.t("about_image"),
                "https://github.com/HadirouTamdamba/AI-Diagnostic-Imaging-Agent/pkgs/container/ai-diagnostic-imaging-agent",
                width='stretch',
            )

    def render_main_content(self):
        """Enhanced main content rendering with page navigation"""
        try:
            # Route to different pages based on session state
            if st.session_state.current_page == "about":
                self.render_about_page()
                return

            # Main application content (default page)
            # Header
            st.title(self.t("app_title"))
            st.markdown(self.t("app_subtitle"))
            st.markdown("---")

            # Check API key status
            if not st.session_state.GOOGLE_API_KEY:
                self._render_setup_instructions()
                return

            # Main application interface
            self._render_upload_interface()

        except Exception as e:
            logger.error(f"Main content rendering failed: {e}")
            st.error(f"Interface error: {e}")

    def _render_setup_instructions(self):
        """Render setup instructions when API key is missing"""
        st.warning(self.t("config_required"))

        st.markdown("""
        To get started:
        1. 🔑 Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
        2. 📝 Enter your API key in the sidebar
        3. 🚀 Start analyzing medical images!

        **Free tier**: the daily quota is small and varies by model. Check your usage in Google AI Studio.
        """)

        # Show demo information
        with st.expander("🎯 What This Tool Can Do", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **Image Types Supported:**
                - 🦴 X-rays (Chest, Skeletal, Dental)
                - 🧠 MRI Scans (Brain, Spine, Joints)
                - 🫁 CT Scans (Thoracic, Abdominal)
                - ❤️ Ultrasound (Cardiac, Obstetric)
                """)

            with col2:
                st.markdown("""
                **Analysis Features:**
                - 📋 Structured diagnostic reports
                - 🔍 Detailed findings identification
                - 📚 Medical literature integration
                - 👥 Patient-friendly explanations
                """)

    def _render_upload_interface(self):
        """Render the main upload and analysis interface"""
        st.subheader(self.t("upload_title"))

        # File uploader with enhanced validation
        uploaded_file = st.file_uploader(
            self.t("upload_choose"),
            type=settings.supported_formats,
            help=f"Supported formats: {', '.join(settings.supported_formats).upper()} | Max size: {settings.max_image_size / (1024*1024):.1f}MB"
        )

        if uploaded_file is not None:
            self._process_uploaded_image(uploaded_file)
        else:
            self._render_upload_placeholder()

    def _render_upload_placeholder(self):
        """Render placeholder content when no image is uploaded"""
        st.info(self.t("upload_placeholder"))

        # Sample images or instructions could go here
        with st.expander("💡 Tips for Best Results", expanded=False):
            st.markdown("""
            **Image Quality Guidelines:**
            - ✅ High resolution images (minimum 200x200 pixels)
            - ✅ Clear, well-contrasted medical images
            - ✅ Proper anatomical positioning
            - ✅ Standard medical imaging formats

            **Supported Image Types:**
            - JPEG/JPG files
            - PNG files
            - DICOM files (converted to display format)
            """)

    def _process_uploaded_image(self, uploaded_file):
        """Enhanced image processing with better error handling"""
        try:
            # Create main layout
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader(self.t("uploaded_image"))
                self._display_image_preview(uploaded_file)

            with col2:
                st.subheader(self.t("analysis_results"))
                self._display_analysis_section(uploaded_file)

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            st.error(f"Failed to process image: {e}")
            st.error("Please try uploading a different image or check the file format.")

    def _display_image_preview(self, uploaded_file):
        """Display image preview with metadata"""
        try:
            with st.spinner(self.t("processing_image")):
                optimized_image, temp_path = self.image_processor.optimize_for_analysis(uploaded_file)
                st.session_state.temp_image_path = temp_path

            # Display optimized image
            st.image(
                optimized_image,
                caption=f"Optimized: {optimized_image.size[0]}×{optimized_image.size[1]}px",
                width='stretch'
            )

            # Image metadata
            file_size_kb = uploaded_file.size / 1024
            with st.expander(self.t("image_details"), expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{self.t('filename')}**: {uploaded_file.name}")
                    st.write(f"**{self.t('size')}**: {file_size_kb:.1f} KB")
                with col2:
                    st.write(f"**{self.t('dimensions')}**: {optimized_image.size[0]}×{optimized_image.size[1]}px")
                    # In-memory optimized image has no .format; it is saved as PNG
                    st.write(f"**{self.t('format')}**: {optimized_image.format or 'PNG (optimized)'}")

        except Exception as e:
            logger.error(f"Image preview failed: {e}")
            st.error(f"Failed to process image: {e}")

    def _display_analysis_section(self, uploaded_file):
        """Display analysis section with enhanced controls"""
        # Analysis button
        if st.button(self.t("analyze_button"), type="primary", width='stretch'):
            if hasattr(st.session_state, 'temp_image_path'):
                self._perform_analysis(st.session_state.temp_image_path, uploaded_file.name)
            else:
                st.error("Please wait for image processing to complete")

        # Display current analysis if available
        if st.session_state.current_analysis:
            self._display_analysis_results(st.session_state.current_analysis)
        else:
            st.info(self.t("click_to_start"))

            # Show analysis preview
            with st.expander("🔍 What to Expect", expanded=False):
                st.markdown("""
                **Analysis will include:**
                - 🔍 Image type and quality assessment
                - 📋 Systematic medical findings
                - 🎯 Diagnostic assessment with confidence levels
                - 👥 Patient-friendly explanations
                - 📚 Research references and literature
                - ⏱️ Typical analysis time: 30-120 seconds
                """)

    def _perform_analysis(self, image_path: str, filename: str):
        """Enhanced analysis with comprehensive error handling and progress tracking"""
        try:
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Initialize agent
            status_text.text(self.t("init_agent"))
            progress_bar.progress(20)

            agent = get_agent(
                st.session_state.GOOGLE_API_KEY, settings.model_id,
                st.session_state.enable_web_search, settings.fallback_model_id,
            )

            # Step 2: Prepare image
            status_text.text(self.t("prep_image"))
            progress_bar.progress(40)

            agno_image = self.image_processor.create_agno_image(image_path)

            # Step 3: Perform analysis
            status_text.text(self.t("analyzing"))
            progress_bar.progress(60)

            start_time = time.time()
            result = agent.analyze_image(agno_image, language=st.session_state.ui_lang)
            analysis_time = time.time() - start_time

            # Step 4: Process results
            status_text.text("📋 Processing results...")
            progress_bar.progress(80)

            # Store results
            result["filename"] = filename
            result["analysis_time"] = round(analysis_time, 2)
            st.session_state.current_analysis = result
            st.session_state.analysis_history.append(result)

            # Step 5: Cleanup
            status_text.text(self.t("cleaning_up"))
            progress_bar.progress(100)

            self.image_processor.cleanup_temp_files(image_path)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Show results
            if result["success"]:
                st.success(self.t("analysis_success").format(t=result['analysis_time']))
                st.rerun()
            else:
                st.error(self.t("analysis_failed").format(e=result.get('error', 'Unknown error')))

        except Exception as e:
            logger.error(f"Analysis failed: {traceback.format_exc()}")
            st.error(f"Analysis error: {str(e)}")

            # Increment error counter
            st.session_state.error_count = st.session_state.get("error_count", 0) + 1

            # Show troubleshooting if multiple errors
            if st.session_state.error_count >= 2:
                with st.expander("🔧 Troubleshooting", expanded=True):
                    st.markdown("""
                    **Common Solutions:**
                    - 🔄 Try refreshing the page
                    - 🔑 Verify your API key is correct
                    - 🖼️ Try a different image format
                    - 🌐 Check your internet connection
                    - 📏 Ensure image is under 5MB
                    """)

    def _display_analysis_results(self, result: dict):
        """Enhanced results display with better formatting"""
        if result["success"]:
            # Success header
            st.success(self.t("analysis_complete"))

            # Quick metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(self.t("analysis_time"), f"{result['analysis_time']}s")
            with col2:
                st.metric(self.t("model"), result['model_used'])
            with col3:
                st.metric(self.t("timestamp"), result['timestamp'].split()[1])
            with col4:
                tokens = result.get("token_usage", {}).get("total_tokens", 0)
                st.metric(self.t("tokens"), f"{tokens:,}" if tokens else "n/a")

            # Main analysis content
            st.markdown(self.t("report_title"))
            st.markdown("---")
            st.markdown(result["content"])

            # Action buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                # Download report
                report_filename = f"medical_analysis_{result['timestamp'].replace(' ', '_').replace(':', '-')}.md"
                st.download_button(
                    label=self.t("download_report"),
                    data=result["content"],
                    file_name=report_filename,
                    mime="text/markdown",
                    width='stretch'
                )

            with col2:
                # Translate report to the other language (single request)
                report_lang = result.get("language", "en")
                target = "en" if report_lang == "fr" else "fr"
                translate_label = self.t("translate_to_en") if target == "en" else self.t("translate_to_fr")
                if st.button(translate_label, width='stretch'):
                    self._translate_current_report(result, target)

            with col3:
                # Start new analysis
                if st.button(self.t("new_analysis"), width='stretch'):
                    st.session_state.current_analysis = None
                    st.rerun()

        else:
            st.error(self.t("analysis_failed_header"))
            st.error(result.get("error", "Unknown error occurred"))

            # Retry option
            if st.button(self.t("retry_analysis")):
                st.session_state.current_analysis = None
                st.rerun()

    def _translate_current_report(self, result: dict, target_language: str):
        """Translate the current report to `target_language` in one request."""
        try:
            with st.spinner(self.t("translating")):
                agent = get_agent(
                    st.session_state.GOOGLE_API_KEY, settings.model_id,
                    st.session_state.enable_web_search,
                )
                translation = agent.translate_report(result["content"], target_language)

            if translation["success"]:
                result["content"] = translation["content"]
                result["language"] = target_language
                st.session_state.current_analysis = result
                st.success(self.t("translate_success"))
                st.rerun()
            else:
                st.error(translation.get("error", "Translation failed"))
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            st.error(f"Translation error: {e}")

    def run(self):
        """Main application runner with comprehensive error handling"""
        try:
            # Check if session is properly initialized
            if not st.session_state.get("app_initialized"):
                st.error("Application not properly initialized. Please refresh the page.")
                return

            # Render main interface
            self.render_sidebar()
            self.render_main_content()

            # Footer (only show on main page)
            if st.session_state.current_page == "main":
                st.markdown("---")
                st.markdown(
                    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
                    "Medical Imaging Diagnosis Agent v1.1 | "
                    "Developed by <a href='https://www.linkedin.com/in/hadirou-tamdamba/' target='_blank'>Hadirou Tamdamba</a> | "
                    f"{self.t('footer')}"
                    "</div>",
                    unsafe_allow_html=True
                )

        except Exception:
            logger.error(f"Application runtime error: {traceback.format_exc()}")
            st.error("Critical application error occurred. Please refresh the page.")

            # Emergency reset option
            if st.button("🚨 Emergency Reset"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# Application entry point with error handling
if __name__ == "__main__":
    try:
        app = MedicalImagingApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to start application: {e}")
        st.error("Please check your installation and try again.")
        logger.error(f"Application startup failed: {traceback.format_exc()}")
