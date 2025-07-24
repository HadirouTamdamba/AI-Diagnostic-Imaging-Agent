"""Enhanced Main Streamlit application with latest optimizations for Medical Imaging Diagnosis Agent 
"""
import streamlit as st
import logging
import sys
import os
from pathlib import Path
from typing import Optional
import time
import traceback
import warnings


# Suppress specific warnings
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning:duckduckgo_search'
warnings.filterwarnings('ignore', category=RuntimeWarning, module='duckduckgo_search')
warnings.filterwarnings('ignore', category=ResourceWarning, message='.*unclosed file.*')

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir
sys.path.insert(0, str(src_dir))

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('medical_agent.log', mode='a')
    ]
) 
logger = logging.getLogger(__name__)

try:
    # Local imports with error handling
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

class MedicalImagingApp:
    """Enhanced main application class with improved error handling"""
    
    def __init__(self):
        """Initialize application with comprehensive error handling"""
        try:
            self.image_processor = ImageProcessor()
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
            session_defaults = {
                "analysis_history": [],
                "GOOGLE_API_KEY": None,
                "current_analysis": None,
                "app_initialized": True,
                "error_count": 0,
                "current_page": "main"  # Add page navigation
            }
            
            for key, default_value in session_defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value
            
            # Validate session state
            self.session_validator.validate_session_state()
            
        except Exception as e:
            logger.error(f"Session state initialization failed: {e}")
            st.error("Failed to initialize session. Please refresh the page.")
    
    def render_sidebar(self):
        """Enhanced sidebar with better error handling and navigation"""
        try:
            with st.sidebar:
                st.title("⚙️ Configuration")
                
                # Navigation Menu
                self._render_navigation_menu()
                
                st.divider()
                
                # API Key Configuration Section
                self._render_api_config()
                
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
        st.subheader("🧭 Navigation")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.current_page = "main"
                st.rerun()
        
        with col2:
            if st.button("👨‍💻 About", use_container_width=True):
                st.session_state.current_page = "about"
                st.rerun()
    
    def _render_api_config(self):
        """Render API configuration section"""
        if not st.session_state.GOOGLE_API_KEY:
            st.subheader("🔑 API Configuration")
            
            api_key = st.text_input(
                "Google API Key:",
                type="password",
                help="Get your API key from Google AI Studio",
                placeholder="Enter your API key here..."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Key", type="primary"):
                    if self._validate_and_save_api_key(api_key):
                        st.success("✅ API Key saved!")
                        time.sleep(1)
                        st.rerun()
            
            with col2:
                st.link_button(
                    "🔗 Get API Key",
                    "https://aistudio.google.com/apikey",
                    help="Get your free Google AI Studio API key"
                )
        else:
            st.success("✅ API Key configured")
            if st.button("🔄 Reset API Key"):
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
    
    def _render_app_info(self):
        """Render application information"""
        st.subheader("ℹ️ Application Info")
        
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
        st.subheader("🤖 Model Status")
        try:
            # Test agent initialization
            agent = MedicalImagingAgent(st.session_state.GOOGLE_API_KEY)
            agent_info = agent.get_agent_info()
            
            st.success("🟢 Model Ready")
            st.write(f"**Model**: {agent_info['model_id']}")
            st.write(f"**Tools**: {', '.join(agent_info['tools'])}")
            
        except Exception as e:
            st.error("🔴 Model Error")
            st.error(f"Failed to initialize: {str(e)}")
    
    def _render_analysis_history(self):
        """Render analysis history section"""
        if st.session_state.analysis_history:
            st.subheader("📊 Analysis History")
            
            history_count = len(st.session_state.analysis_history)
            successful_analyses = sum(1 for analysis in st.session_state.analysis_history if analysis.get("success", False))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", history_count)
            with col2:
                st.metric("Successful", successful_analyses)
            
            if st.button("🗑️ Clear History"):
                st.session_state.analysis_history = []
                st.session_state.current_analysis = None
                st.rerun()
    
    def _render_system_status(self):
        """Render system status information"""
        st.subheader("💻 System Status")
        
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
        st.error(
            "⚠️ **MEDICAL DISCLAIMER**\n\n"
            "This tool is for **educational purposes only**. "
            "All analyses must be reviewed by qualified healthcare professionals. "
            "**Do not make medical decisions** based solely on this analysis."
        )
    
    def render_about_page(self):
        """Render the About Me page"""
        st.title("👨‍💻 About the Author")
        st.markdown("---")
        
        # Professional profile section
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Professional photo placeholder or icon
            st.markdown(
                """
                <div style='text-align: center; padding: 20px;'>
                    <div style='font-size: 80px; color: #1f77b4;'>👨‍💻</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown("""
            ## **Hadirou Tamdamba**
            ### *AI x ML(Ops) Engineer Consultant | Microsoft Certified Generative AI Engineer *
            #### *4+ Years Experience | Master's Degree in Applied Mathematics & Statistics*
            """)
        
        st.markdown("---")
        
        # Professional summary
        st.subheader("🎯 Professional Summary")
        st.markdown("""
        **Artificial Intelligence (AI) & Machine Learning (ML) Engineer** with **4+ years of experience** in the Data/AI field 
        for strategic projects across **Switzerland, Luxembourg, France, and Burkina Faso**. Holding a **Master's Degree in 
        Applied Mathematics and Statistics**, along with certifications in **AI & ML Engineering, MLOps, LLMOps, Cloud, and 
        Generative AI** (Microsoft, DataCamp, NASBA & PMI in USA, LinkedIn).
        
        I specialize in **predictive modeling, advanced analytics, and deploying innovative AI solutions** across diverse industries. 
        My expertise spans the complete AI lifecycle, from research and development to production deployment, ensuring seamless 
        integration while maintaining the highest standards of modern **MLOps, LLMOps, and DevOps best practices**.
        
        **Co-author of peer-reviewed scientific publications** and passionate about deriving actionable insights and 
        driving innovation in AI within dynamic, collaborative environments.
        """)
        
        # Industry Expertise & Project Experience
        st.subheader("🏭 Industry Experience & Strategic Projects")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🏥 Healthcare & Life Sciences**
            - Wastewater-Based-Epidemiology
            - Respiratory Viruses (COVID-19, Influenza)
            - Chronic Diseases (Diabetes, Heart Disease)
            - Cancer Research & HIV Studies
            - Statistical Analysis Plans
            - Clinical Data Modeling
            """)
        
        with col2:
            st.markdown("""
            **💰 Finance & Banking**
            - Credit Risk Modeling
            - Fraud Detection Systems
            - Financial Analytics
            - Risk Assessment Models
            - Algorithmic Solutions
            """)
        
        with col3:
            st.markdown("""
            **📊 Marketing & Real Estate**
            - Airbnb Market Analysis (Europe)
            - Recommendation Systems
            - Telecommunication Customer Churn Prediction
            - Social Media Sentiment Analysis
            - Predictive Analytics
            """)
        
        # Geographic Experience
        st.info("🌍 **International Experience**: Strategic AI projects across Switzerland, Luxembourg, France, and Burkina Faso")
        
        # Core Technical Expertise
        st.subheader("🚀 Core Technical Skills")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Programming & AI/ML:**
            - 🐍 **Python**: Scikit-Learn, PyTorch, TensorFlow, Keras, XGBoost
            - 🧠 **Deep Learning**: LSTM, RNNs, Transformers, Prophet
            - 🔍 **Computer Vision**: OpenCV, VGG 16/19, YOLO
            - 📊 **Data Science**: Skrub, Pandas, NumPy, Polars, PySpark, Pydantic
            - 📈 **Visualization**: Matplotlib, Seaborn, Plotly, Power BI
            - 💻 **Languages**: R, SQL, NoSQL, Java, JavaScript, C++, SAS
            """)
        
        with col2:
            st.markdown("""
            **MLOps & Cloud Platforms:**
            - ☁️ **AWS**: EC2, Lambda, S3, SageMaker, Route 53
            - 🔵 **Azure**: DevOps, ML Services, Cloud Computing
            - 🤖 **IBM Watson X**: AI Studio, ML Pipelines
            - 🐳 **DevOps**: Docker, Kubernetes, Terraform, Jenkins
            - 🔄 **CI/CD**: GitHub Actions, GitLab, Azure DevOps
            - 📊 **Data Platforms**: Snowflake, Databricks, BigQuery
            """)
        
        # Developer Tools & Frameworks
        st.markdown("---")
        
        dev_col1, dev_col2 = st.columns(2)
        
        with dev_col1:
            st.markdown("""
            **🛠️ Developer Tools & APIs:**
            - Version Control: Git, GitHub, GitLab
            - IDEs: VSCode, Cursor, Jupyter, PyCharm
            - Testing: Pytest, Unit Testing
            - APIs: OpenAI, GPT, Gemini, Claude, LLaMA, Mistral
            - Frameworks: LangChain, RAG, HuggingFace Transformers
            - Web: Flask, FastAPI, Django, Streamlit, React, Node.js
            """)
        
        with dev_col2:
            st.markdown("""
            **💾 Data & Infrastructure:**
            - Databases: MySQL, MongoDB, Oracle, PostgreSQL
            - Big Data: Apache Spark, Hadoop, Linux/Unix
            - Analytics: SPSS, Stata, SAS, Statistical Modeling
            - NLP: NLTK, Text Processing, Sentiment Analysis
            - MLOps: MLflow, Model Monitoring, Deployment
            - Documentation: LaTeX, Technical Writing
            """)
        
        # Services offered
        st.subheader("💼 Consulting Services")
        
        services = [
            {
                "title": "🎯 AI Strategy & Digital Transformation",
                "description": """Strategic planning and execution of AI initiatives tailored to your industry. 
                From initial assessment to full-scale implementation across Healthcare, Finance, and Enterprise sectors."""
            },
            {
                "title": "🔧 MLOps & DevOps Infrastructure",
                "description": """Design and implement robust MLOps pipelines with modern DevOps practices. 
                Cloud-native architectures on AWS, GCP, and IBM Watson X with automated CI/CD workflows."""
            },
            {
                "title": "🏥 Healthcare AI Solutions",
                "description": """Specialized medical AI applications including diagnostic imaging, clinical decision support, 
                and patient monitoring systems with full regulatory compliance."""
            },
            {
                "title": "💰 Financial AI & Risk Management",
                "description": """Advanced financial models for credit risk assessment, fraud detection, algorithmic trading, 
                and regulatory compliance solutions."""
            },
            {
                "title": "🤖 Custom AI Application Development",
                "description": """End-to-end development of production-ready AI applications with modern software engineering 
                practices, scalable architecture, and comprehensive testing."""
            },
            {
                "title": "📚 Team Training & Technology Transfer",
                "description": """Comprehensive training programs in AI/ML technologies, MLOps methodologies, 
                cloud platforms, and modern development practices for your technical teams."""
            }
        ]
        
        for service in services:
            with st.expander(service["title"], expanded=False):
                st.markdown(service["description"])
        
        # Featured Projects Portfolio
        st.subheader("🔬 Featured Projects Portfolio")
        
        projects = [
            {
                "title": "🏥 Medical Imaging Diagnosis Agent",
                "description": "AI-powered medical image analysis using Google's Gemini AI with production-ready Streamlit interface",
                "tech": "Python, Streamlit, Google AI, Computer Vision, Medical Imaging",
                "link": "Current Application",
                "industry": "Healthcare"
            },
            {
                "title": "💳 Credit Risk Modeling & Payment Default Prediction",
                "description": "Advanced ML models for predicting payment defaults in financial services with comprehensive risk assessment",
                "tech": "Python, Scikit-learn, XGBoost, AWS SageMaker, Statistical Modeling",
                "link": "https://github.com/HadirouTamdamba/Prediction-payments-default-finance",
                "industry": "Finance"
            },
            {
                "title": "🛡️ Fraud Detection in Financial Transactions (EU)",
                "description": "Real-time fraud detection system for European financial transactions with advanced anomaly detection",
                "tech": "Python, TensorFlow, Apache Kafka, Docker, Real-time Processing",
                "link": "https://github.com/HadirouTamdamba/Fraud-Detection-in-Financial-Transactions-EU",
                "industry": "Finance"
            },
            {
                "title": "🦠 Wastewater-Based Epidemiology Research",
                "description": "Statistical analysis and predictive modeling for COVID-19 and Influenza tracking through wastewater monitoring",
                "tech": "R, Python, Statistical Analysis, Time Series, Epidemiological Modeling",
                "link": "Research Publications",
                "industry": "Healthcare"
            },
            {
                "title": "🏠 Airbnb Market Analysis (Europe)",
                "description": "Comprehensive market analysis and price prediction models for European Airbnb markets",
                "tech": "Python, Pandas, Scikit-learn, Data Visualization, Market Analytics",
                "link": "Analytics Project",
                "industry": "Real Estate"
            },
            {
                "title": "📱 Telecommunication Customer Churn Prediction",
                "description": "Machine learning solution for predicting and preventing customer churn in telecommunications",
                "tech": "Python, XGBoost, Feature Engineering, Predictive Analytics",
                "link": "ML Project",
                "industry": "Telecommunications"
            }
        ]
        
        for project in projects:
            with st.expander(f"{project['title']} - {project['industry']}", expanded=False):
                st.markdown(f"""
                **Description**: {project['description']}
                
                **Technologies**: {project['tech']}
                
                **Industry Focus**: {project['industry']}
                """)
                
                if project['link'].startswith('http'):
                    st.link_button("🔗 View Project", project['link'])
                else:
                    st.info(f"📍 {project['link']}")
        
        # Certifications & achievements
        st.subheader("🏆 Certifications & Academic Background")
        
        cert_col1, cert_col2 = st.columns(2)
        
        with cert_col1:
            st.markdown("""
            **🎓 Education & Certifications:**
            - **Master's Degree in Applied Mathematics & Statistics**
            - 🎯 **Microsoft Certified: Generative AI Engineer**
            - 📊 **DataCamp Certified: ML & Data Science**
            - 🇺🇸 **NASBA & PMI (USA) Certified**
            - 💼 **LinkedIn Learning Certifications**
            """)
        
        with cert_col2:
            st.markdown("""
            **📚 Research & Publications:**
            - 🔬 **Co-author of Peer-Reviewed Scientific Publications**
            - 📝 **Research in Wastewater-Based Epidemiology**
            - 🦠 **COVID-19 & Influenza Studies**
            - 📊 **Statistical Methodology Expert**
            - 🧪 **Healthcare Analytics Research**
            """)
        
        # Soft Skills
        st.subheader("💡 Professional Competencies")
        
        soft_skills_col1, soft_skills_col2 = st.columns(2)
        
        with soft_skills_col1:
            st.markdown("""
            **🤝 Leadership & Collaboration:**
            - Autonomy & Self-direction
            - Teamwork & Cross-functional Collaboration
            - Agility & Adaptability to Fast-changing Technologies
            - Multi-tasking & Project Management
            """)
        
        with soft_skills_col2:
            st.markdown("""
            **🧠 Problem-Solving & Communication:**
            - Creative Thinking & Innovation
            - Self-learner & Continuous Improvement
            - Excellent Problem-solving Skills
            - Complex Technical Communication to Non-technical Stakeholders
            """)
        
        # Technology Stack
        st.subheader("⚙️ Comprehensive Technology Stack")
        
        tech_tabs = st.tabs(["🐍 Python Ecosystem", "☁️ Cloud & MLOps", "📊 Data & Analytics", "🌐 Web & APIs"])
        
        with tech_tabs[0]:
            st.markdown("""
            **Core Python Libraries:**
            - **ML/DL**: Scikit-Learn, PyTorch, TensorFlow, Keras, XGBoost, Prophet
            - **Deep Learning**: LSTM, RNNs, Transformers, Neural Networks
            - **Computer Vision**: OpenCV, VGG 16/19, YOLO, Image Processing
            - **Data Processing**: Pandas, NumPy, Polars, PySpark, Pydantic, Skrub
            - **Visualization**: Matplotlib, Seaborn, Plotly, Interactive Dashboards
            - **NLP**: NLTK, Transformers, Text Processing, Sentiment Analysis
            - **Testing**: Pytest, Unit Testing, Quality Assurance
            """)
        
        with tech_tabs[1]:
            st.markdown("""
            **Cloud Platforms & MLOps:**
            - **AWS**: EC2, Lambda, S3, SageMaker, Route 53/Nginx
            - **Microsoft Azure**: ML Services, DevOps, Cloud Computing
            - **IBM Watson X**: AI Studio, Watson Machine Learning
            - **MLOps**: MLflow, Model Monitoring, Deployment Pipelines
            - **Containers**: Docker, Kubernetes/K8s, Container Orchestration
            - **CI/CD**: GitHub Actions, GitLab, Azure DevOps, Jenkins
            - **Infrastructure**: Terraform, Shell/Bash, Linux/Unix
            """)
        
        with tech_tabs[2]:
            st.markdown("""
            **Data Platforms & Analytics:**
            - **Big Data**: Apache Spark, Hadoop, Distributed Computing
            - **Cloud Data**: Snowflake, Databricks, BigQuery
            - **Databases**: MySQL, MongoDB, Oracle, PostgreSQL
            - **Statistical Software**: R, SAS, Stata, SPSS
            - **BI Tools**: Power BI, Data Visualization, Reporting
            - **Data Engineering**: ETL Pipelines, Data Processing
            """)
        
        with tech_tabs[3]:
            st.markdown("""
            **Web Development & APIs:**
            - **Web Frameworks**: Flask, FastAPI, Django, Streamlit
            - **Frontend**: JavaScript, Node.js, React, HTML/CSS
            - **AI APIs**: OpenAI, GPT, Gemini, Claude, LLaMA, Mistral
            - **LLM Frameworks**: LangChain, RAG, HuggingFace
            - **Development**: Git, VSCode, API Development
            - **Deployment**: Render, Cloud Deployment, Web Services
            """)
        
        # Contact information
        st.subheader("📞 Let's Connect & Collaborate")
        st.markdown("---")
        
        # Contact buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.link_button(
                "💼 LinkedIn Profile",
                "https://www.linkedin.com/in/hadirou-tamdamba/",
                use_container_width=True
            )
        
        with col2:
            st.link_button(
                "🔗 GitHub Portfolio",
                "https://github.com/HadirouTamdamba",
                use_container_width=True
            )
        
        with col3:
            if st.button("📧 Email Contact", use_container_width=True):
                st.info("📧 **Email**: hadirou.tamdamba@outlook.fr")
        
        # Call to action
        st.markdown("---")
        st.success("""
        **🚀 Ready to Transform Your Business with AI?** 
        
        Whether you're in **Healthcare**, **Finance**, or **Technology**, I specialize in delivering 
        end-to-end AI solutions that drive real business value. From initial strategy to production 
        deployment on cloud platforms, let's discuss how cutting-edge AI can transform your operations.
        
        **Key Differentiators:**
        - ✅ Industry-specific expertise across Healthcare, Finance & Enterprise
        - ✅ Full-stack AI development with modern MLOps/DevOps practices  
        - ✅ Multi-cloud proficiency (AWS, Azure, IBM Watson X)
        - ✅ Production-ready solutions with enterprise-grade security
        - ✅ Comprehensive team training and knowledge transfer
        """)
        
        # Industries served
        st.subheader("🎯 Domain Expertise & Applications")
        
        industries_tabs = st.tabs(["🏥 Healthcare & Research", "💰 Finance & Banking", "📊 Marketing & Business"])
        
        with industries_tabs[0]:
            st.markdown("""
            **Healthcare AI & Research Applications:**
            - 🔬 **Wastewater-Based Epidemiology**: COVID-19 & Influenza tracking and prediction
            - 🏥 **Medical Image Analysis**: Diagnostic AI for radiology and medical imaging
            - 🦠 **Infectious Disease Modeling**: Respiratory viruses, HIV, and epidemic forecasting
            - 💊 **Chronic Disease Research**: Diabetes, Heart Disease, Cancer analytics
            - 📋 **Clinical Data Analysis**: Statistical Analysis Plans, Missing Data Handling
            - 🧬 **Biostatistics**: Advanced statistical modeling for healthcare research
            - 📊 **Epidemiological Studies**: Population health analytics and public health insights
            """)
        
        with industries_tabs[1]:
            st.markdown("""
            **Financial Services & Banking AI:**
            - 📊 **Credit Risk Assessment**: Advanced models for loan default prediction and risk scoring
            - 🛡️ **Fraud Detection**: Real-time transaction monitoring and anomaly detection systems
            - 💳 **Payment Analytics**: Financial transaction analysis and pattern recognition
            - 🏦 **Banking Solutions**: Customer analytics and financial product optimization
            - 💰 **Financial Forecasting**: Market prediction and economic modeling
            """)
        
        with industries_tabs[2]:
            st.markdown("""
            **Marketing & Business Intelligence:**
            - 🏠 **Real Estate Analytics**: Airbnb market analysis and price prediction (Europe)
            - 🎯 **Recommendation Systems**: Personalized content and product recommendations
            - 📱 **Customer Analytics**: Churn prediction and retention strategies (Telecommunications)
            - 💬 **Social Media Intelligence**: Sentiment analysis and brand monitoring
            - 📊 **Business Intelligence**: KPI dashboards and performance analytics  
            - 🎪 **Marketing Optimization**: Campaign effectiveness and customer segmentation
            - 📈 **Predictive Analytics**: Sales forecasting and demand prediction
            """)
        
        # Project showcase with technical details
        st.subheader("🔬 Technical Project Showcase")
        st.markdown("""
        This **Medical Imaging Diagnosis Agent** exemplifies my approach to building production-ready AI applications:
        
        **🏗️ Architecture & Engineering:**
        - **Scalable Design**: Modular architecture with proper separation of concerns
        - **Cloud Integration**: Google AI API integration with error handling and rate limiting  
        - **Security**: Secure API key management and data handling protocols
        - **User Experience**: Professional Streamlit interface with real-time feedback
        - **DevOps Ready**: Containerizable deployment with comprehensive logging
        
        **🔧 Technical Implementation:**
        - **Modern Python**: Type hints, proper error handling, and clean code principles
        - **AI Integration**: Advanced prompt engineering for medical image analysis
        - **Performance**: Optimized image processing with memory management
        - **Monitoring**: Detailed logging and session state management
        - **Extensibility**: Plugin-ready architecture for additional AI models
        """)
    
    def render_main_content(self):
        """Enhanced main content rendering with page navigation"""
        try:
            # Route to different pages based on session state
            if st.session_state.current_page == "about":
                self.render_about_page()
                return
            
            # Main application content (default page)
            # Header
            st.title("🏥 Medical Imaging Diagnosis Agent")
            st.markdown("### Advanced AI-Powered Medical Image Analysis")
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
        st.warning("⚠️ **Configuration Required**")
        
        st.markdown("""
        To get started:
        1. 🔑 Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
        2. 📝 Enter your API key in the sidebar
        3. 🚀 Start analyzing medical images!
        
        **Free Tier**: 1,500 requests per day at no cost.
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
        st.subheader("📤 Upload Medical Image")
        
        # File uploader with enhanced validation
        uploaded_file = st.file_uploader(
            "Choose a medical image file",
            type=settings.supported_formats,
            help=f"Supported formats: {', '.join(settings.supported_formats).upper()} | Max size: {settings.max_image_size / (1024*1024):.1f}MB"
        )
        
        if uploaded_file is not None:
            self._process_uploaded_image(uploaded_file)
        else:
            self._render_upload_placeholder()
    
    def _render_upload_placeholder(self):
        """Render placeholder content when no image is uploaded"""
        st.info("👆 **Upload a medical image to begin analysis**")
        
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
                st.subheader("📷 Uploaded Image")
                self._display_image_preview(uploaded_file)
            
            with col2:
                st.subheader("📋 Analysis Results")
                self._display_analysis_section(uploaded_file)
                
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            st.error(f"Failed to process image: {e}")
            st.error("Please try uploading a different image or check the file format.")
    
    def _display_image_preview(self, uploaded_file):
        """Display image preview with metadata"""
        try:
            with st.spinner("🔄 Processing image..."):
                optimized_image, temp_path = self.image_processor.optimize_for_analysis(uploaded_file)
                st.session_state.temp_image_path = temp_path
            
            # Display optimized image
            st.image(
                optimized_image,
                caption=f"Optimized: {optimized_image.size[0]}×{optimized_image.size[1]}px",
                use_container_width=True
            )
            
            # Image metadata
            file_size_kb = uploaded_file.size / 1024
            with st.expander("📊 Image Details", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Filename**: {uploaded_file.name}")
                    st.write(f"**Size**: {file_size_kb:.1f} KB")
                with col2:
                    st.write(f"**Dimensions**: {optimized_image.size[0]}×{optimized_image.size[1]}px")
                    st.write(f"**Format**: {optimized_image.format}")
            
        except Exception as e:
            logger.error(f"Image preview failed: {e}")
            st.error(f"Failed to process image: {e}")
    
    def _display_analysis_section(self, uploaded_file):
        """Display analysis section with enhanced controls"""
        # Analysis button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            if hasattr(st.session_state, 'temp_image_path'):
                self._perform_analysis(st.session_state.temp_image_path, uploaded_file.name)
            else:
                st.error("Please wait for image processing to complete")
        
        # Display current analysis if available
        if st.session_state.current_analysis:
            self._display_analysis_results(st.session_state.current_analysis)
        else:
            st.info("👈 Click 'Analyze Image' to start medical analysis")
            
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
            status_text.text("🤖 Initializing AI agent...")
            progress_bar.progress(20)
            
            agent = MedicalImagingAgent(st.session_state.GOOGLE_API_KEY)
            
            # Step 2: Prepare image
            status_text.text("🖼️ Preparing image for analysis...")
            progress_bar.progress(40)
            
            agno_image = self.image_processor.create_agno_image(image_path)
            
            # Step 3: Perform analysis
            status_text.text("🔍 Analyzing medical image... This may take up to 2 minutes...")
            progress_bar.progress(60)
            
            start_time = time.time()
            result = agent.analyze_image(agno_image)
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
            status_text.text("🧹 Cleaning up...")
            progress_bar.progress(100)
            
            self.image_processor.cleanup_temp_files(image_path)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Show results
            if result["success"]:
                st.success(f"✅ Analysis completed successfully in {result['analysis_time']}s")
                st.rerun()
            else:
                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                
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
            st.success("✅ Analysis Complete")
            
            # Quick metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Analysis Time", f"{result['analysis_time']}s")
            with col2:
                st.metric("Model", result['model_used'])
            with col3:
                st.metric("Timestamp", result['timestamp'].split()[1])
            
            # Main analysis content
            st.markdown("### 📋 Medical Analysis Report")
            st.markdown("---")
            st.markdown(result["content"])
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download report
                report_filename = f"medical_analysis_{result['timestamp'].replace(' ', '_').replace(':', '-')}.md"
                st.download_button(
                    label="📥 Download Report",
                    data=result["content"],
                    file_name=report_filename,
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                # Share analysis (copy to clipboard simulation)
                if st.button("📋 Copy Report", use_container_width=True):
                    st.success("Report copied to clipboard! (use Ctrl+C)")
            
            with col3:
                # Start new analysis
                if st.button("🔄 New Analysis", use_container_width=True):
                    st.session_state.current_analysis = None
                    st.rerun()
            
        else:
            st.error("❌ Analysis Failed")
            st.error(result.get("error", "Unknown error occurred"))
            
            # Retry option
            if st.button("🔄 Retry Analysis"):
                st.session_state.current_analysis = None
                st.rerun()
    
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
                    "Medical Imaging Diagnosis Agent v1.0 | "
                    "Developed by <a href='https://www.linkedin.com/in/hadirou-tamdamba/' target='_blank'>Hadirou Tamdamba</a> | "
                    "For educational purposes only | "
                    "Always consult healthcare professionals"
                    "</div>",
                    unsafe_allow_html=True
                )
            
        except Exception as e:
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