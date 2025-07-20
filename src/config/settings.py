"""
Configuration settings for Medical Imaging Diagnosis Agent
"""
import os
from typing import Dict, Any
from pydantic import BaseSettings, validator
import streamlit as st

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # API Configuration
    google_api_key: str = ""
    model_id: str = "gemini-2.0-flash"
    
    # Image Processing
    max_image_size: int = 5 * 1024 * 1024  # 5MB
    supported_formats: list = ["jpg", "jpeg", "png", "dicom"]
    max_image_width: int = 1024
    max_image_height: int = 1024
    
    # Analysis Configuration
    max_analysis_time: int = 120  # seconds
    confidence_threshold: float = 0.7
    
    # UI Configuration
    page_title: str = "🏥 Medical Imaging Diagnosis Agent"
    page_icon: str = "🩻"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @validator('google_api_key')
    def validate_api_key(cls, v):
        if not v and not st.session_state.get('GOOGLE_API_KEY'):
            return ""
        return v or st.session_state.get('GOOGLE_API_KEY', "")

# Global settings instance
settings = Settings()