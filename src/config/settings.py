"""
Configuration settings for Medical Imaging Diagnosis Agent
Fixed for Pydantic v2+ compatibility
"""
import os
from typing import Dict, Any, List
from pydantic_settings import BaseSettings
from pydantic import field_validator
import streamlit as st

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # API Configuration
    google_api_key: str = ""
    model_id: str = "gemini-2.0-flash"
    
    # Image Processing
    max_image_size: int = 5 * 1024 * 1024  # 5MB
    supported_formats: List[str] = ["jpg", "jpeg", "png", "dicom"]
    max_image_width: int = 1024
    max_image_height: int = 1024
    
    # Analysis Configuration
    max_analysis_time: int = 120  # seconds
    confidence_threshold: float = 0.7
    
    # UI Configuration
    page_title: str = "🏥 Medical Imaging Diagnosis Agent"
    page_icon: str = "🩻"
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    @field_validator('google_api_key')
    @classmethod
    def validate_api_key(cls, v):
        if not v and st.session_state.get('GOOGLE_API_KEY'):
            return st.session_state.get('GOOGLE_API_KEY', "")
        return v
    
    @field_validator('supported_formats')
    @classmethod
    def validate_formats(cls, v):
        if not isinstance(v, list):
            return ["jpg", "jpeg", "png", "dicom"]
        return v

# Global settings instance
settings = Settings()