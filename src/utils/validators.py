"""
Validation utilities for session management and data integrity
"""
import streamlit as st
import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class SessionValidator:
    """Session validation and management utilities"""
    
    def __init__(self):
        self.max_history_size = 50
        self.session_timeout = 3600  # 1 hour
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate Google API key format"""
        if not api_key:
            return {"valid": False, "error": "API key is required"}
        
        if len(api_key) < 20:
            return {"valid": False, "error": "API key appears to be too short"}
        
        if not api_key.startswith("AIza"):
            return {"valid": False, "error": "Invalid Google API key format"}
        
        return {"valid": True}
    
    def clean_session_history(self):
        """Clean old session history to prevent memory issues"""
        if "analysis_history" in st.session_state:
            history = st.session_state.analysis_history
            if len(history) > self.max_history_size:
                st.session_state.analysis_history = history[-self.max_history_size:]
                logger.info(f"Cleaned session history, kept {self.max_history_size} entries")
    
    def validate_session_state(self) -> bool:
        """Validate current session state"""
        try:
            # Check required session variables
            required_vars = ["GOOGLE_API_KEY", "analysis_history"]
            for var in required_vars:
                if var not in st.session_state:
                    st.session_state[var] = None if var == "GOOGLE_API_KEY" else []
            
            # Clean old history
            self.clean_session_history()
            
            return True
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False 
             