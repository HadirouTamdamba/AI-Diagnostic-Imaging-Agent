"""
Validation utilities for session management and data integrity
"""
import logging
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

class SessionValidator:
    """Session validation and management utilities"""

    # Placeholder shipped in .env.example — never a usable key
    PLACEHOLDER_KEYS = frozenset({"your_google_api_key_here"})

    def __init__(self):
        self.max_history_size = 50
        self.session_timeout = 3600  # 1 hour

    def validate_api_key(self, api_key: str) -> dict[str, Any]:
        """Validate Google API key format"""
        api_key = (api_key or "").strip()

        if not api_key:
            return {"valid": False, "error": "API key is required"}

        if api_key in self.PLACEHOLDER_KEYS:
            return {"valid": False, "error": "API key is still the placeholder from .env.example — replace it with a real key"}

        if len(api_key) < 20:
            return {"valid": False, "error": "API key appears to be too short"}

        if not api_key.startswith("AIza"):
            return {"valid": False, "error": "Invalid Google API key format (expected to start with 'AIza')"}

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
