"""
Unit tests for the i18n helper.
"""
from src.config.i18n import LANGUAGES, TRANSLATIONS, t


class TestI18n:

    def test_languages_available(self):
        assert set(LANGUAGES) == {"en", "fr"}

    def test_translation_returns_language_string(self):
        assert t("analyze_button", "fr") == "🔍 Analyser l'image"
        assert t("analyze_button", "en") == "🔍 Analyze Image"

    def test_unknown_language_falls_back_to_english(self):
        assert t("analyze_button", "de") == t("analyze_button", "en")

    def test_unknown_key_returns_key(self):
        assert t("__missing__", "fr") == "__missing__"

    def test_en_and_fr_have_the_same_keys(self):
        assert set(TRANSLATIONS["en"]) == set(TRANSLATIONS["fr"])
