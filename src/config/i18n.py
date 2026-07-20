"""
Lightweight i18n for the main UI (English / French).

The About page is intentionally not translated. `t(key, lang)` falls back to the
English string, then to the key itself, so a missing translation never crashes.
"""

LANGUAGES = {"en": "🇬🇧 English", "fr": "🇫🇷 Français"}

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Header / footer
        "app_title": "🏥 Medical Imaging Diagnosis Agent",
        "app_subtitle": "### Advanced AI-Powered Medical Image Analysis",
        "footer": "For educational purposes only | Always consult healthcare professionals",
        # Sidebar sections
        "configuration": "⚙️ Configuration",
        "language": "🌐 Language",
        "navigation": "🧭 Navigation",
        "home": "🏠 Home",
        "about": "👨‍💻 About",
        "api_configuration": "🔑 API Configuration",
        "api_key_label": "Google API Key:",
        "api_key_placeholder": "Enter your API key here...",
        "save_key": "💾 Save Key",
        "get_api_key": "🔗 Get API Key",
        "api_key_saved": "✅ API Key saved!",
        "api_key_configured": "✅ API Key configured",
        "reset_api_key": "🔄 Reset API Key",
        "analysis_options": "🔎 Analysis Options",
        "web_search_label": "Live web search (references)",
        "web_search_help": (
            "ON: the agent searches the web for live medical references (richer "
            "report, several API requests). OFF: one request per analysis — best to "
            "conserve the free-tier daily quota."
        ),
        "web_search_on": "🌐 Live references ON",
        "web_search_off": "⚡ Quota-saver mode (1 request/analysis)",
        "app_info": "ℹ️ Application Info",
        "model_status": "🤖 Model Status",
        "model_ready": "🟢 Model Ready",
        "model_error": "🔴 Model Error",
        "analysis_history": "📊 Analysis History",
        "total": "Total",
        "successful": "Successful",
        "clear_history": "🗑️ Clear History",
        "system_status": "💻 System Status",
        "disclaimer": (
            "⚠️ **MEDICAL DISCLAIMER**\n\nThis tool is for **educational purposes "
            "only**. All analyses must be reviewed by qualified healthcare "
            "professionals. **Do not make medical decisions** based solely on this "
            "analysis."
        ),
        # Main content
        "config_required": "⚠️ **Configuration Required**",
        "upload_title": "📤 Upload Medical Image",
        "upload_choose": "Choose a medical image file",
        "upload_placeholder": "👆 **Upload a medical image to begin analysis**",
        "uploaded_image": "📷 Uploaded Image",
        "analysis_results": "📋 Analysis Results",
        "analyze_button": "🔍 Analyze Image",
        "processing_image": "🔄 Processing image...",
        "image_details": "📊 Image Details",
        "filename": "Filename",
        "size": "Size",
        "dimensions": "Dimensions",
        "format": "Format",
        "click_to_start": "👈 Click 'Analyze Image' to start medical analysis",
        # Progress
        "init_agent": "🤖 Initializing AI agent...",
        "prep_image": "🖼️ Preparing image for analysis...",
        "analyzing": "🔍 Analyzing medical image... This may take up to 2 minutes...",
        "processing_results": "📋 Processing results...",
        "cleaning_up": "🧹 Cleaning up...",
        "analysis_success": "✅ Analysis completed successfully in {t}s",
        "analysis_failed": "❌ Analysis failed: {e}",
        # Results
        "analysis_complete": "✅ Analysis Complete",
        "analysis_time": "Analysis Time",
        "model": "Model",
        "timestamp": "Timestamp",
        "tokens": "Tokens",
        "report_title": "### 📋 Medical Analysis Report",
        "download_report": "📥 Download Report",
        "translate_to_fr": "🌐 Traduire en Français",
        "translate_to_en": "🌐 Translate to English",
        "translating": "🌐 Translating report...",
        "translate_success": "✅ Report translated",
        "new_analysis": "🔄 New Analysis",
        "retry_analysis": "🔄 Retry Analysis",
        "analysis_failed_header": "❌ Analysis Failed",
    },
    "fr": {
        # Header / footer
        "app_title": "🏥 Agent de Diagnostic par Imagerie Médicale",
        "app_subtitle": "### Analyse d'images médicales avancée par IA",
        "footer": "À but éducatif uniquement | Consultez toujours des professionnels de santé",
        # Sidebar sections
        "configuration": "⚙️ Configuration",
        "language": "🌐 Langue",
        "navigation": "🧭 Navigation",
        "home": "🏠 Accueil",
        "about": "👨‍💻 À propos",
        "api_configuration": "🔑 Configuration de l'API",
        "api_key_label": "Clé API Google :",
        "api_key_placeholder": "Saisissez votre clé API ici...",
        "save_key": "💾 Enregistrer la clé",
        "get_api_key": "🔗 Obtenir une clé API",
        "api_key_saved": "✅ Clé API enregistrée !",
        "api_key_configured": "✅ Clé API configurée",
        "reset_api_key": "🔄 Réinitialiser la clé",
        "analysis_options": "🔎 Options d'analyse",
        "web_search_label": "Recherche web en direct (références)",
        "web_search_help": (
            "ACTIVÉ : l'agent recherche des références médicales en direct sur le web "
            "(rapport plus riche, plusieurs requêtes API). DÉSACTIVÉ : une requête par "
            "analyse — idéal pour préserver le quota journalier gratuit."
        ),
        "web_search_on": "🌐 Références en direct ACTIVÉES",
        "web_search_off": "⚡ Mode économie de quota (1 requête/analyse)",
        "app_info": "ℹ️ Informations sur l'application",
        "model_status": "🤖 État du modèle",
        "model_ready": "🟢 Modèle prêt",
        "model_error": "🔴 Erreur du modèle",
        "analysis_history": "📊 Historique des analyses",
        "total": "Total",
        "successful": "Réussies",
        "clear_history": "🗑️ Effacer l'historique",
        "system_status": "💻 État du système",
        "disclaimer": (
            "⚠️ **AVERTISSEMENT MÉDICAL**\n\nCet outil est à but **éducatif "
            "uniquement**. Toutes les analyses doivent être revues par des "
            "professionnels de santé qualifiés. **Ne prenez aucune décision médicale** "
            "sur la seule base de cette analyse."
        ),
        # Main content
        "config_required": "⚠️ **Configuration requise**",
        "upload_title": "📤 Téléverser une image médicale",
        "upload_choose": "Choisissez un fichier image médicale",
        "upload_placeholder": "👆 **Téléversez une image médicale pour lancer l'analyse**",
        "uploaded_image": "📷 Image téléversée",
        "analysis_results": "📋 Résultats de l'analyse",
        "analyze_button": "🔍 Analyser l'image",
        "processing_image": "🔄 Traitement de l'image...",
        "image_details": "📊 Détails de l'image",
        "filename": "Nom du fichier",
        "size": "Taille",
        "dimensions": "Dimensions",
        "format": "Format",
        "click_to_start": "👈 Cliquez sur « Analyser l'image » pour lancer l'analyse",
        # Progress
        "init_agent": "🤖 Initialisation de l'agent IA...",
        "prep_image": "🖼️ Préparation de l'image pour l'analyse...",
        "analyzing": "🔍 Analyse de l'image médicale... Cela peut prendre jusqu'à 2 minutes...",
        "processing_results": "📋 Traitement des résultats...",
        "cleaning_up": "🧹 Nettoyage...",
        "analysis_success": "✅ Analyse terminée avec succès en {t}s",
        "analysis_failed": "❌ Échec de l'analyse : {e}",
        # Results
        "analysis_complete": "✅ Analyse terminée",
        "analysis_time": "Durée d'analyse",
        "model": "Modèle",
        "timestamp": "Horodatage",
        "tokens": "Jetons",
        "report_title": "### 📋 Rapport d'analyse médicale",
        "download_report": "📥 Télécharger le rapport",
        "translate_to_fr": "🌐 Traduire en Français",
        "translate_to_en": "🌐 Traduire en Anglais",
        "translating": "🌐 Traduction du rapport...",
        "translate_success": "✅ Rapport traduit",
        "new_analysis": "🔄 Nouvelle analyse",
        "retry_analysis": "🔄 Relancer l'analyse",
        "analysis_failed_header": "❌ Échec de l'analyse",
    },
}


def t(key: str, lang: str = "en") -> str:
    """Translate `key` into `lang`, falling back to English then the key itself."""
    lang_map = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return lang_map.get(key) or TRANSLATIONS["en"].get(key, key)
