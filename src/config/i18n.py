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
        "about": "ℹ️ About",
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

        # About page
        "about_title": "ℹ️ About this platform",
        "about_intro": (
            "Medical Imaging Diagnosis Agent is an AI assistant that reads a medical image and "
            "drafts a structured report. It is an educational demo of an end-to-end generative-AI "
            "product, not a medical device."
        ),
        "about_what_h": "What it does",
        "about_what": (
            "- Reads X-ray, MRI, CT and ultrasound images (JPG or PNG, up to 5 MB)\n"
            "- Assesses image quality and lists the findings systematically\n"
            "- Proposes a most-likely assessment with a confidence level, plus alternatives\n"
            "- Explains the result in plain language\n"
            "- Can look up live references on the web (optional toggle)\n"
            "- Writes in English or French, and translates an existing report in one click"
        ),
        "about_how_h": "How it works",
        "about_step1_h": "Upload",
        "about_step1_b": "Drop in a medical image. No software to install, just your browser.",
        "about_step2_h": "Examine",
        "about_step2_b": "The agent looks at the image the way a first reader would.",
        "about_step3_h": "Report",
        "about_step3_b": "It writes an organised report: findings, assessment, confidence level.",
        "about_step4_h": "Explain",
        "about_step4_b": "It restates the result in everyday words and what to double-check.",
        "about_tech_h": "Under the hood",
        "about_tech": (
            "- Google Gemini vision model, driven by the Agno agent framework\n"
            "- Retries on transient errors and falls back to a second model when the first is overloaded\n"
            "- Your image is sent to Google's Gemini API for analysis; temporary files are deleted "
            "right after and nothing is stored on the server\n"
            "- Your API key is only kept for your session\n"
            "- Built with Streamlit and shipped as a multi-architecture Docker image"
        ),
        "about_limits_h": "Good to know",
        "about_limits": (
            "- For educational use only: its output can be wrong\n"
            "- Every result must be reviewed by a qualified healthcare professional\n"
            "- You need your own Google AI Studio API key; the free quota is small and varies by model\n"
            "- On a free hosting plan the app may sleep when idle, so the first load can take about a minute"
        ),
        "about_links_h": "Open source",
        "about_repo": "💻 Source code",
        "about_issues": "🐞 Report an issue",
        "about_image": "🐳 Docker image",
        # Sidebar, setup and help blocks
        "setup_steps": "To get started:\n1. 🔑 Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)\n2. 📝 Enter your API key in the sidebar\n3. 🚀 Start analyzing medical images!",
        "setup_free_tier": "**Free tier**: the daily quota is small and varies by model. Check your usage in Google AI Studio.",
        "setup_what_h": "🎯 What This Tool Can Do",
        "setup_types": "**Image types supported:**\n- 🦴 X-rays (chest, skeletal, dental)\n- 🧠 MRI scans (brain, spine, joints)\n- 🫁 CT scans (thoracic, abdominal)\n- ❤️ Ultrasound (cardiac, obstetric)",
        "setup_features": "**Analysis features:**\n- 📋 Structured diagnostic reports\n- 🔍 Detailed findings identification\n- 📚 Medical literature integration\n- 👥 Patient-friendly explanations",
        "info_features_h": "📋 Features",
        "info_features": "**Core capabilities:**\n- 🔍 Multi-modal medical image analysis\n- 📊 Structured diagnostic reporting\n- 🔬 Research literature integration\n- 👥 Patient-friendly explanations\n- 📥 Report export",
        "info_specs_h": "🛠️ Technical Specs",
        "info_specs": "- **Model**: {model}\n- **Max image size**: {size}MB\n- **Supported formats**: {formats}\n- **Analysis timeout**: {timeout}s",
        "tools_label": "Tools",
        "label_dependencies": "Dependencies",
        "label_session": "Session",
        "status_running": "🟢 Running",
        "status_loaded": "🟢 Loaded",
        "status_missing": "🔴 Missing",
        "status_active": "🟢 Active",
        "status_error": "🔴 Error",
        "tips_h": "💡 Tips for Best Results",
        "tips": "**Image quality guidelines:**\n- ✅ High-resolution images (at least 200x200 pixels)\n- ✅ Clear, well-contrasted medical images\n- ✅ Proper anatomical positioning\n- ✅ Standard imaging formats\n\n**Supported image types:**\n- JPEG/JPG files\n- PNG files",
        "expect_h": "🔍 What to Expect",
        "expect": "**The analysis will include:**\n- 🔍 Image type and quality assessment\n- 📋 Systematic medical findings\n- 🎯 Diagnostic assessment with confidence levels\n- 👥 Patient-friendly explanations\n- 📚 Research references and literature\n- ⏱️ Typical analysis time: 30-120 seconds",
        "trouble_h": "🔧 Troubleshooting",
        "trouble": "**Common solutions:**\n- 🔄 Try refreshing the page\n- 🔑 Verify your API key is correct\n- 🖼️ Try a different image format\n- 🌐 Check your internet connection\n- 📏 Make sure the image is under 5MB",
        "wait_processing": "Please wait for image processing to complete",
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
        "about": "ℹ️ À propos",
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

        # About page
        "about_title": "ℹ️ À propos de la plateforme",
        "about_intro": (
            "Medical Imaging Diagnosis Agent est un assistant IA qui lit une image médicale et "
            "rédige un premier jet de compte rendu structuré. C'est une démo éducative d'un produit "
            "d'IA générative de bout en bout, pas un dispositif médical."
        ),
        "about_what_h": "Ce qu'elle fait",
        "about_what": (
            "- Lit des images de radiographie, IRM, scanner et échographie (JPG ou PNG, jusqu'à 5 Mo)\n"
            "- Évalue la qualité de l'image et liste les observations de façon systématique\n"
            "- Propose une hypothèse la plus probable avec un niveau de confiance, et des alternatives\n"
            "- Explique le résultat en langage clair\n"
            "- Peut chercher des références en direct sur le web (option à activer)\n"
            "- Rédige en anglais ou en français, et traduit un rapport existant en un clic"
        ),
        "about_how_h": "Comment ça marche",
        "about_step1_h": "Import",
        "about_step1_b": "Déposez une image médicale. Rien à installer, juste votre navigateur.",
        "about_step2_h": "Examen",
        "about_step2_b": "L'agent regarde l'image comme le ferait un premier lecteur.",
        "about_step3_h": "Rapport",
        "about_step3_b": "Il rédige un rapport organisé : observations, hypothèse, niveau de confiance.",
        "about_step4_h": "Explication",
        "about_step4_b": "Il reformule le résultat en mots simples et ce qu'il faut vérifier.",
        "about_tech_h": "Sous le capot",
        "about_tech": (
            "- Modèle de vision Google Gemini, piloté par le framework d'agents Agno\n"
            "- Nouvelles tentatives sur erreur transitoire, et bascule sur un second modèle si le premier est saturé\n"
            "- Votre image est envoyée à l'API Gemini de Google pour l'analyse ; les fichiers temporaires "
            "sont supprimés juste après et rien n'est conservé sur le serveur\n"
            "- Votre clé API n'est gardée que le temps de votre session\n"
            "- Construite avec Streamlit et livrée en image Docker multi-architecture"
        ),
        "about_limits_h": "Bon à savoir",
        "about_limits": (
            "- À usage éducatif uniquement : le résultat peut être faux\n"
            "- Tout résultat doit être revu par un professionnel de santé qualifié\n"
            "- Il faut votre propre clé API Google AI Studio ; le quota gratuit est faible et varie selon le modèle\n"
            "- Sur un hébergement gratuit, l'application peut s'endormir : le premier chargement peut prendre environ une minute"
        ),
        "about_links_h": "Open source",
        "about_repo": "💻 Code source",
        "about_issues": "🐞 Signaler un problème",
        "about_image": "🐳 Image Docker",
        # Sidebar, setup and help blocks
        "setup_steps": "Pour commencer :\n1. 🔑 Obtenez une clé API gratuite sur [Google AI Studio](https://aistudio.google.com/apikey)\n2. 📝 Saisissez votre clé API dans la barre latérale\n3. 🚀 Lancez l'analyse d'images médicales !",
        "setup_free_tier": "**Offre gratuite** : le quota quotidien est faible et varie selon le modèle. Consultez votre consommation dans Google AI Studio.",
        "setup_what_h": "🎯 Ce que fait cet outil",
        "setup_types": "**Types d'images pris en charge :**\n- 🦴 Radiographies (thorax, squelette, dentaire)\n- 🧠 IRM (cerveau, rachis, articulations)\n- 🫁 Scanners (thoracique, abdominal)\n- ❤️ Échographie (cardiaque, obstétricale)",
        "setup_features": "**Fonctions d'analyse :**\n- 📋 Rapports diagnostiques structurés\n- 🔍 Identification détaillée des observations\n- 📚 Références de la littérature médicale\n- 👥 Explications accessibles aux patients",
        "info_features_h": "📋 Fonctionnalités",
        "info_features": "**Capacités principales :**\n- 🔍 Analyse d'images médicales multimodale\n- 📊 Rapports diagnostiques structurés\n- 🔬 Intégration de la littérature scientifique\n- 👥 Explications accessibles aux patients\n- 📥 Export du rapport",
        "info_specs_h": "🛠️ Caractéristiques techniques",
        "info_specs": "- **Modèle** : {model}\n- **Taille max. de l'image** : {size} Mo\n- **Formats pris en charge** : {formats}\n- **Délai d'analyse** : {timeout} s",
        "tools_label": "Outils",
        "label_dependencies": "Dépendances",
        "label_session": "Session",
        "status_running": "🟢 En cours",
        "status_loaded": "🟢 Chargées",
        "status_missing": "🔴 Manquantes",
        "status_active": "🟢 Active",
        "status_error": "🔴 Erreur",
        "tips_h": "💡 Conseils pour de meilleurs résultats",
        "tips": "**Qualité de l'image :**\n- ✅ Images en haute résolution (au moins 200x200 pixels)\n- ✅ Images médicales nettes et bien contrastées\n- ✅ Positionnement anatomique correct\n- ✅ Formats d'imagerie standards\n\n**Types d'images pris en charge :**\n- Fichiers JPEG/JPG\n- Fichiers PNG",
        "expect_h": "🔍 À quoi s'attendre",
        "expect": "**L'analyse comprendra :**\n- 🔍 Type d'image et évaluation de sa qualité\n- 📋 Observations médicales systématiques\n- 🎯 Évaluation diagnostique avec niveaux de confiance\n- 👥 Explications accessibles aux patients\n- 📚 Références de la littérature\n- ⏱️ Durée habituelle de l'analyse : 30 à 120 secondes",
        "trouble_h": "🔧 Dépannage",
        "trouble": "**Solutions courantes :**\n- 🔄 Rechargez la page\n- 🔑 Vérifiez que votre clé API est correcte\n- 🖼️ Essayez un autre format d'image\n- 🌐 Vérifiez votre connexion internet\n- 📏 Assurez-vous que l'image fait moins de 5 Mo",
        "wait_processing": "Veuillez attendre la fin du traitement de l'image",
    },
}


def t(key: str, lang: str = "en") -> str:
    """Translate `key` into `lang`, falling back to English then the key itself."""
    lang_map = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return lang_map.get(key) or TRANSLATIONS["en"].get(key, key)
