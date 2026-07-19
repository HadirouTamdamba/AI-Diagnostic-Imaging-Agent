# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.x     | ✅        |

## Reporting a Vulnerability

Please **do not open a public GitHub issue** for security vulnerabilities.

Instead, report them privately:

- Email: **hadirou.tamdamba@outlook.fr** (subject: `[SECURITY] AI-Diagnostic-Imaging-Agent`)
- Or use GitHub's [private vulnerability reporting](https://github.com/HadirouTamdamba/AI-Diagnostic-Imaging-Agent/security/advisories/new)

You can expect an acknowledgement within 72 hours.

## Scope & Data Handling

- Images are processed in memory and in short-lived temporary files that are deleted after each analysis; no medical data is persisted server-side.
- API keys are kept in the Streamlit session state or the `.env` file (never committed — see `.gitignore`) and are never logged.
- Dependencies are audited weekly via Dependabot and `pip-audit` in CI.

## Disclaimer

This application is for **educational purposes only** and must not be used as a
standalone medical device. Analyses require review by qualified healthcare
professionals.
