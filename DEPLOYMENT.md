# 🚀 Deployment Guide

This guide covers the two supported deployment targets: **Docker / docker-compose** (local, VPS, or cloud) and **Streamlit Community Cloud** (free public demo).

---

## 1. Docker / docker-compose

### Build & run

```bash
cp .env.example .env      # set GOOGLE_API_KEY (users can also enter it in the UI)
docker compose up --build -d
```

App: http://localhost:8501 — health endpoint: `http://localhost:8501/_stcore/health`

### Useful commands

```bash
docker compose logs -f            # follow logs
docker compose ps                 # status + healthcheck
docker compose down               # stop
docker compose up --build -d      # rebuild after code changes
```

### Image characteristics
- Multi-stage build on `python:3.12-slim` (small final image, no build tools shipped)
- Runs as non-root user `appuser`
- Built-in `HEALTHCHECK` on the Streamlit health endpoint
- Logs go to stdout (`docker logs`); file log at `/tmp/medical_agent.log`
- Memory limit of 1 GB set in `docker-compose.yml` (adjust to your host)

### Deploying on a VPS / cloud VM (AWS EC2, etc.)
1. Install Docker + the compose plugin on the server
2. Clone the repo, create `.env` with your `GOOGLE_API_KEY`
3. `docker compose up --build -d`
4. Put a reverse proxy (Nginx/Caddy) in front for TLS, forwarding to port 8501. For Streamlit WebSockets, enable `proxy_http_version 1.1` and the `Upgrade`/`Connection` headers.

---

## 2. Streamlit Community Cloud (free)

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. **New app** → select the repo, branch `main`, main file **`src/app.py`**
4. (Optional) In **Advanced settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   ```
   If omitted, users enter their own key in the sidebar.
5. Deploy — you get a public `https://<app>.streamlit.app` URL

Notes:
- Python version is read from the platform settings; select 3.12
- Dependencies are installed from `requirements.txt` automatically

---

## 3. Configuration reference

Environment variables (via `.env`, compose `environment`, or platform secrets):

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | – | Gemini API key; if unset, users enter it in the UI |
| `MODEL_ID` | `gemini-2.0-flash` | Gemini model used for analysis |
| `MAX_IMAGE_SIZE` | `5242880` | Max upload size in bytes (5 MB) |
| `MAX_ANALYSIS_TIME` | `120` | Analysis timeout (seconds) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LOG_FILE` | `medical_agent.log` | Optional log file path |

---

## 4. CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:
1. **Lint** — `ruff check src/ tests/`
2. **Tests** — `pytest` with coverage
3. **Docker** — image build + container smoke test against the health endpoint

To publish images automatically, add a push step to Docker Hub or GHCR with repository secrets.

---

## ⚠️ Reminder

This application is for **educational purposes only** and must not be used for clinical decision-making without review by qualified healthcare professionals.
