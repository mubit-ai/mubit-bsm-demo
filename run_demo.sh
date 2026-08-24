#!/usr/bin/env bash
# Launch the BSM sample demo on port 7873. Loads .env if present.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${MUBIT_ENDPOINT:?set MUBIT_ENDPOINT}"
: "${MUBIT_API_KEY:?set MUBIT_API_KEY}"
if [[ "${DEMO_MODEL:-gemini-3.7-flash}" == gpt-* ]]; then
  : "${OPENAI_API_KEY:?set OPENAI_API_KEY for gpt models}"
else
  : "${GEMINI_API_KEY:?set GEMINI_API_KEY}"
fi
PY=".venv/bin/python"; [ -x "$PY" ] || PY=python3
exec "$PY" -m uvicorn demo.server:app --host 127.0.0.1 --port "${DEMO_PORT:-7873}"
