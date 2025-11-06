#!/bin/bash
set -e

echo "Starting FastAPI service..."
uvicorn src.agent_service:app --reload --host 127.0.0.1 --port 8000 &

echo "Watch extension (run 'npm run watch' in extension/ separately)"
wait
