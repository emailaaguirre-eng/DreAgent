@echo off
echo Starting FastAPI service...
start uvicorn src.agent_service:app --reload --host 127.0.0.1 --port 8000
echo Watch extension: cd extension ^&^& npm run watch
