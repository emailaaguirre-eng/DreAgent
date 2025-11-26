@echo off
ruff check .
black --check .
pytest -q
