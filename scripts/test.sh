#!/bin/bash
set -e

echo "Running linters..."
ruff check .
black --check .
pytest -q
