FROM python:3.11-slim

RUN useradd -m vscode
USER vscode
WORKDIR /workspace

RUN pip install --user ruff black pytest
CMD ["bash"]
