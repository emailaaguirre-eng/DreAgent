#!/bin/bash
# =============================================================================
# HUMMINGBIRD-LEA - Ollama Setup Script
# Powered by CoDre-X | B & D Servicing LLC
# =============================================================================
# This script installs Ollama and pulls required models
#
# Usage: ./scripts/setup-ollama.sh
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# =============================================================================
# Install Ollama
# =============================================================================

install_ollama() {
    if command -v ollama &> /dev/null; then
        log_info "Ollama is already installed"
        ollama --version
        return
    fi

    log_info "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh

    log_success "Ollama installed successfully"
}

# =============================================================================
# Pull Models
# =============================================================================

pull_models() {
    log_info "Pulling required models..."
    echo ""

    # Chat model
    log_info "Pulling llama3.1:8b (chat/reasoning)..."
    ollama pull llama3.1:8b

    # Coding model
    log_info "Pulling deepseek-coder:6.7b (coding)..."
    ollama pull deepseek-coder:6.7b

    # Vision model
    log_info "Pulling llava-llama3:8b (vision)..."
    ollama pull llava-llama3:8b

    # Embedding model
    log_info "Pulling nomic-embed-text (embeddings)..."
    ollama pull nomic-embed-text

    echo ""
    log_success "All models pulled successfully!"
}

# =============================================================================
# Verify Installation
# =============================================================================

verify_installation() {
    log_info "Verifying installation..."
    echo ""

    echo "Installed models:"
    ollama list

    echo ""
    log_info "Testing llama3.1:8b..."
    echo "Hello" | ollama run llama3.1:8b "Say 'Hummingbird is ready!' in exactly 4 words"

    echo ""
    log_success "Ollama setup complete!"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo "=============================================="
    echo "  HUMMINGBIRD-LEA Ollama Setup"
    echo "=============================================="
    echo ""

    install_ollama
    echo ""
    pull_models
    echo ""
    verify_installation

    echo ""
    echo "=============================================="
    echo "  Setup Complete!"
    echo "=============================================="
    echo ""
    echo "Models installed:"
    echo "  - llama3.1:8b        (Chat/Reasoning)"
    echo "  - deepseek-coder:6.7b (Coding)"
    echo "  - llava-llama3:8b    (Vision/OCR)"
    echo "  - nomic-embed-text   (Embeddings)"
    echo ""
    echo "Start Ollama service:"
    echo "  sudo systemctl enable ollama"
    echo "  sudo systemctl start ollama"
    echo ""
}

main "$@"
