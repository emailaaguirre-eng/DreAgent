#!/bin/bash
# =============================================================================
# HUMMINGBIRD-LEA - Deployment Script
# Powered by CoDre-X | B & D Servicing LLC
# =============================================================================
# This script handles deployment to a production server
#
# Usage:
#   ./scripts/deploy.sh [--docker|--native]
#
# Options:
#   --docker   Deploy using Docker Compose (recommended)
#   --native   Deploy natively with systemd
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="hummingbird-lea"
APP_USER="hummingbird"
APP_GROUP="hummingbird"
APP_DIR="/opt/hummingbird-lea"
DATA_DIR="/var/lib/hummingbird"
LOG_DIR="/var/log/hummingbird"
VENV_DIR="$APP_DIR/venv"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# =============================================================================
# System Setup
# =============================================================================

setup_user() {
    log_info "Setting up user and group..."

    if ! getent group $APP_GROUP > /dev/null 2>&1; then
        groupadd -r $APP_GROUP
    fi

    if ! id -u $APP_USER > /dev/null 2>&1; then
        useradd -r -g $APP_GROUP -d $APP_DIR -s /sbin/nologin $APP_USER
    fi

    log_success "User $APP_USER created"
}

setup_directories() {
    log_info "Creating directories..."

    mkdir -p $APP_DIR
    mkdir -p $DATA_DIR/{uploads,knowledge,memory,templates,logs}
    mkdir -p $LOG_DIR
    mkdir -p /etc/hummingbird

    chown -R $APP_USER:$APP_GROUP $APP_DIR
    chown -R $APP_USER:$APP_GROUP $DATA_DIR
    chown -R $APP_USER:$APP_GROUP $LOG_DIR

    log_success "Directories created"
}

# =============================================================================
# Docker Deployment
# =============================================================================

deploy_docker() {
    log_info "Deploying with Docker..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    # Copy files
    cp -r . $APP_DIR/
    cd $APP_DIR

    # Create .env if not exists
    if [[ ! -f .env ]]; then
        log_warning "No .env file found. Copying from .env.production..."
        cp config/.env.production .env
        log_warning "Please edit .env with your production values!"
    fi

    # Pull and build
    log_info "Pulling images..."
    docker-compose pull

    log_info "Building application..."
    docker-compose build

    # Start services
    log_info "Starting services..."
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

    # Initialize models
    log_info "Initializing Ollama models (this may take a while)..."
    docker-compose --profile init up ollama-init

    log_success "Docker deployment complete!"
    log_info "Check status with: docker-compose ps"
}

# =============================================================================
# Native Deployment
# =============================================================================

deploy_native() {
    log_info "Deploying natively..."

    # Check Python
    if ! command -v python3.11 &> /dev/null; then
        log_warning "Python 3.11 not found, trying python3..."
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python3.11"
    fi

    # Setup user and directories
    setup_user
    setup_directories

    # Copy application
    log_info "Copying application files..."
    cp -r . $APP_DIR/
    cd $APP_DIR

    # Create virtual environment
    log_info "Creating virtual environment..."
    $PYTHON_CMD -m venv $VENV_DIR
    source $VENV_DIR/bin/activate

    # Install dependencies
    log_info "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install gunicorn

    # Create .env if not exists
    if [[ ! -f .env ]]; then
        log_warning "No .env file found. Copying from .env.production..."
        cp config/.env.production .env
        chmod 600 .env
        log_warning "Please edit .env with your production values!"
    fi

    # Install systemd service
    log_info "Installing systemd service..."
    cp scripts/hummingbird.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable hummingbird

    # Set permissions
    chown -R $APP_USER:$APP_GROUP $APP_DIR
    chown -R $APP_USER:$APP_GROUP $DATA_DIR

    # Start service
    log_info "Starting service..."
    systemctl start hummingbird

    log_success "Native deployment complete!"
    log_info "Check status with: systemctl status hummingbird"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo "=============================================="
    echo "  HUMMINGBIRD-LEA Deployment Script"
    echo "  Powered by CoDre-X"
    echo "=============================================="
    echo ""

    check_root

    case "${1:-}" in
        --docker)
            deploy_docker
            ;;
        --native)
            deploy_native
            ;;
        *)
            echo "Usage: $0 [--docker|--native]"
            echo ""
            echo "Options:"
            echo "  --docker   Deploy using Docker Compose (recommended)"
            echo "  --native   Deploy natively with systemd"
            exit 1
            ;;
    esac

    echo ""
    log_success "Deployment complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Edit /opt/hummingbird-lea/.env with production values"
    echo "  2. Configure Nginx (see config/nginx/)"
    echo "  3. Set up SSL certificates"
    echo "  4. Configure firewall rules"
    echo ""
}

main "$@"
