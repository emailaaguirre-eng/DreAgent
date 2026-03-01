from datetime import datetime
"""
=============================================================================
HUMMINGBIRD-LEA - Main Application
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
FastAPI application entry point.
=============================================================================
"""

import logging
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from core.utils.config import get_settings
from core.providers.ollama import get_ollama_client
from core.utils.monitoring import (
    setup_logging,
    get_metrics_collector,
    RequestMetrics,
)
from core.utils.security import (
    setup_exception_handlers,
    RateLimitMiddleware,
    RateLimitConfig,
    SecurityHeadersMiddleware,
    RequestIDMiddleware,
    AuthRequiredMiddleware,
)

# Import API routers
from .api.chat import router as chat_router
from .api.auth import router as auth_router
from .api.health import router as health_router
from .api.knowledge import router as knowledge_router  # Phase 3: RAG
from .api.vision import router as vision_router  # Phase 4: Vision/OCR
from .api.documents import router as documents_router  # Phase 5: Document Generation
from .api.ide import router as ide_router  # Phase 7: Chiquis IDE
from .api.microsoft import router as microsoft_router

settings = get_settings()

# Setup logging
setup_logging(
    log_level=settings.log_level,
    log_file=settings.log_file if settings.is_production else None,
    json_format=settings.is_production,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan Events
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    Runs on startup and shutdown.
    """
    # Startup
    logger.info("Starting Hummingbird-LEA...")

    # Check Ollama connection
    ollama = get_ollama_client()
    if await ollama.is_available():
        models = await ollama.list_models()
        logger.info(f"Ollama connected. Available models: {models}")
    else:
        logger.warning("Ollama not available. AI features will be limited.")

    # Ensure data directories exist
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.knowledge_path.mkdir(parents=True, exist_ok=True)
    settings.memory_path.mkdir(parents=True, exist_ok=True)

    logger.info("Hummingbird-LEA is ready!")

    yield

    # Shutdown
    logger.info("Shutting down Hummingbird-LEA...")


# =============================================================================
# Create Application
# =============================================================================

app = FastAPI(
    title="Hummingbird-LEA",
    description="Your AI Team - Powered by CoDre-X",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,  # Hide docs in production
    redoc_url="/redoc" if settings.debug else None,
)


# =============================================================================
# Exception Handlers (Phase 6)
# =============================================================================

setup_exception_handlers(app, debug=settings.debug)


# =============================================================================
# Middleware (Order matters - first added = last executed)
# =============================================================================

# CORS - Allow requests from configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers (Phase 6)
app.add_middleware(SecurityHeadersMiddleware)

# Require authentication for all /api/* routes except allowlisted ones
app.add_middleware(AuthRequiredMiddleware)

# Rate limiting (Phase 6) - only in production
if settings.is_production:
    rate_limit_config = RateLimitConfig(
        requests_per_minute=settings.rate_limit_per_minute,
        requests_per_hour=settings.rate_limit_per_hour,
        burst_limit=settings.rate_limit_burst,
        block_duration_minutes=settings.rate_limit_block_minutes,
    )
    app.add_middleware(RateLimitMiddleware, config=rate_limit_config)

# Request ID middleware (Phase 6)
app.add_middleware(RequestIDMiddleware)


# =============================================================================
# Metrics Middleware
# =============================================================================

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect request metrics"""
    start_time = time.time()

    # Get or generate request ID
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    response = await call_next(request)

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # Record metrics
    metrics_collector = get_metrics_collector()
    metrics_collector.record(RequestMetrics(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round(latency_ms, 2),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        client_ip=request.client.host if request.client else None,
    ))

    # Add timing header
    response.headers["X-Response-Time"] = f"{latency_ms:.2f}ms"

    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to prevent indexing"""
    response = await call_next(request)

    # Block search engine indexing
    response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"

    return response


# =============================================================================
# Routers
# =============================================================================

app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(health_router, prefix="/api/health", tags=["Health"])
app.include_router(knowledge_router, prefix="/api/knowledge", tags=["Knowledge Base"])  # Phase 3: RAG
app.include_router(vision_router, prefix="/api/vision", tags=["Vision & OCR"])  # Phase 4: Vision/OCR
app.include_router(documents_router, prefix="/api/documents", tags=["Documents"])  # Phase 5: Document Generation
app.include_router(ide_router, prefix="/api/ide", tags=["Chiquis IDE"])  # Phase 7: IDE
app.include_router(microsoft_router, prefix="/api/microsoft", tags=["Microsoft"])
app.include_router(microsoft_router, prefix="/auth/microsoft", tags=["Microsoft (Auth)"])


# =============================================================================
# Static Files & Frontend
# =============================================================================

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Serve robots.txt
@app.get("/robots.txt", include_in_schema=False)
async def robots():
    """Serve robots.txt to block crawlers"""
    robots_path = Path(__file__).parent.parent.parent / "config" / "robots.txt"
    if robots_path.exists():
        return FileResponse(robots_path)
    return HTMLResponse("User-agent: *\nDisallow: /")


# Serve frontend

@app.get("/api/version", include_in_schema=False)
async def api_version():
    """Return live build/version information for deploy verification."""
    repo_root = Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()
        except Exception:
            return "unknown"

    return {
        "app": "Hummingbird-LEA",
        "commitSha": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "buildTimestamp": _git("show", "-s", "--format=%cI", "HEAD"),
        "now": datetime.utcnow().isoformat() + "Z",
    }

@app.get("/", include_in_schema=False)
async def root():
    """Serve the main frontend page"""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("""
    <html>
        <head><title>Hummingbird-LEA</title></head>
        <body>
            <h1>Hummingbird-LEA</h1>
            <p>Frontend not found. Please check the static directory.</p>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """)


# Serve Chiquis IDE (Phase 7)
@app.get("/ide", include_in_schema=False)
async def ide():
    """Serve the Chiquis IDE page"""
    ide_path = Path(__file__).parent / "static" / "ide.html"
    if ide_path.exists():
        return FileResponse(ide_path)
    return HTMLResponse("""
    <html>
        <head><title>Chiquis IDE</title></head>
        <body>
            <h1>Chiquis IDE</h1>
            <p>IDE not found. Please check the static directory.</p>
            <p><a href="/">Back to Chat</a></p>
        </body>
    </html>
    """)


# =============================================================================
# Run with Uvicorn
# =============================================================================

def run():
    """Run the application with Uvicorn"""
    import uvicorn
    uvicorn.run(
        "apps.hummingbird.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
