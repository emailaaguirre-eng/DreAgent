"""
=============================================================================
HUMMINGBIRD-LEA - Health API
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Health check, status, and monitoring endpoints.
=============================================================================
"""

from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel

from core.utils.config import get_settings
from core.providers.ollama import get_ollama_client
from core.utils.monitoring import (
    get_health_checker,
    get_metrics_collector,
    get_system_metrics,
    HealthStatus as HealthStatusEnum,
)
from core.utils.security.cache import get_cache_stats

router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================

class HealthStatus(BaseModel):
    """Health status response"""
    status: str
    timestamp: datetime
    version: str
    ollama_available: bool
    ollama_models: Optional[list] = None


class OllamaStatus(BaseModel):
    """Ollama-specific status"""
    available: bool
    host: str
    models: list


class DetailedHealthResponse(BaseModel):
    """Detailed health report"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    components: list
    system_metrics: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Application metrics response"""
    requests: Dict[str, Any]
    cache: Dict[str, Any]
    system: Dict[str, Any]


# =============================================================================
# Health Check Functions
# =============================================================================

async def check_ollama() -> Dict[str, Any]:
    """Check Ollama service health"""
    ollama = get_ollama_client()
    try:
        available = await ollama.is_available()
        if available:
            models = await ollama.list_models()
            return {
                "status": "healthy",
                "message": f"{len(models)} models available",
                "details": {"models": models},
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Ollama not responding",
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": str(e),
        }


async def check_database() -> Dict[str, Any]:
    """Check database health"""
    settings = get_settings()
    try:
        # Simple check - database file exists or connection works
        from pathlib import Path
        if "sqlite" in settings.database_url:
            # Extract path from sqlite URL
            db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")
            if db_path.startswith("./"):
                db_path = db_path[2:]
            if Path(db_path).exists() or Path(db_path).parent.exists():
                return {"status": "healthy", "message": "Database accessible"}
        return {"status": "healthy", "message": "Database configured"}
    except Exception as e:
        return {"status": "degraded", "message": str(e)}


async def check_storage() -> Dict[str, Any]:
    """Check storage directories"""
    settings = get_settings()
    try:
        upload_ok = settings.upload_path.exists()
        knowledge_ok = settings.knowledge_path.exists()

        if upload_ok and knowledge_ok:
            return {"status": "healthy", "message": "Storage directories OK"}
        elif upload_ok or knowledge_ok:
            return {"status": "degraded", "message": "Some storage directories missing"}
        else:
            return {"status": "unhealthy", "message": "Storage directories not accessible"}
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


# =============================================================================
# Register Health Checks
# =============================================================================

def _setup_health_checks():
    """Register all health checks"""
    checker = get_health_checker()
    checker.register_check("ollama", check_ollama)
    checker.register_check("database", check_database)
    checker.register_check("storage", check_storage)


# Initialize on import
_setup_health_checks()


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/", response_model=HealthStatus)
async def health_check():
    """
    Check the health of the application.

    Returns:
    - Application status
    - Ollama availability
    - Available models
    """
    settings = get_settings()
    ollama = get_ollama_client()

    # Check Ollama
    ollama_available = await ollama.is_available()
    models = []

    if ollama_available:
        models = await ollama.list_models()

    return HealthStatus(
        status="healthy" if ollama_available else "degraded",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        ollama_available=ollama_available,
        ollama_models=models if models else None,
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health():
    """
    Get detailed health report with all components.

    Returns comprehensive health information including:
    - Individual component status
    - System metrics (CPU, memory, disk)
    - Application uptime
    """
    checker = get_health_checker()
    report = await checker.run_checks()

    return DetailedHealthResponse(
        status=report.status.value,
        timestamp=report.timestamp,
        version=report.version,
        uptime_seconds=report.uptime_seconds,
        components=[c.to_dict() for c in report.components],
        system_metrics=report.system_metrics,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get application metrics.

    Returns:
    - Request statistics (count, latency, error rate)
    - Cache statistics (hits, misses, hit rate)
    - System metrics (CPU, memory, disk)
    """
    metrics_collector = get_metrics_collector()

    return MetricsResponse(
        requests=metrics_collector.get_summary(),
        cache=get_cache_stats(),
        system=get_system_metrics(),
    )


@router.get("/ollama", response_model=OllamaStatus)
async def ollama_status():
    """
    Check Ollama status specifically.

    Returns detailed information about the Ollama connection.
    """
    settings = get_settings()
    ollama = get_ollama_client()

    available = await ollama.is_available()
    models = []

    if available:
        models = await ollama.list_models()

    return OllamaStatus(
        available=available,
        host=settings.ollama_host,
        models=models,
    )


@router.get("/ping")
async def ping():
    """Simple ping endpoint for basic connectivity check"""
    return {"ping": "pong", "timestamp": datetime.utcnow()}


@router.get("/ready")
async def readiness():
    """
    Kubernetes-style readiness probe.

    Returns 200 if application is ready to receive traffic.
    """
    ollama = get_ollama_client()
    available = await ollama.is_available()

    if available:
        return {"ready": True, "timestamp": datetime.utcnow()}
    else:
        # Return 200 but indicate degraded state
        return {
            "ready": True,
            "degraded": True,
            "message": "Ollama unavailable",
            "timestamp": datetime.utcnow(),
        }


@router.get("/live")
async def liveness():
    """
    Kubernetes-style liveness probe.

    Returns 200 if application is alive.
    """
    return {"alive": True, "timestamp": datetime.utcnow()}
