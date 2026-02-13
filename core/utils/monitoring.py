"""
=============================================================================
HUMMINGBIRD-LEA - Monitoring and Logging System
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Health monitoring, metrics collection, and structured logging.

Features:
- Application health checks
- System metrics (CPU, memory, disk)
- Request metrics and latency tracking
- Structured JSON logging
- Log rotation and archival
=============================================================================
"""

import logging
import logging.handlers
import json
import time
import asyncio
import platform
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Health Status
# =============================================================================

class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component"""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


@dataclass
class HealthReport:
    """Overall health report"""
    status: HealthStatus
    timestamp: str
    version: str
    uptime_seconds: float
    components: List[ComponentHealth]
    system_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "components": [c.to_dict() for c in self.components],
            "system_metrics": self.system_metrics,
        }


# =============================================================================
# System Metrics
# =============================================================================

def get_system_metrics() -> Dict[str, Any]:
    """Collect system metrics (CPU, memory, disk)"""
    metrics = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
    }

    try:
        import psutil

        # CPU
        metrics["cpu"] = {
            "percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
        }

        # Memory
        mem = psutil.virtual_memory()
        metrics["memory"] = {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent_used": mem.percent,
        }

        # Disk
        disk = psutil.disk_usage("/")
        metrics["disk"] = {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": round(disk.percent, 1),
        }

        # Process
        proc = psutil.Process()
        metrics["process"] = {
            "memory_mb": round(proc.memory_info().rss / (1024**2), 2),
            "threads": proc.num_threads(),
            "open_files": len(proc.open_files()),
        }

    except ImportError:
        metrics["warning"] = "psutil not installed - limited metrics available"
    except Exception as e:
        metrics["error"] = str(e)

    return metrics


# =============================================================================
# Health Checker
# =============================================================================

class HealthChecker:
    """
    Application health checker with component monitoring.
    """

    def __init__(self, app_version: str = "1.0.0"):
        self.app_version = app_version
        self.start_time = time.time()
        self._checks: Dict[str, callable] = {}

    def register_check(self, name: str, check_func: callable):
        """Register a health check function"""
        self._checks[name] = check_func

    @property
    def uptime(self) -> float:
        """Get application uptime in seconds"""
        return time.time() - self.start_time

    async def check_component(self, name: str, check_func: callable) -> ComponentHealth:
        """Run a single component health check"""
        start = time.time()
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            latency = (time.time() - start) * 1000

            if isinstance(result, dict):
                status = HealthStatus(result.get("status", "healthy"))
                message = result.get("message", "OK")
                details = result.get("details", {})
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "OK" if result else "Check failed"
                details = {}

            return ComponentHealth(
                name=name,
                status=status,
                message=message,
                latency_ms=round(latency, 2),
                details=details,
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=round(latency, 2),
            )

    async def run_checks(self) -> HealthReport:
        """Run all registered health checks"""
        components = []

        for name, check_func in self._checks.items():
            health = await self.check_component(name, check_func)
            components.append(health)

        # Determine overall status
        if any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return HealthReport(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            version=self.app_version,
            uptime_seconds=self.uptime,
            components=components,
            system_metrics=get_system_metrics(),
        )


# =============================================================================
# Request Metrics
# =============================================================================

@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    method: str
    path: str
    status_code: int
    latency_ms: float
    timestamp: str
    client_ip: Optional[str] = None
    user_id: Optional[str] = None
    error: Optional[str] = None


class MetricsCollector:
    """
    Collect and aggregate request metrics.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._requests: List[RequestMetrics] = []
        self._total_requests = 0
        self._total_errors = 0
        self._latency_sum = 0.0
        self._status_counts: Dict[int, int] = {}
        self._path_counts: Dict[str, int] = {}

    def record(self, metrics: RequestMetrics):
        """Record a request"""
        self._requests.append(metrics)
        if len(self._requests) > self.max_history:
            self._requests.pop(0)

        self._total_requests += 1
        self._latency_sum += metrics.latency_ms

        # Track status codes
        self._status_counts[metrics.status_code] = \
            self._status_counts.get(metrics.status_code, 0) + 1

        # Track paths
        self._path_counts[metrics.path] = \
            self._path_counts.get(metrics.path, 0) + 1

        # Track errors
        if metrics.status_code >= 500:
            self._total_errors += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self._requests:
            return {
                "total_requests": 0,
                "requests_per_minute": 0,
                "average_latency_ms": 0,
                "error_rate": 0,
            }

        # Calculate rates
        latencies = [r.latency_ms for r in self._requests]

        return {
            "total_requests": self._total_requests,
            "recent_requests": len(self._requests),
            "average_latency_ms": round(sum(latencies) / len(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "p95_latency_ms": round(self._percentile(latencies, 95), 2),
            "error_rate": round(self._total_errors / self._total_requests * 100, 2),
            "status_codes": dict(sorted(self._status_counts.items())),
            "top_paths": dict(sorted(
                self._path_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
        }

    @staticmethod
    def _percentile(data: List[float], p: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


# =============================================================================
# Structured Logging
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
):
    """
    Configure application logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        json_format: Use JSON formatting for structured logs
        max_bytes: Max size before rotation
        backup_count: Number of backup files to keep
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (with rotation)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.info(f"Logging configured: level={log_level}, file={log_file}")


# =============================================================================
# Global Instances
# =============================================================================

_health_checker: Optional[HealthChecker] = None
_metrics_collector: Optional[MetricsCollector] = None


def get_health_checker() -> HealthChecker:
    """Get or create health checker singleton"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def get_metrics_collector() -> MetricsCollector:
    """Get or create metrics collector singleton"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_monitoring() -> None:
    """Reset monitoring singletons (for testing)"""
    global _health_checker, _metrics_collector
    _health_checker = None
    _metrics_collector = None
