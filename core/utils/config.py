"""
=============================================================================
HUMMINGBIRD-LEA - Configuration Management
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Centralized configuration using Pydantic Settings.
Loads from environment variables and .env file.
=============================================================================
"""

from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Usage:
        from core.utils.config import get_settings
        settings = get_settings()
        print(settings.app_name)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = "Hummingbird-LEA"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-this-in-production"
    
    # -------------------------------------------------------------------------
    # Server
    # -------------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    
    # -------------------------------------------------------------------------
    # Ollama (Local AI)
    # -------------------------------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_model_chat: str = "llama3.1:8b"
    ollama_model_code: str = "deepseek-coder:6.7b"
    ollama_model_vision: str = "llava-llama3:8b"
    ollama_model_embed: str = "nomic-embed-text"
    ollama_timeout: int = 120  # seconds
    
    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------
    admin_username: str = "dre"
    admin_password: str = "changeme123"
    jwt_secret_key: str = "change-this-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./data/hummingbird.db"
    
    # -------------------------------------------------------------------------
    # File Storage
    # -------------------------------------------------------------------------
    upload_dir: str = "./data/uploads"
    knowledge_dir: str = "./data/knowledge"
    memory_dir: str = "./data/memory"
    templates_dir: str = "./data/templates"
    max_upload_size_mb: int = 50
    
    # -------------------------------------------------------------------------
    # Agent Settings
    # -------------------------------------------------------------------------
    default_agent: str = "lea"
    max_conversation_history: int = 50
    confidence_threshold: float = 0.85
    max_reasoning_steps: int = 10
    
    # -------------------------------------------------------------------------
    # Optional Cloud AI (fallback)
    # -------------------------------------------------------------------------
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = "INFO"
    log_file: Optional[str] = "./data/hummingbird.log"

    # -------------------------------------------------------------------------
    # Security (Production)
    # -------------------------------------------------------------------------
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    rate_limit_burst: int = 10
    rate_limit_block_minutes: int = 15

    # Session
    session_secure: bool = False  # Set True in production with HTTPS
    session_httponly: bool = True
    session_samesite: str = "lax"

    # CORS
    cors_origins: str = "*"  # Restrict in production

    # -------------------------------------------------------------------------
    # Performance (Caching)
    # -------------------------------------------------------------------------
    cache_max_size: int = 5000
    cache_default_ttl: int = 300  # 5 minutes
    embedding_cache_size: int = 10000
    embedding_cache_ttl: int = 3600  # 1 hour

    # -------------------------------------------------------------------------
    # Workers (Gunicorn/Uvicorn)
    # -------------------------------------------------------------------------
    workers: int = 4
    worker_connections: int = 1000
    worker_timeout: int = 120

    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"
    
    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def knowledge_path(self) -> Path:
        path = Path(self.knowledge_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def memory_path(self) -> Path:
        path = Path(self.memory_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def templates_path(self) -> Path:
        path = Path(self.templates_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list:
        """Parse CORS origins as list"""
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def log_path(self) -> Optional[Path]:
        """Get log file path"""
        if self.log_file:
            path = Path(self.log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return None


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid re-reading .env on every call.
    """
    return Settings()


# Convenience export
settings = get_settings()
