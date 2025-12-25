# echelon/server/config.py
"""Server configuration with sensible defaults for LAN use."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server settings, overridable via environment variables."""

    # SSE
    SSE_MAX_CLIENTS: int = 16
    SSE_KEEPALIVE_S: float = 15.0
    SSE_BROADCAST_TIMEOUT_S: float = 1.0

    # Caches
    WORLD_CACHE_MAX: int = 32
    NAV_CACHE_MAX: int = 8

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8090
    RUNS_DIR: Path = Path("runs")

    model_config = SettingsConfigDict(env_prefix="ECHELON_", env_file=".env", extra="ignore")


settings = Settings()
