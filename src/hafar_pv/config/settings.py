"""Centralized application configuration using Pydantic settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseSettings, Field


class AppSettings(BaseSettings):
    """Runtime configuration pulled from environment variables or defaults."""

    env: Literal["dev", "test", "prod"] = Field("dev", description="Execution environment.")
    data_root: Path = Field(Path("data"), description="Base directory for all datasets.")
    models_root: Path = Field(Path("models"), description="Directory to store trained weights.")
    experiments_root: Path = Field(
        Path("experiments"), description="Directory where experiment artifacts are stored."
    )

    class Config:
        env_prefix = "HAFAR_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached settings instance."""

    return AppSettings()
