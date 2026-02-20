"""Configuration management for the VIX Alert Bot.

Loads settings from .env file with sensible defaults for mock/demo mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file, or cwd)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 4001
    client_id: int = 1
    timeout: int = 30
    readonly: bool = True


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass
class ModelConfig:
    version: str = "v001"
    models_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "models")
    # Alert thresholds (from spec: p_revert > 0.7 AND p_spike_first < 0.3 AND vix_zscore > 1.0)
    p_revert_threshold: float = 0.7
    p_spike_threshold: float = 0.3
    zscore_threshold: float = 1.0


@dataclass
class AlertConfig:
    cooldown_hours: int = 24
    max_portfolio_risk_full: float = 0.03  # 3%
    max_portfolio_risk_half: float = 0.015  # 1.5%
    max_portfolio_risk_small: float = 0.01  # 1%


@dataclass
class DataConfig:
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    db_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "vix_bot.db")
    poll_interval_seconds: int = 300  # 5 minutes
    staleness_threshold_seconds: int = 300  # 5 minutes


@dataclass
class HealthConfig:
    port: int = 8080
    host: str = "0.0.0.0"


@dataclass
class BotConfig:
    """Top-level configuration container."""

    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    data: DataConfig = field(default_factory=DataConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    mock_mode: bool = True

    @classmethod
    def from_env(cls) -> BotConfig:
        """Build configuration from environment variables with defaults."""
        ibkr = IBKRConfig(
            host=os.getenv("IBKR_HOST", "127.0.0.1"),
            port=int(os.getenv("IBKR_PORT", "4001")),
            client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
        )
        telegram = TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )
        model = ModelConfig(
            version=os.getenv("MODEL_VERSION", "v001"),
            models_dir=Path(os.getenv("MODELS_DIR", str(_PROJECT_ROOT / "models"))),
            p_revert_threshold=float(os.getenv("ALERT_P_REVERT_THRESHOLD", "0.7")),
            p_spike_threshold=float(os.getenv("ALERT_P_SPIKE_THRESHOLD", "0.3")),
            zscore_threshold=float(os.getenv("ALERT_ZSCORE_THRESHOLD", "1.0")),
        )
        alert = AlertConfig(
            cooldown_hours=int(os.getenv("ALERT_COOLDOWN_HOURS", "24")),
        )
        data = DataConfig(
            data_dir=Path(os.getenv("DATA_DIR", str(_PROJECT_ROOT / "data"))),
            db_path=Path(os.getenv("DB_PATH", str(_PROJECT_ROOT / "data" / "vix_bot.db"))),
        )
        health = HealthConfig(
            port=int(os.getenv("HEALTH_PORT", "8080")),
        )

        # Auto-detect mock mode: mock if telegram not configured
        mock_mode = not telegram.is_configured

        return cls(
            ibkr=ibkr,
            telegram=telegram,
            model=model,
            alert=alert,
            data=data,
            health=health,
            mock_mode=mock_mode,
        )
