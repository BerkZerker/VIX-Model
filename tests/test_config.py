"""Tests for bot/config.py - Configuration loading."""

import os
from pathlib import Path

import pytest

from bot.config import (
    AlertConfig,
    BotConfig,
    DataConfig,
    HealthConfig,
    IBKRConfig,
    ModelConfig,
    TelegramConfig,
)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_ibkr_defaults(self):
        cfg = IBKRConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 4001
        assert cfg.client_id == 1
        assert cfg.timeout == 30
        assert cfg.readonly is True

    def test_telegram_defaults(self):
        cfg = TelegramConfig()
        assert cfg.bot_token == ""
        assert cfg.chat_id == ""
        assert cfg.is_configured is False

    def test_telegram_configured(self):
        cfg = TelegramConfig(bot_token="abc123", chat_id="456")
        assert cfg.is_configured is True

    def test_telegram_not_configured_missing_token(self):
        cfg = TelegramConfig(bot_token="", chat_id="456")
        assert cfg.is_configured is False

    def test_telegram_not_configured_missing_chat_id(self):
        cfg = TelegramConfig(bot_token="abc123", chat_id="")
        assert cfg.is_configured is False

    def test_model_defaults(self):
        cfg = ModelConfig()
        assert cfg.version == "v001"
        assert cfg.p_revert_threshold == 0.7
        assert cfg.p_spike_threshold == 0.3
        assert cfg.zscore_threshold == 1.0
        assert isinstance(cfg.models_dir, Path)

    def test_alert_defaults(self):
        cfg = AlertConfig()
        assert cfg.cooldown_hours == 24
        assert cfg.max_portfolio_risk_full == 0.03
        assert cfg.max_portfolio_risk_half == 0.015
        assert cfg.max_portfolio_risk_small == 0.01

    def test_data_defaults(self):
        cfg = DataConfig()
        assert isinstance(cfg.data_dir, Path)
        assert isinstance(cfg.db_path, Path)
        assert cfg.poll_interval_seconds == 300
        assert cfg.staleness_threshold_seconds == 300

    def test_health_defaults(self):
        cfg = HealthConfig()
        assert cfg.port == 8080
        assert cfg.host == "0.0.0.0"


# ---------------------------------------------------------------------------
# BotConfig
# ---------------------------------------------------------------------------

class TestBotConfig:
    def test_default_creation(self):
        cfg = BotConfig()
        assert cfg.mock_mode is True
        assert isinstance(cfg.ibkr, IBKRConfig)
        assert isinstance(cfg.telegram, TelegramConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.alert, AlertConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.health, HealthConfig)

    def test_from_env_defaults(self, monkeypatch):
        """from_env should produce sensible defaults without any env vars."""
        # Clear relevant env vars to ensure defaults
        for key in [
            "IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID",
            "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
            "MODEL_VERSION", "MODELS_DIR",
            "ALERT_P_REVERT_THRESHOLD", "ALERT_P_SPIKE_THRESHOLD",
            "ALERT_ZSCORE_THRESHOLD", "ALERT_COOLDOWN_HOURS",
            "DATA_DIR", "DB_PATH", "HEALTH_PORT",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg = BotConfig.from_env()
        assert cfg.ibkr.host == "127.0.0.1"
        assert cfg.ibkr.port == 4001
        assert cfg.model.version == "v001"
        assert cfg.model.p_revert_threshold == 0.7
        # Without telegram configured, mock_mode should be True
        assert cfg.mock_mode is True

    def test_from_env_with_telegram(self, monkeypatch):
        """from_env with telegram env vars should disable mock mode."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "456789")

        cfg = BotConfig.from_env()
        assert cfg.telegram.bot_token == "123:ABC"
        assert cfg.telegram.chat_id == "456789"
        assert cfg.telegram.is_configured is True
        assert cfg.mock_mode is False

    def test_from_env_custom_ports(self, monkeypatch):
        monkeypatch.setenv("IBKR_PORT", "7496")
        monkeypatch.setenv("HEALTH_PORT", "9090")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

        cfg = BotConfig.from_env()
        assert cfg.ibkr.port == 7496
        assert cfg.health.port == 9090

    def test_from_env_custom_thresholds(self, monkeypatch):
        monkeypatch.setenv("ALERT_P_REVERT_THRESHOLD", "0.8")
        monkeypatch.setenv("ALERT_P_SPIKE_THRESHOLD", "0.2")
        monkeypatch.setenv("ALERT_ZSCORE_THRESHOLD", "1.5")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

        cfg = BotConfig.from_env()
        assert cfg.model.p_revert_threshold == 0.8
        assert cfg.model.p_spike_threshold == 0.2
        assert cfg.model.zscore_threshold == 1.5

    def test_from_env_custom_model_version(self, monkeypatch):
        monkeypatch.setenv("MODEL_VERSION", "v003")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

        cfg = BotConfig.from_env()
        assert cfg.model.version == "v003"
