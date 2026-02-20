"""Telegram notification for the VIX Alert Bot.

TelegramNotifier sends formatted alerts via python-telegram-bot.
MockNotifier logs messages to the console/file instead.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from bot.feature_pipeline import FeatureVector
from bot.inference import Prediction
from bot.trade_suggester import TradeSuggestion

logger = logging.getLogger(__name__)


class BaseNotifier(ABC):
    @abstractmethod
    async def send_alert(
        self,
        features: FeatureVector,
        prediction: Prediction,
        trade: TradeSuggestion,
    ) -> bool: ...

    @abstractmethod
    async def send_daily_digest(
        self,
        features: FeatureVector,
        prediction: Prediction,
        staleness_events: int = 0,
    ) -> bool: ...

    @abstractmethod
    async def send_staleness_warning(self, stale_symbols: list[str], last_times: dict[str, str]) -> bool: ...


def format_alert_message(
    features: FeatureVector,
    prediction: Prediction,
    trade: TradeSuggestion,
) -> str:
    """Format a full alert message per the spec templates."""
    vix = features.vix_spot

    # Header — adapts based on tier
    if trade.tier == "major_spike":
        header = "--- VIX ALERT: High Confidence Signal ---"
    elif trade.tier == "moderate":
        header = "--- VIX ALERT: Moderate Setup ---"
    else:
        header = "--- VIX ALERT: Mild Setup ---"

    # Sizing label
    risk_pct = f"{trade.max_risk_pct * 100:.1f}%"

    # Term structure description
    m1 = features.futures.get("VX_M1")
    m2 = features.futures.get("VX_M2")
    if m1 and vix > 0:
        backwardation_pct = (m1 - vix) / vix * 100
        if backwardation_pct < 0:
            term_desc = f"backwardation: {backwardation_pct:.1f}%"
        else:
            term_desc = f"contango: +{backwardation_pct:.1f}%"
    else:
        term_desc = "N/A"

    # VVIX description
    if features.vvix > 120:
        vvix_desc = "elevated"
    elif features.vvix > 100:
        vvix_desc = "moderate"
    else:
        vvix_desc = "low"

    # VIX9D/VIX interpretation
    ratio = features.vix9d_vix_ratio
    if ratio < 0.93:
        vix9d_desc = "near-term fear fading"
    elif ratio > 1.05:
        vix9d_desc = "near-term fear elevated"
    else:
        vix9d_desc = "neutral"

    # Bid-ask display
    if trade.bid is not None and trade.ask is not None:
        spread_str = (
            f"${trade.bid:.2f} / ${trade.ask:.2f} "
            f"({trade.spread_pct:.1%} spread)" if trade.spread_pct is not None
            else f"${trade.bid:.2f} / ${trade.ask:.2f}"
        )
    else:
        spread_str = "N/A"

    msg = f"""{header}
Model: vix_xgb_{prediction.model_version} ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})

P(revert): {prediction.p_revert:.2f}    P(spike first): {prediction.p_spike_first:.2f}
Expected reversion: -{prediction.expected_magnitude:.0f}%
Sizing suggestion: {trade.position_size} (max risk: {risk_pct} of portfolio)
Timestamp: {features.computed_at}

-- Market Snapshot --
VIX Spot: {vix:.1f} (z-score: {features.vix_zscore:+.1f}, {features.vix_percentile:.0%} percentile)
VVIX: {features.vvix:.0f} ({vvix_desc})
VIX9D/VIX: {features.vix9d_vix_ratio:.2f} ({vix9d_desc})
SKEW: {features.skew:.0f}
VIX Futures M1: {m1 or 0:.1f} ({term_desc})
VIX Futures M2: {m2 or 0:.1f}
Term slope z-score: {features.term_slope_zscore:+.1f}
SPY: {features.spy_price:.2f} ({features.spy_drawdown:+.1%} from 20d high)

-- Suggested Trade --
Instrument: {trade.instrument} ({trade.tier_label.lower()})
Settlement: {trade.settlement_type}
Suggested Strike: {trade.suggested_strike}
Suggested Expiry: {trade.suggested_expiry}
Bid-ask at suggested strike: {spread_str}
{trade.liquidity_note}

-- Context --
Days VIX elevated: {features.days_elevated} (above 60d mean + 1 std)
{trade.settlement_note}

---"""
    return msg


def format_daily_digest(
    features: FeatureVector,
    prediction: Prediction,
    staleness_events: int = 0,
) -> str:
    """Format the end-of-day digest message."""
    m1 = features.futures.get("VX_M1")
    m2 = features.futures.get("VX_M2")

    # Term structure summary
    if m1 and m2:
        if features.term_slope < -0.01:
            term_summary = f"Backwardation (M1: {m1:.1f}, M2: {m2:.1f}, slope: {features.term_slope:.3f})"
        elif features.term_slope > 0.01:
            term_summary = f"Contango (M1: {m1:.1f}, M2: {m2:.1f}, slope: {features.term_slope:.3f})"
        else:
            term_summary = f"Flat (M1: {m1:.1f}, M2: {m2:.1f}, slope: {features.term_slope:.3f})"
    else:
        term_summary = "Futures data unavailable"

    staleness_line = ""
    if staleness_events > 0:
        staleness_line = f"\nData staleness events today: {staleness_events}"

    return f"""--- VIX Daily Digest ---
Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

VIX Close: {features.vix_spot:.2f} (z-score: {features.vix_zscore:+.2f})
P(revert): {prediction.p_revert:.2f}    P(spike first): {prediction.p_spike_first:.2f}
Term Structure: {term_summary}

Model: vix_xgb_{prediction.model_version}
Last data: {features.computed_at}{staleness_line}

---"""


class CooldownTracker:
    """Enforces alert cooldown: max 1 alert per 24h per tier.

    Upgrade alerts (tier changes to a higher severity) bypass cooldown.
    """

    TIER_SEVERITY = {"mild": 0, "moderate": 1, "major_spike": 2}

    def __init__(self, cooldown_hours: int = 24) -> None:
        self.cooldown_hours = cooldown_hours
        self._last_alert: dict[str, datetime] = {}  # tier -> last alert time
        self._last_tier: str | None = None

    def can_send(self, tier: str) -> bool:
        """Check if an alert can be sent for this tier."""
        now = datetime.now(timezone.utc)

        # Upgrade alerts bypass cooldown
        if self._last_tier and self.TIER_SEVERITY.get(tier, 0) > self.TIER_SEVERITY.get(
            self._last_tier, 0
        ):
            return True

        last = self._last_alert.get(tier)
        if last is None:
            return True

        elapsed = (now - last).total_seconds() / 3600
        return elapsed >= self.cooldown_hours

    def record_alert(self, tier: str) -> None:
        """Record that an alert was sent for a tier."""
        self._last_alert[tier] = datetime.now(timezone.utc)
        self._last_tier = tier


class TelegramNotifier(BaseNotifier):
    """Sends alerts via Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str, cooldown_hours: int = 24) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown = CooldownTracker(cooldown_hours)

    async def send_alert(
        self,
        features: FeatureVector,
        prediction: Prediction,
        trade: TradeSuggestion,
    ) -> bool:
        if not self.cooldown.can_send(trade.tier):
            logger.info("Alert suppressed by cooldown for tier: %s", trade.tier)
            return False

        msg = format_alert_message(features, prediction, trade)
        sent = await self._send(msg)
        if sent:
            self.cooldown.record_alert(trade.tier)
        return sent

    async def send_daily_digest(
        self,
        features: FeatureVector,
        prediction: Prediction,
        staleness_events: int = 0,
    ) -> bool:
        msg = format_daily_digest(features, prediction, staleness_events)
        return await self._send(msg)

    async def send_staleness_warning(self, stale_symbols: list[str], last_times: dict[str, str]) -> bool:
        lines = ["-- Data Feed Stale --"]
        for sym in stale_symbols:
            last = last_times.get(sym, "never")
            lines.append(f"  {sym}: last quote at {last}")
        lines.append("Model inference paused until data resumes.")
        return await self._send("\n".join(lines))

    async def _send(self, text: str) -> bool:
        try:
            from telegram import Bot

            bot = Bot(token=self.bot_token)
            await bot.send_message(chat_id=self.chat_id, text=text)
            logger.info("Telegram message sent (%d chars)", len(text))
            return True
        except Exception:
            logger.exception("Failed to send Telegram message")
            return False


class MockNotifier(BaseNotifier):
    """Logs messages instead of sending them. For demo/testing."""

    def __init__(self, cooldown_hours: int = 24) -> None:
        self.cooldown = CooldownTracker(cooldown_hours)
        self.messages: list[str] = []  # Store for testing

    async def send_alert(
        self,
        features: FeatureVector,
        prediction: Prediction,
        trade: TradeSuggestion,
    ) -> bool:
        if not self.cooldown.can_send(trade.tier):
            logger.info("[MOCK] Alert suppressed by cooldown for tier: %s", trade.tier)
            return False

        msg = format_alert_message(features, prediction, trade)
        self.messages.append(msg)
        logger.info("[MOCK ALERT]\n%s", msg)
        self.cooldown.record_alert(trade.tier)
        return True

    async def send_daily_digest(
        self,
        features: FeatureVector,
        prediction: Prediction,
        staleness_events: int = 0,
    ) -> bool:
        msg = format_daily_digest(features, prediction, staleness_events)
        self.messages.append(msg)
        logger.info("[MOCK DIGEST]\n%s", msg)
        return True

    async def send_staleness_warning(self, stale_symbols: list[str], last_times: dict[str, str]) -> bool:
        lines = ["[MOCK] -- Data Feed Stale --"]
        for sym in stale_symbols:
            last = last_times.get(sym, "never")
            lines.append(f"  {sym}: last quote at {last}")
        lines.append("Model inference paused until data resumes.")
        msg = "\n".join(lines)
        self.messages.append(msg)
        logger.info(msg)
        return True
