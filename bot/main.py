"""Main entry point for the VIX Alert Bot.

Sets up APScheduler, market calendar awareness, and the main
poll -> features -> inference -> alert pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading
from datetime import datetime, time, timedelta, timezone

import exchange_calendars as xcals
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from bot.config import BotConfig
from bot.data_poller import BasePoller, IBKRPoller, MockPoller
from bot.db import Database
from bot.feature_pipeline import FeaturePipeline, FeatureVector
from bot.health import app as health_app
from bot.health import set_state, update_last_data_timestamp
from bot.inference import BaseInference, MockInference, ModelInference
from bot.notifier import BaseNotifier, MockNotifier, TelegramNotifier
from bot.staleness import StalenessTracker
from bot.trade_suggester import classify_tier, suggest_trade

logger = logging.getLogger("bot")

# US Eastern timezone offset (for market hour checks)
ET = timezone(timedelta(hours=-5))


class VIXAlertBot:
    """Orchestrates the full polling -> inference -> notification pipeline."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.db = Database(config.data.db_path)
        self.staleness = StalenessTracker(config.data.staleness_threshold_seconds)
        self.pipeline = FeaturePipeline(self.db)
        self.scheduler = AsyncIOScheduler()
        self._shutdown_event = asyncio.Event()
        self._staleness_events_today = 0
        self._calendar = xcals.get_calendar("XNYS")  # NYSE calendar

        # Components initialized in setup()
        self.poller: BasePoller | None = None
        self.inference: BaseInference | None = None
        self.notifier: BaseNotifier | None = None

    async def setup(self) -> None:
        """Initialize all components."""
        cfg = self.config

        # Database
        await self.db.connect()

        # Poller
        if cfg.mock_mode:
            self.poller = MockPoller()
        else:
            self.poller = IBKRPoller(
                host=cfg.ibkr.host,
                port=cfg.ibkr.port,
                client_id=cfg.ibkr.client_id,
            )
        await self.poller.connect()

        # Inference
        if cfg.mock_mode:
            self.inference = MockInference()
        else:
            self.inference = ModelInference(
                models_dir=cfg.model.models_dir,
                version=cfg.model.version,
            )
        self.inference.load()

        # Notifier
        if cfg.mock_mode or not cfg.telegram.is_configured:
            self.notifier = MockNotifier(cooldown_hours=cfg.alert.cooldown_hours)
        else:
            self.notifier = TelegramNotifier(
                bot_token=cfg.telegram.bot_token,
                chat_id=cfg.telegram.chat_id,
                cooldown_hours=cfg.alert.cooldown_hours,
            )

        # Wire health endpoint state
        set_state(
            model_version=self.inference.model_version(),
            staleness_tracker=self.staleness,
            db=self.db,
        )

        logger.info(
            "Bot initialized (mock=%s, model=%s)",
            cfg.mock_mode,
            self.inference.model_version(),
        )

    def _is_market_open(self) -> bool:
        """Check if the market is currently open using exchange_calendars."""
        now = datetime.now(timezone.utc)
        today = now.date()
        try:
            if not self._calendar.is_session(today):
                return False
            # Get open/close times for today
            open_time = self._calendar.session_open(today)
            close_time = self._calendar.session_close(today)
            return open_time <= now <= close_time
        except Exception:
            # Fallback: assume market hours Mon-Fri 9:30-16:00 ET
            et_now = now.astimezone(ET)
            if et_now.weekday() >= 5:
                return False
            market_open = time(9, 30)
            market_close = time(16, 0)
            return market_open <= et_now.time() <= market_close

    def _is_trading_day(self) -> bool:
        """Check if today is a trading day."""
        today = datetime.now(timezone.utc).date()
        try:
            return self._calendar.is_session(today)
        except Exception:
            return datetime.now(timezone.utc).weekday() < 5

    async def poll_cycle(self) -> None:
        """Single poll -> feature -> inference -> alert cycle."""
        try:
            if not self._is_market_open():
                logger.debug("Market closed, skipping poll cycle")
                return

            if self.poller is None or not self.poller.is_connected():
                logger.warning("Poller not connected, attempting reconnect...")
                if self.poller:
                    try:
                        await self.poller.connect()
                    except Exception:
                        logger.exception("Reconnect failed")
                        return
                else:
                    return

            # 1. Poll market data
            snapshot = await self.poller.poll()
            logger.info("Polled %d bars, %d futures", len(snapshot.bars), len(snapshot.futures))

            # 2. Store bars in database
            for bar in snapshot.bars:
                await self.db.insert_bar(
                    timestamp=bar.timestamp,
                    symbol=bar.symbol,
                    open_=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )

            # 3. Update staleness tracker
            symbols = [b.symbol for b in snapshot.bars]
            if snapshot.futures:
                symbols.append("VX_M1")
            if snapshot.vvix is not None:
                symbols.append("VVIX")
            if snapshot.vix9d is not None:
                symbols.append("VIX9D")
            if snapshot.skew is not None:
                symbols.append("SKEW")
            self.staleness.record_updates(symbols)

            update_last_data_timestamp(snapshot.timestamp)

            # 4. Check staleness
            staleness_status = self.staleness.check()
            if staleness_status.any_critical_stale:
                stale = self.staleness.stale_symbols()
                logger.warning("Critical data stale: %s — skipping inference", stale)
                self._staleness_events_today += 1
                last_times = {
                    s: (self.staleness.get_last_update(s).strftime("%H:%M UTC")
                        if self.staleness.get_last_update(s) else "never")
                    for s in stale
                }
                await self.notifier.send_staleness_warning(stale, last_times)
                return

            # 5. Compute features
            features = await self.pipeline.compute(
                vvix=snapshot.vvix,
                vix9d=snapshot.vix9d,
                skew=snapshot.skew,
                futures=snapshot.futures,
            )
            if not features.is_valid:
                logger.warning("Feature computation returned invalid — insufficient data")
                return

            # 6. Run inference
            prediction = self.inference.predict(features.daily_features)
            logger.info(
                "Prediction: p_revert=%.3f, p_spike=%.3f, magnitude=%.1f",
                prediction.p_revert,
                prediction.p_spike_first,
                prediction.expected_magnitude,
            )

            # 7. Log signal
            tier = classify_tier(features.vix_spot)
            await self.db.insert_signal(
                timestamp=features.computed_at,
                model_version=prediction.model_version,
                p_revert=prediction.p_revert,
                p_spike_first=prediction.p_spike_first,
                expected_magnitude=prediction.expected_magnitude,
                tier=tier,
            )

            # 8. Check alert conditions
            cfg = self.config.model
            if (
                prediction.p_revert > cfg.p_revert_threshold
                and prediction.p_spike_first < cfg.p_spike_threshold
                and features.vix_zscore > cfg.zscore_threshold
            ):
                trade = suggest_trade(
                    features, prediction, mock_liquidity=self.config.mock_mode
                )
                sent = await self.notifier.send_alert(features, prediction, trade)
                if sent:
                    # Update the signal record to mark alert as sent
                    signals = await self.db.get_signals_today()
                    if signals:
                        latest_id = signals[0]["id"]
                        await self.db.db.execute(
                            "UPDATE signals SET alert_sent = 1, tier = ? WHERE id = ?",
                            (tier, latest_id),
                        )
                        await self.db.db.commit()
                    logger.info("Alert sent for tier: %s", tier)
                else:
                    logger.info("Alert not sent (cooldown or error)")
            else:
                logger.debug(
                    "No alert: p_revert=%.2f (need >%.2f), p_spike=%.2f (need <%.2f), "
                    "zscore=%.2f (need >%.2f)",
                    prediction.p_revert,
                    cfg.p_revert_threshold,
                    prediction.p_spike_first,
                    cfg.p_spike_threshold,
                    features.vix_zscore,
                    cfg.zscore_threshold,
                )

        except Exception:
            logger.exception("Error in poll cycle")

    async def daily_digest(self) -> None:
        """Send end-of-day digest."""
        try:
            if not self._is_trading_day():
                logger.debug("Not a trading day, skipping digest")
                return

            # Compute current features for digest
            features = await self.pipeline.compute()
            if not features.is_valid:
                logger.warning("Cannot generate digest — feature computation invalid")
                return

            prediction = self.inference.predict(features.daily_features)

            await self.notifier.send_daily_digest(
                features, prediction, staleness_events=self._staleness_events_today
            )

            # Store digest in DB
            await self.db.insert_daily_digest(
                dt=datetime.now(timezone.utc).date(),
                vix_close=features.vix_spot,
                zscore=features.vix_zscore,
                p_revert=prediction.p_revert,
                p_spike_first=prediction.p_spike_first,
                term_slope=features.term_slope,
            )

            # Reset daily staleness counter
            self._staleness_events_today = 0
            logger.info("Daily digest sent")

        except Exception:
            logger.exception("Error sending daily digest")

    async def heartbeat(self) -> None:
        """Lightweight heartbeat during off-hours / weekends."""
        logger.debug("Heartbeat: bot alive, market closed")

    async def run(self) -> None:
        """Start the bot scheduler and run until shutdown."""
        await self.setup()

        # Main polling job: every 5 minutes during market hours
        self.scheduler.add_job(
            self.poll_cycle,
            IntervalTrigger(seconds=self.config.data.poll_interval_seconds),
            id="poll_cycle",
            name="Market data poll cycle",
            max_instances=1,
        )

        # Daily digest at 16:05 ET (just after market close)
        self.scheduler.add_job(
            self.daily_digest,
            CronTrigger(hour=21, minute=5, timezone="UTC"),  # 16:05 ET = 21:05 UTC
            id="daily_digest",
            name="Daily digest",
        )

        # Heartbeat every hour (for weekend/holiday monitoring)
        self.scheduler.add_job(
            self.heartbeat,
            IntervalTrigger(hours=1),
            id="heartbeat",
            name="Heartbeat",
        )

        self.scheduler.start()
        logger.info("Scheduler started. Polling every %ds.", self.config.data.poll_interval_seconds)

        # Run an initial poll immediately
        await self.poll_cycle()

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Cleanup
        self.scheduler.shutdown(wait=False)
        if self.poller:
            await self.poller.disconnect()
        await self.db.close()
        logger.info("Bot shut down gracefully")

    def request_shutdown(self) -> None:
        """Signal the bot to shut down."""
        self._shutdown_event.set()


def start_health_server(config: BotConfig) -> None:
    """Run the FastAPI health server in a background thread."""
    uvicorn.run(
        health_app,
        host=config.health.host,
        port=config.health.port,
        log_level="warning",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VIX Alert Bot")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/demo mode with synthetic data",
    )
    parser.add_argument(
        "--no-health",
        action="store_true",
        help="Disable the health check HTTP server",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = BotConfig.from_env()
    if args.mock:
        config.mock_mode = True

    logger.info("Starting VIX Alert Bot (mock=%s)", config.mock_mode)

    # Start health server in background thread
    if not args.no_health:
        health_thread = threading.Thread(
            target=start_health_server,
            args=(config,),
            daemon=True,
        )
        health_thread.start()
        logger.info("Health server started on port %d", config.health.port)

    # Create and run the bot
    bot = VIXAlertBot(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Handle SIGINT/SIGTERM for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, bot.request_shutdown)

    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        bot.request_shutdown()
        # Give time for cleanup
        loop.run_until_complete(asyncio.sleep(1))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
