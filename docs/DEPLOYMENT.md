# Deployment Guide

## Overview

The bot is designed to run continuously on a low-power device (e.g., Raspberry Pi 5) with:

- IB Gateway for market data
- The bot polling every 5 minutes during market hours
- Telegram for alerts
- A health endpoint for monitoring

## Requirements

- Python 3.11+
- IB Gateway or TWS running and accessible
- Telegram bot token and chat ID
- ~500 MB RAM, minimal CPU

## Step 1: Install Dependencies

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> && cd VIX-Model
uv venv
uv pip install -e .

# macOS only: XGBoost needs libomp
brew install libomp
```

## Step 2: Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
IBKR_HOST=127.0.0.1
IBKR_PORT=4001
IBKR_CLIENT_ID=1

TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=123456789

MODEL_VERSION=v001
```

## Step 3: Set Up IB Gateway

1. Download [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
2. Log in with your IBKR credentials
3. Enable API access: Configure > API > Settings
   - Check "Enable ActiveX and Socket Clients"
   - Set Socket Port to `4001` (live) or `4002` (paper)
   - Uncheck "Read-Only API" if you need order placement later
4. Add `127.0.0.1` to Trusted IPs

## Step 4: Run the Bot

```bash
# Live mode
uv run python -m bot.main

# Or with explicit log level
uv run python -m bot.main --log-level INFO
```

The bot will:

- Start polling immediately if market is open
- Send a daily digest at 4:05 PM ET
- Run heartbeats every hour during off-hours
- Expose health endpoints on port 8080

## Step 5: Verify

```bash
# Check health
curl http://localhost:8080/health

# Check detailed status
curl http://localhost:8080/status
```

## Running as a Service (Linux/Pi)

Create a systemd service file at `/etc/systemd/system/vix-bot.service`:

```ini
[Unit]
Description=VIX Alert Bot
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/VIX-Model
Environment=PATH=/home/pi/VIX-Model/.venv/bin:/usr/bin:/bin
ExecStart=/home/pi/VIX-Model/.venv/bin/python -m bot.main
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable vix-bot
sudo systemctl start vix-bot
sudo journalctl -u vix-bot -f  # View logs
```

## Running with Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv venv && uv pip install -e .

EXPOSE 8080

CMD ["python", "-m", "bot.main"]
```

```bash
docker build -t vix-bot .
docker run -d --name vix-bot \
    --env-file .env \
    -p 8080:8080 \
    --restart unless-stopped \
    vix-bot
```

## Alert Behavior

### Alert Conditions

An alert fires when ALL of these are true:

- `p_revert > 0.7` (model predicts high probability of VIX drop)
- `p_spike_first < 0.3` (low risk of further spike)
- `vix_zscore > 1.0` (VIX is meaningfully elevated)

### Cooldown

After sending an alert for a given tier (major/moderate/mild), the bot won't send another alert for the same tier for 24 hours (configurable via `ALERT_COOLDOWN_HOURS`). Tier upgrades (e.g., moderate → major) bypass the cooldown.

### Daily Digest

Sent at 4:05 PM ET every trading day with:

- Current VIX level and z-score
- Model predictions (p_revert, p_spike_first)
- Term structure status
- Any data staleness events from the day

## Monitoring

### Health Endpoint Responses

**Healthy (HTTP 200):**

```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "model_version": "v001",
  "data_fresh": true,
  "memory_mb": 120
}
```

**Degraded (HTTP 503):**

```json
{
  "status": "degraded",
  "data_fresh": false,
  "staleness": {
    "VIX": { "stale": true, "age_seconds": 600 }
  }
}
```

### Logs

Key log messages to watch for:

| Message                                      | Meaning                      |
| -------------------------------------------- | ---------------------------- |
| `Alert sent for tier: MAJOR_SPIKE`           | Trade alert delivered        |
| `Critical data stale: ['VIX']`               | Data feed interrupted        |
| `Poller not connected, attempting reconnect` | IBKR connection lost         |
| `Daily digest sent`                          | End-of-day summary delivered |
| `No alert: p_revert=0.55 (need >0.70)`       | Signal below threshold       |

## Troubleshooting

### Bot won't connect to IBKR

- Verify IB Gateway is running and logged in
- Check port matches (`4001` for live, `4002` for paper)
- Ensure API connections are enabled in Gateway settings
- Check that `127.0.0.1` is in Trusted IPs

### No alerts firing

- Run with `--log-level DEBUG` to see prediction values
- Check if market is open (bot skips polling when market is closed)
- Verify model is loaded: check health endpoint for `model_version`
- Lower thresholds temporarily for testing (not recommended for live trading)

### Data staleness warnings

- Check IBKR connection status
- Verify market data subscriptions are active
- Check if market is in a holiday/early close
