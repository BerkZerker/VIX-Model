"""Trade suggestion logic for the VIX Alert Bot.

Implements tiered instrument selection, DTE heuristics, strike heuristics,
position sizing, settlement notes, and liquidity checks per the spec.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from bot.feature_pipeline import FeatureVector
from bot.inference import Prediction

logger = logging.getLogger(__name__)


@dataclass
class TradeSuggestion:
    """Complete trade suggestion output."""

    # Tier classification
    tier: str  # "major_spike", "moderate", "mild"
    tier_label: str  # Human-readable, e.g. "Major Spike Tier"

    # Instrument
    instrument: str  # e.g. "UVXY puts", "VIX puts", "Short VIX call spread"
    settlement_type: str  # e.g. "American-style, equity settlement"
    settlement_note: str  # Additional context

    # Strike and expiry
    suggested_strike: str  # e.g. "$28" or "21"
    strike_rationale: str
    suggested_expiry: str  # e.g. "May 2026 (~90 DTE)"
    dte: int

    # Sizing
    position_size: str  # "FULL", "HALF", "SMALL"
    max_risk_pct: float  # e.g. 0.03
    sizing_rationale: str

    # Liquidity (mock or real)
    bid: float | None = None
    ask: float | None = None
    spread_pct: float | None = None
    open_interest: int | None = None
    liquidity_ok: bool = True
    liquidity_note: str = ""


def classify_tier(vix: float) -> str:
    """Classify VIX level into tiers."""
    if vix >= 30:
        return "major_spike"
    elif vix >= 22:
        return "moderate"
    else:
        return "mild"


TIER_LABELS = {
    "major_spike": "Major Spike Tier",
    "moderate": "Moderate Elevation Tier",
    "mild": "Mild Elevation Tier",
}


def suggest_trade(
    features: FeatureVector,
    prediction: Prediction,
    mock_liquidity: bool = True,
) -> TradeSuggestion:
    """Generate a complete trade suggestion based on VIX level and model output.

    Args:
        features: Current feature vector with market snapshot data.
        prediction: Model prediction output.
        mock_liquidity: If True, generate synthetic liquidity data.

    Returns:
        TradeSuggestion with all trade parameters.
    """
    vix = features.vix_spot
    tier = classify_tier(vix)
    tier_label = TIER_LABELS[tier]

    # ── Instrument selection ────────────────────────────────────────

    if tier == "major_spike":
        instrument = "UVXY puts"
        settlement_type = "American-style, equity settlement"
        settlement_note = (
            "UVXY options are American-style and can be exercised early. "
            "Standard equity option settlement (T+1)."
        )
    elif tier == "moderate":
        # Prefer VIX puts; short call spread as alternative
        if features.term_slope < -0.02:
            # Backwardation — VIX puts more attractive
            instrument = "VIX puts"
        else:
            instrument = "VIX puts"
        settlement_type = "European-style, AM cash settlement"
        settlement_note = (
            "VIX options are European-style, cash-settled. Settlement value (VRO) "
            "is calculated from opening SPX option prices, NOT VIX spot close. "
            "Can settle significantly different from prior day's VIX close."
        )
    else:  # mild
        if prediction.p_revert > 0.75:
            instrument = "Small UVXY puts"
            settlement_type = "American-style, equity settlement"
            settlement_note = (
                "UVXY options are American-style. Small position given mild elevation."
            )
        else:
            instrument = "Short VIX call spread"
            settlement_type = "European-style, AM cash settlement"
            settlement_note = (
                "VIX options are European-style, cash-settled. Spread limits risk "
                "given uncertain setup at mild VIX elevation."
            )

    # ── DTE selection ───────────────────────────────────────────────

    if tier == "major_spike":
        dte = 75  # 60-90 DTE range, pick middle
        dte_range = "60-90"
    elif tier == "moderate":
        dte = 45  # 30-60 DTE range
        dte_range = "30-60"
    else:
        dte = 45
        dte_range = "30-60"

    expiry_date = datetime.now(timezone.utc) + timedelta(days=dte)
    expiry_str = f"{expiry_date.strftime('%b %Y')} (~{dte} DTE)"

    # ── Strike selection ────────────────────────────────────────────

    m1 = features.futures.get("VX_M1")

    if "UVXY" in instrument:
        # UVXY puts: strike at 60-70% of current price
        # Approximate UVXY price from VIX (rough heuristic for mock)
        uvxy_approx = vix * 1.5  # very rough approximation
        strike_pct = 0.65
        strike_val = round(uvxy_approx * strike_pct, 0)
        suggested_strike = f"${strike_val:.0f}"
        strike_rationale = (
            f"Strike at ~{strike_pct*100:.0f}% of estimated UVXY price "
            f"(~${uvxy_approx:.1f}), targeting expected decay on VIX reversion."
        )
    elif "VIX puts" in instrument:
        # VIX puts: strike near front-month futures (ATM relative to futures)
        if m1:
            strike_val = round(m1)
            suggested_strike = f"{strike_val}"
            strike_rationale = (
                f"Strike near front-month futures ({m1:.1f}), ATM relative to "
                "futures settlement, not VIX spot."
            )
        else:
            strike_val = round(vix - 2)
            suggested_strike = f"{strike_val}"
            strike_rationale = "Strike near VIX spot minus small buffer (futures data unavailable)."
    elif "call spread" in instrument:
        # Short VIX call spread: short near spot, long 5-10 pts higher
        short_strike = round(vix)
        long_strike = short_strike + 7
        suggested_strike = f"{short_strike}/{long_strike} call spread"
        strike_val = short_strike
        strike_rationale = (
            f"Short {short_strike} / Long {long_strike} call spread. "
            "Short strike near current spot, long strike caps risk."
        )
    else:
        strike_val = round(vix)
        suggested_strike = f"{strike_val}"
        strike_rationale = "Strike near current VIX level."

    # ── Position sizing ─────────────────────────────────────────────

    p_revert = prediction.p_revert
    p_spike = prediction.p_spike_first

    if vix >= 30 and p_revert > 0.8 and p_spike < 0.2:
        position_size = "FULL"
        max_risk_pct = 0.03
        sizing_rationale = (
            f"Full size: VIX {vix:.1f} (>30), P(revert) {p_revert:.2f} (>0.8), "
            f"P(spike first) {p_spike:.2f} (<0.2). Max risk: 3% of portfolio."
        )
    elif vix >= 22 and p_revert > 0.7 and p_spike < 0.3:
        position_size = "HALF"
        max_risk_pct = 0.015
        sizing_rationale = (
            f"Half size: VIX {vix:.1f} (22-30), P(revert) {p_revert:.2f} (>0.7), "
            f"P(spike first) {p_spike:.2f} (<0.3). Max risk: 1.5% of portfolio."
        )
    else:
        position_size = "SMALL"
        max_risk_pct = 0.01
        sizing_rationale = (
            f"Small/starter size: VIX {vix:.1f}, P(revert) {p_revert:.2f}, "
            f"P(spike first) {p_spike:.2f}. Max risk: 1% of portfolio."
        )

    # ── Liquidity checks ───────────────────────────────────────────

    if mock_liquidity:
        bid, ask, oi, liquidity_ok, liquidity_note = _mock_liquidity_check(
            instrument, strike_val, vix
        )
    else:
        # Real liquidity checks would go through IBKR here
        bid, ask, oi = None, None, None
        liquidity_ok = True
        liquidity_note = "Live liquidity check not implemented yet."

    spread_pct = None
    if bid is not None and ask is not None:
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid if mid > 0 else 0.0

    return TradeSuggestion(
        tier=tier,
        tier_label=tier_label,
        instrument=instrument,
        settlement_type=settlement_type,
        settlement_note=settlement_note,
        suggested_strike=suggested_strike,
        strike_rationale=strike_rationale,
        suggested_expiry=expiry_str,
        dte=dte,
        position_size=position_size,
        max_risk_pct=max_risk_pct,
        sizing_rationale=sizing_rationale,
        bid=bid,
        ask=ask,
        spread_pct=spread_pct,
        open_interest=oi,
        liquidity_ok=liquidity_ok,
        liquidity_note=liquidity_note,
    )


def _mock_liquidity_check(
    instrument: str, strike_val: float, vix: float
) -> tuple[float, float, int, bool, str]:
    """Generate synthetic liquidity data for demo mode."""
    # Simulate realistic bid-ask based on instrument type
    if "UVXY" in instrument:
        mid = max(0.5, (vix - 15) * 0.5 + random.uniform(0, 2))
        spread_factor = 0.08  # 8% typical for UVXY
    elif "VIX put" in instrument.lower():
        mid = max(0.3, (vix - strike_val) * 0.4 + random.uniform(0.5, 2))
        spread_factor = 0.05  # 5% typical for liquid VIX options
    else:
        mid = max(0.2, random.uniform(0.5, 3))
        spread_factor = 0.07

    half_spread = mid * spread_factor / 2
    bid = round(mid - half_spread, 2)
    ask = round(mid + half_spread, 2)
    oi = random.randint(200, 5000)

    actual_spread_pct = (ask - bid) / mid if mid > 0 else 0.0
    liquidity_ok = actual_spread_pct < 0.15 and oi > 500

    if not liquidity_ok:
        reasons = []
        if actual_spread_pct >= 0.15:
            reasons.append(f"spread {actual_spread_pct:.1%} exceeds 15% threshold")
        if oi <= 500:
            reasons.append(f"open interest {oi} below 500 minimum")
        note = "Liquidity concern: " + "; ".join(reasons) + ". Consider nearest liquid alternative."
    else:
        note = f"Liquidity OK: spread {actual_spread_pct:.1%}, OI {oi}"

    return bid, ask, oi, liquidity_ok, note
