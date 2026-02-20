"""Tests for bot/trade_suggester.py - Trade suggestion logic."""

import pytest
from bot.feature_pipeline import FeatureVector
from bot.inference import Prediction
from bot.trade_suggester import (
    TradeSuggestion,
    classify_tier,
    suggest_trade,
    TIER_LABELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_features(vix: float, term_slope: float = 0.03, m1: float | None = None) -> FeatureVector:
    """Helper to build a FeatureVector with controlled values."""
    return FeatureVector(
        daily_features={"vix_spot": vix},
        vix_spot=vix,
        vix_zscore=2.0,
        vix_percentile=0.85,
        term_slope=term_slope,
        term_slope_zscore=-1.0,
        spy_price=400.0,
        spy_drawdown=-0.03,
        vvix=110.0,
        vix9d=vix * 0.95,
        vix9d_vix_ratio=0.95,
        skew=130.0,
        days_elevated=3,
        futures={"VX_M1": m1 or vix * 1.03, "VX_M2": vix * 1.06},
        computed_at="2025-01-15 16:00:00",
        is_valid=True,
    )


def _make_prediction(p_revert: float = 0.85, p_spike: float = 0.15, magnitude: float = 18.0) -> Prediction:
    return Prediction(
        p_revert=p_revert,
        p_spike_first=p_spike,
        expected_magnitude=magnitude,
        model_version="test_v001",
    )


# ---------------------------------------------------------------------------
# classify_tier
# ---------------------------------------------------------------------------

class TestClassifyTier:
    def test_major_spike(self):
        assert classify_tier(30.0) == "major_spike"
        assert classify_tier(45.0) == "major_spike"

    def test_moderate(self):
        assert classify_tier(22.0) == "moderate"
        assert classify_tier(29.9) == "moderate"

    def test_mild(self):
        assert classify_tier(18.0) == "mild"
        assert classify_tier(21.9) == "mild"
        assert classify_tier(10.0) == "mild"

    def test_tier_labels_exist(self):
        for tier in ["major_spike", "moderate", "mild"]:
            assert tier in TIER_LABELS


# ---------------------------------------------------------------------------
# Instrument selection
# ---------------------------------------------------------------------------

class TestInstrumentSelection:
    def test_major_spike_uses_uvxy_puts(self):
        features = _make_features(vix=35.0)
        prediction = _make_prediction()
        trade = suggest_trade(features, prediction, mock_liquidity=True)
        assert "UVXY" in trade.instrument
        assert trade.tier == "major_spike"

    def test_moderate_uses_vix_puts(self):
        features = _make_features(vix=25.0)
        prediction = _make_prediction()
        trade = suggest_trade(features, prediction, mock_liquidity=True)
        assert "VIX puts" in trade.instrument
        assert trade.tier == "moderate"

    def test_mild_high_revert_uses_small_uvxy(self):
        features = _make_features(vix=19.0)
        prediction = _make_prediction(p_revert=0.80)
        trade = suggest_trade(features, prediction, mock_liquidity=True)
        # With p_revert > 0.75, mild tier should pick Small UVXY puts
        assert "UVXY" in trade.instrument or "call spread" in trade.instrument
        assert trade.tier == "mild"

    def test_mild_low_revert_uses_call_spread(self):
        features = _make_features(vix=19.0)
        prediction = _make_prediction(p_revert=0.60)
        trade = suggest_trade(features, prediction, mock_liquidity=True)
        assert "call spread" in trade.instrument
        assert trade.tier == "mild"


# ---------------------------------------------------------------------------
# DTE selection
# ---------------------------------------------------------------------------

class TestDTESelection:
    def test_major_spike_dte(self):
        features = _make_features(vix=35.0)
        trade = suggest_trade(features, _make_prediction(), mock_liquidity=True)
        assert trade.dte == 75

    def test_moderate_dte(self):
        features = _make_features(vix=25.0)
        trade = suggest_trade(features, _make_prediction(), mock_liquidity=True)
        assert trade.dte == 45

    def test_mild_dte(self):
        features = _make_features(vix=19.0)
        trade = suggest_trade(features, _make_prediction(), mock_liquidity=True)
        assert trade.dte == 45


# ---------------------------------------------------------------------------
# Sizing logic
# ---------------------------------------------------------------------------

class TestSizing:
    def test_full_size_conditions(self):
        """FULL size: VIX >= 30, p_revert > 0.8, p_spike < 0.2."""
        features = _make_features(vix=35.0)
        prediction = _make_prediction(p_revert=0.85, p_spike=0.15)
        trade = suggest_trade(features, prediction, mock_liquidity=True)
        assert trade.position_size == "FULL"
        assert trade.max_risk_pct == 0.03

    def test_half_size_conditions(self):
        """HALF size: VIX >= 22, p_revert > 0.7, p_spike < 0.3."""
        features = _make_features(vix=25.0)
        prediction = _make_prediction(p_revert=0.75, p_spike=0.25)
        trade = suggest_trade(features, prediction, mock_liquidity=True)
        assert trade.position_size == "HALF"
        assert trade.max_risk_pct == 0.015

    def test_small_size_fallback(self):
        """SMALL size when conditions don't meet FULL or HALF."""
        features = _make_features(vix=19.0)
        prediction = _make_prediction(p_revert=0.65, p_spike=0.35)
        trade = suggest_trade(features, prediction, mock_liquidity=True)
        assert trade.position_size == "SMALL"
        assert trade.max_risk_pct == 0.01

    def test_full_size_requires_all_conditions(self):
        """FULL size should not trigger with moderate p_spike."""
        features = _make_features(vix=35.0)
        prediction = _make_prediction(p_revert=0.85, p_spike=0.25)  # p_spike >= 0.2
        trade = suggest_trade(features, prediction, mock_liquidity=True)
        # Should fall through to HALF (VIX >= 22, p_revert > 0.7, p_spike < 0.3)
        assert trade.position_size == "HALF"


# ---------------------------------------------------------------------------
# Settlement info
# ---------------------------------------------------------------------------

class TestSettlement:
    def test_uvxy_settlement(self):
        features = _make_features(vix=35.0)
        trade = suggest_trade(features, _make_prediction(), mock_liquidity=True)
        assert "American" in trade.settlement_type

    def test_vix_settlement(self):
        features = _make_features(vix=25.0)
        trade = suggest_trade(features, _make_prediction(), mock_liquidity=True)
        assert "European" in trade.settlement_type
        assert "AM" in trade.settlement_type


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_all_fields_populated(self):
        features = _make_features(vix=30.0)
        prediction = _make_prediction()
        trade = suggest_trade(features, prediction, mock_liquidity=True)

        assert isinstance(trade, TradeSuggestion)
        assert trade.tier in ["major_spike", "moderate", "mild"]
        assert trade.tier_label != ""
        assert trade.instrument != ""
        assert trade.settlement_type != ""
        assert trade.suggested_strike != ""
        assert trade.strike_rationale != ""
        assert trade.suggested_expiry != ""
        assert trade.dte > 0
        assert trade.position_size in ["FULL", "HALF", "SMALL"]
        assert trade.max_risk_pct > 0
        assert trade.sizing_rationale != ""

    def test_mock_liquidity_provides_bid_ask(self):
        features = _make_features(vix=25.0)
        trade = suggest_trade(features, _make_prediction(), mock_liquidity=True)
        assert trade.bid is not None
        assert trade.ask is not None
        assert trade.bid < trade.ask
        assert trade.spread_pct is not None
        assert trade.open_interest is not None

    def test_no_mock_liquidity(self):
        features = _make_features(vix=25.0)
        trade = suggest_trade(features, _make_prediction(), mock_liquidity=False)
        assert trade.bid is None
        assert trade.ask is None
        assert trade.liquidity_ok is True
