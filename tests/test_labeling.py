"""Tests for label generation in data/scripts/build_dataset.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.scripts.build_dataset import generate_labels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def elevated_vix_df():
    """DataFrame with a VIX spike and reversion."""
    dates = pd.bdate_range("2020-01-01", periods=80)
    vix = [20.0] * 20 + [35.0] * 5 + list(np.linspace(35, 18, 25)) + [18.0] * 30
    vix = vix[:80]
    zscore = [0.5] * 20 + [2.5] * 5 + [2.0] * 10 + [1.0] * 15 + [0.3] * 30
    zscore = zscore[:80]
    return pd.DataFrame({"vix_close": vix, "vix_zscore": zscore}, index=dates)


# ---------------------------------------------------------------------------
# Tiered horizons
# ---------------------------------------------------------------------------

class TestTieredHorizons:
    def test_vix_20_gets_15day_horizon(self):
        """VIX 20 (< 25) should get 15 trading day horizon."""
        dates = pd.bdate_range("2020-01-01", periods=40)
        df = pd.DataFrame(
            {"vix_close": [20.0] * 40, "vix_zscore": [1.5] * 40}, index=dates
        )
        labels = generate_labels(df)
        eligible = labels[labels["eligible"]]
        assert len(eligible) > 0
        assert eligible["horizon"].iloc[0] == 15

    def test_vix_30_gets_30day_horizon(self):
        """VIX 30 (25-35) should get 30 trading day horizon."""
        dates = pd.bdate_range("2020-01-01", periods=60)
        df = pd.DataFrame(
            {"vix_close": [30.0] * 60, "vix_zscore": [2.0] * 60}, index=dates
        )
        labels = generate_labels(df)
        eligible = labels[labels["eligible"]]
        assert len(eligible) > 0
        assert eligible["horizon"].iloc[0] == 30

    def test_vix_40_gets_45day_horizon(self):
        """VIX 40 (>= 35) should get 45 trading day horizon."""
        dates = pd.bdate_range("2020-01-01", periods=80)
        df = pd.DataFrame(
            {"vix_close": [40.0] * 80, "vix_zscore": [2.0] * 80}, index=dates
        )
        labels = generate_labels(df)
        eligible = labels[labels["eligible"]]
        assert len(eligible) > 0
        assert eligible["horizon"].iloc[0] == 45

    def test_boundary_vix_25_gets_30day(self):
        """VIX exactly at 25 goes to the 25-35 tier (30 days)."""
        dates = pd.bdate_range("2020-01-01", periods=60)
        df = pd.DataFrame(
            {"vix_close": [25.0] * 60, "vix_zscore": [1.5] * 60}, index=dates
        )
        labels = generate_labels(df)
        eligible = labels[labels["eligible"]]
        assert len(eligible) > 0
        assert eligible["horizon"].iloc[0] == 30

    def test_boundary_vix_35_gets_45day(self):
        """VIX exactly at 35 goes to the 35+ tier (45 days)."""
        dates = pd.bdate_range("2020-01-01", periods=80)
        df = pd.DataFrame(
            {"vix_close": [35.0] * 80, "vix_zscore": [2.0] * 80}, index=dates
        )
        labels = generate_labels(df)
        eligible = labels[labels["eligible"]]
        assert len(eligible) > 0
        assert eligible["horizon"].iloc[0] == 45


# ---------------------------------------------------------------------------
# label_revert
# ---------------------------------------------------------------------------

class TestLabelRevert:
    def test_reversion_detected(self):
        """VIX dropping >= 15% should set label_revert = 1."""
        dates = pd.bdate_range("2020-01-01", periods=30)
        # VIX at 30, drops to 25 (16.7% drop) by day 1
        vix = [30.0] + [25.0] * 29
        df = pd.DataFrame(
            {"vix_close": vix, "vix_zscore": [2.0] * 30}, index=dates
        )
        labels = generate_labels(df)
        assert labels.loc[labels.index[0], "label_revert"] == 1

    def test_no_reversion(self):
        """VIX staying flat should set label_revert = 0."""
        dates = pd.bdate_range("2020-01-01", periods=30)
        vix = [30.0] * 30
        df = pd.DataFrame(
            {"vix_close": vix, "vix_zscore": [2.0] * 30}, index=dates
        )
        labels = generate_labels(df)
        assert labels.loc[labels.index[0], "label_revert"] == 0

    def test_small_drop_not_reversion(self):
        """VIX dropping only 10% should NOT set label_revert = 1."""
        dates = pd.bdate_range("2020-01-01", periods=30)
        vix = [30.0] + [27.0] * 29  # 10% drop
        df = pd.DataFrame(
            {"vix_close": vix, "vix_zscore": [2.0] * 30}, index=dates
        )
        labels = generate_labels(df)
        assert labels.loc[labels.index[0], "label_revert"] == 0

    def test_exactly_15pct_drop(self):
        """VIX dropping exactly 15% should set label_revert = 1."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        vix = [20.0] + [17.0] * 49  # 15% drop
        df = pd.DataFrame(
            {"vix_close": vix, "vix_zscore": [1.5] * 50}, index=dates
        )
        labels = generate_labels(df)
        assert labels.loc[labels.index[0], "label_revert"] == 1

    def test_magnitude_computed(self):
        """label_magnitude should reflect max % drop."""
        dates = pd.bdate_range("2020-01-01", periods=30)
        # VIX at 30, drops to 24 (20% drop) is the minimum in the window
        vix = [30.0, 28.0, 26.0, 24.0] + [25.0] * 26
        df = pd.DataFrame(
            {"vix_close": vix, "vix_zscore": [2.0] * 30}, index=dates
        )
        labels = generate_labels(df)
        # magnitude = -min_change = -(24-30)/30 = 0.2 = 20%
        assert labels.loc[labels.index[0], "label_magnitude"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# label_spike_first
# ---------------------------------------------------------------------------

class TestLabelSpikeFirst:
    def test_spike_before_reversion(self):
        """VIX rising 10%+ before reverting should set label_spike_first = 1."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        # VIX at 30, spikes to 34 (+13.3%), then drops to 25 (-16.7%)
        vix = [30.0, 34.0, 33.0, 32.0] + [25.0] * 46
        df = pd.DataFrame(
            {"vix_close": vix, "vix_zscore": [2.0] * 50}, index=dates
        )
        labels = generate_labels(df)
        assert labels.loc[labels.index[0], "label_spike_first"] == 1

    def test_no_spike_before_reversion(self):
        """VIX reverting directly should set label_spike_first = 0."""
        dates = pd.bdate_range("2020-01-01", periods=30)
        # VIX at 30, drops directly to 25 (no spike)
        vix = [30.0] + [25.0] * 29
        df = pd.DataFrame(
            {"vix_close": vix, "vix_zscore": [2.0] * 30}, index=dates
        )
        labels = generate_labels(df)
        assert labels.loc[labels.index[0], "label_spike_first"] == 0

    def test_spike_no_reversion(self):
        """VIX rises >= 10% but never reverts: label_spike_first = 1."""
        dates = pd.bdate_range("2020-01-01", periods=30)
        # VIX at 30, rises to 34 (>10%), stays elevated
        vix = [30.0, 34.0] + [35.0] * 28
        df = pd.DataFrame(
            {"vix_close": vix, "vix_zscore": [2.0] * 30}, index=dates
        )
        labels = generate_labels(df)
        assert labels.loc[labels.index[0], "label_spike_first"] == 1

    def test_no_spike_no_reversion(self):
        """VIX stays flat: both spike and revert should be 0."""
        dates = pd.bdate_range("2020-01-01", periods=30)
        vix = [30.0] * 30
        df = pd.DataFrame(
            {"vix_close": vix, "vix_zscore": [2.0] * 30}, index=dates
        )
        labels = generate_labels(df)
        assert labels.loc[labels.index[0], "label_spike_first"] == 0
        assert labels.loc[labels.index[0], "label_revert"] == 0


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_low_zscore_not_eligible(self):
        """Days with vix_zscore <= 1.0 should not be eligible."""
        dates = pd.bdate_range("2020-01-01", periods=30)
        df = pd.DataFrame(
            {"vix_close": [20.0] * 30, "vix_zscore": [0.5] * 30}, index=dates
        )
        labels = generate_labels(df)
        assert not labels["eligible"].any()

    def test_low_vix_not_eligible(self):
        """Days with VIX < 18 should not be eligible even with high z-score."""
        dates = pd.bdate_range("2020-01-01", periods=30)
        df = pd.DataFrame(
            {"vix_close": [15.0] * 30, "vix_zscore": [2.0] * 30}, index=dates
        )
        labels = generate_labels(df)
        assert not labels["eligible"].any()

    def test_eligible_when_both_conditions_met(self):
        """VIX >= 18 and zscore > 1.0 should be eligible."""
        dates = pd.bdate_range("2020-01-01", periods=30)
        df = pd.DataFrame(
            {"vix_close": [22.0] * 30, "vix_zscore": [1.5] * 30}, index=dates
        )
        labels = generate_labels(df)
        assert labels["eligible"].any()

    def test_nan_labels_for_non_eligible(self):
        """Non-eligible rows should have NaN labels."""
        dates = pd.bdate_range("2020-01-01", periods=30)
        df = pd.DataFrame(
            {"vix_close": [15.0] * 30, "vix_zscore": [0.5] * 30}, index=dates
        )
        labels = generate_labels(df)
        assert labels["label_revert"].isna().all()
        assert labels["label_spike_first"].isna().all()
        assert labels["label_magnitude"].isna().all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_end_of_data(self):
        """Last row should still get labels (possibly 0 if minimal forward data)."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame(
            {"vix_close": [30.0] * 5, "vix_zscore": [2.0] * 5}, index=dates
        )
        labels = generate_labels(df)
        # Last row has no forward data -> labels should still exist (0)
        last_eligible = labels[labels["eligible"]]
        assert len(last_eligible) > 0

    def test_all_nan_zscore(self):
        """All NaN z-scores should produce no eligible days."""
        dates = pd.bdate_range("2020-01-01", periods=10)
        df = pd.DataFrame(
            {"vix_close": [20.0] * 10, "vix_zscore": [np.nan] * 10}, index=dates
        )
        labels = generate_labels(df)
        assert not labels["eligible"].any()

    def test_single_row(self):
        """Single row should not crash."""
        dates = pd.bdate_range("2020-01-01", periods=1)
        df = pd.DataFrame(
            {"vix_close": [30.0], "vix_zscore": [2.0]}, index=dates
        )
        labels = generate_labels(df)
        assert len(labels) == 1

    def test_no_eligible_days(self):
        """When no days are eligible, labels should be all NaN."""
        dates = pd.bdate_range("2020-01-01", periods=20)
        df = pd.DataFrame(
            {"vix_close": [12.0] * 20, "vix_zscore": [0.3] * 20}, index=dates
        )
        labels = generate_labels(df)
        assert not labels["eligible"].any()
        assert labels["label_revert"].isna().all()

    def test_label_columns_exist(self, elevated_vix_df):
        """All expected label columns should be present."""
        labels = generate_labels(elevated_vix_df)
        for col in ["eligible", "horizon", "label_revert", "label_spike_first", "label_magnitude"]:
            assert col in labels.columns
