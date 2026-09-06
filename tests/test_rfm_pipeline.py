"""
test_rfm_pipeline.py
---------------------
Edge-case and failure-handling tests for the RFM pipeline.
Run with:  python -m pytest tests/ -v   (from the project root)

These map directly to the "Dependency, failure & edge-case handling" marks:
each test proves the pipeline degrades gracefully on a specific real-world
data problem instead of crashing or silently producing garbage.
"""
import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rfm_pipeline import (  # noqa: E402
    load_and_clean_transactions, compute_rfm, score_rfm, segment_customers,
    profile_segments, validate_segments, run_full_pipeline,
    RFMDataError, RFMComputationError,
)


def _base_tx(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", "2026-08-01", periods=n)
    return pd.DataFrame({
        "customer_id": [f"C{i % 40}" for i in range(n)],
        "order_date": dates.astype(str),
        "amount": rng.lognormal(3, 1, size=n).round(2),
    })


def test_empty_dataframe_raises():
    with pytest.raises(RFMDataError):
        load_and_clean_transactions(pd.DataFrame())


def test_missing_required_columns_raises():
    with pytest.raises(RFMDataError):
        load_and_clean_transactions(pd.DataFrame({"foo": [1, 2, 3]}))


def test_all_rows_invalid_raises_computation_error():
    bad = pd.DataFrame({
        "customer_id": [None, None],
        "order_date": [None, None],
        "amount": [None, None],
    })
    with pytest.raises(RFMComputationError):
        load_and_clean_transactions(bad)


def test_negative_and_zero_amounts_excluded_by_default():
    tx = _base_tx(20)
    tx.loc[0, "amount"] = -50.0
    tx.loc[1, "amount"] = 0.0
    clean, report = load_and_clean_transactions(tx)
    assert (clean["amount"] > 0).all()
    assert report["refund_rows_found"] == 1
    assert report["zero_amount_rows_found"] == 1


def test_future_dated_rows_dropped():
    tx = _base_tx(20)
    tx.loc[0, "order_date"] = "2099-01-01"
    clean, report = load_and_clean_transactions(tx)
    assert report["dropped_future_dated_rows"] == 1
    assert (clean["order_date"] <= pd.Timestamp.now()).all()


def test_duplicate_rows_deduplicated():
    tx = _base_tx(10)
    dup = pd.concat([tx, tx.iloc[[0, 1]]], ignore_index=True)
    clean, report = load_and_clean_transactions(dup)
    assert report["dropped_exact_duplicates"] == 2
    assert len(clean) <= len(tx)


def test_missing_values_dropped_not_crashed():
    tx = _base_tx(20)
    tx.loc[0, "customer_id"] = np.nan
    tx.loc[1, "amount"] = np.nan
    tx.loc[2, "order_date"] = np.nan
    clean, report = load_and_clean_transactions(tx)
    assert report["dropped_missing_required_fields"] == 3
    assert clean.isna().sum().sum() == 0


def test_single_order_customer_does_not_crash_scoring():
    """A customer base where almost everyone has exactly 1 order used to
    break naive pd.qcut(duplicates raising) -- must not crash."""
    tx = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(50)],
        "order_date": pd.date_range("2026-01-01", periods=50).astype(str),
        "amount": [25.0] * 50,  # identical amounts -> zero variance in monetary
    })
    clean, _ = load_and_clean_transactions(tx)
    rfm = compute_rfm(clean)
    scored = score_rfm(rfm)  # should not raise
    assert scored["m_score"].notna().all()
    assert scored["f_score"].notna().all()
    assert scored["f_score"].nunique() == 1  # everyone tied on frequency -> single, consistent tier


def test_ghost_customers_included_and_flagged():
    tx = _base_tx(20)
    all_ids = pd.Series([f"C{i}" for i in range(40)] + ["GHOST1", "GHOST2"])
    clean, _ = load_and_clean_transactions(tx)
    rfm = compute_rfm(clean, all_customer_ids=all_ids)
    scored = score_rfm(rfm)
    segmented = segment_customers(scored)
    ghost_rows = segmented[segmented["customer_id"].isin(["GHOST1", "GHOST2"])]
    assert (ghost_rows["segment"] == "No Purchase History").all()
    assert (ghost_rows["frequency"] == 0).all()


def test_skewed_monetary_uses_quantiles_not_equal_width():
    """One whale customer should NOT distort every other customer into the
    same bottom bin, which is what equal-width binning would do."""
    rng = np.random.default_rng(1)
    n = 300
    amounts = list(rng.uniform(10, 100, n - 1)) + [1_000_000]  # one huge outlier
    tx = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "order_date": pd.date_range("2025-01-01", periods=n).astype(str),
        "amount": amounts,
    })
    clean, _ = load_and_clean_transactions(tx)
    rfm = compute_rfm(clean)
    scored = score_rfm(rfm)
    # quantile scoring should still spread customers across multiple m_score tiers
    assert scored["m_score"].nunique() >= 4


def test_full_pipeline_runs_on_realistic_messy_data():
    tx = _base_tx(500, seed=3)
    # sprinkle in realistic messiness
    tx.loc[0:5, "customer_id"] = np.nan
    tx.loc[6:8, "amount"] = -10
    dup = pd.concat([tx, tx.iloc[[10, 11]]], ignore_index=True)
    result = run_full_pipeline(dup)
    assert len(result["profile"]) > 0  # produces a non-empty segment profile
    assert result["profile"]["customers"].sum() <= dup["customer_id"].nunique()
    assert (result["profile"]["pct_of_customers"] > 0).all()
    assert "recommended_action" in result["profile"].columns
    assert result["profile"]["recommended_action"].notna().all()


def test_validation_reports_stability_and_warnings():
    tx = _base_tx(600, seed=7)
    result = run_full_pipeline(tx, stability_window_days=60)
    val = result["validation"]
    assert isinstance(val.warnings, list)
    # stability_pct is either a float in [0,100] or None (if not enough history)
    if val.stability_pct is not None:
        assert 0 <= val.stability_pct <= 100


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
