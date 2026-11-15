import pandas as pd
import pytest

from market_study.publish_portfolio import compact_score, generate_readme, metric_table
from market_study.verify_portfolio import verify


def fixture_comparison():
    return pd.DataFrame([
        {"evaluation": "primary", "model": "median_baseline", "heldout_listings": 19905, "mae_usd": 520.317, "rmse_usd": 970.163, "r2": -0.0344},
        {"evaluation": "primary", "model": "random_forest_log_rent", "heldout_listings": 19905, "mae_usd": 312.371, "rmse_usd": 628.83, "r2": 0.5654},
        {"evaluation": "geographic", "model": "median_baseline", "heldout_listings": 16854, "mae_usd": 911.144, "rmse_usd": 1674.89, "r2": -0.3327},
        {"evaluation": "geographic", "model": "random_forest_log_rent", "heldout_listings": 16854, "mae_usd": 1036.3, "rmse_usd": 1644.82, "r2": -0.2852},
    ])


def fixture_cv():
    return pd.DataFrame([
        {"model": "random_forest_log_rent", "mean_cv_mae_usd": 321.611, "mean_cv_r2": 0.5143},
        {"model": "ridge_log_rent", "mean_cv_mae_usd": 336.861, "mean_cv_r2": 0.4137},
        {"model": "median_baseline", "mean_cv_mae_usd": 523.975, "mean_cv_r2": -0.0384},
    ])


def test_compact_score_finds_exact_experiment_model():
    row = compact_score(fixture_comparison(), "primary", "random_forest_log_rent")
    assert row["mae_usd"] == pytest.approx(312.371)


def test_compact_score_rejects_missing_row():
    with pytest.raises(ValueError):
        compact_score(fixture_comparison(), "unspecified", "random_forest_log_rent")


def test_metric_table_includes_both_test_designs():
    result = metric_table(fixture_comparison())
    assert "$312.37" in result
    assert "$1,036.30" in result
    assert "Unseen-state" in result


def test_readme_tells_unfavorable_geographic_result():
    text = generate_readme({}, {"selected_mean_cv_mae_usd": 321.611}, {}, fixture_comparison(), fixture_cv())
    assert "13.7% higher" in text
    assert "40.0% reduction" in text
    assert "Cyril James Lunzaga" in text
    assert "CC BY 4.0" not in text or "Attribution" in text


def test_readme_does_not_claim_global_geographic_success():
    text = generate_readme({}, {"selected_mean_cv_mae_usd": 321.611}, {}, fixture_comparison(), fixture_cv())
    assert "must **not** be generalized" in text
    assert "historical" in text.lower()


def test_release_checker_rejects_missing_evidence(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(tmp_path)
