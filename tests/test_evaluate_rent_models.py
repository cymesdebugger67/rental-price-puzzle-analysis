import json

import numpy as np
import pandas as pd
import pytest

from market_study import evaluate_rent_models as evaluation


def artificial_listings(n=180):
    values = np.arange(n)
    return pd.DataFrame({
        "id": values,
        "square_feet": 450 + (values % 41) * 30,
        "bedrooms": values % 4,
        "bathrooms": 1 + (values % 3),
        "category": np.where(values % 2 == 0, "apartment", "condo"),
        "state": np.where(values % 2 == 0, "AA", "BB"),
        "cityname": np.where(values % 3 == 0, "Alpha", "Beta"),
        "has_photo": np.where(values % 3 == 0, "Yes", "No"),
        "fee": "No",
        "pets_allowed": np.where(values % 3 == 0, "Cats", "Unknown"),
        "price": (800 + 1.45 * (450 + (values % 41) * 30) + 60 * (values % 4)).astype(float),
    })


def sample_residuals():
    return pd.DataFrame({
        "actual_rent": [1000., 2000., 4000., 5000., 1000., 1500.],
        "predicted_rent": [1100., 1600., 4500., 3500., 1200., 1600.],
        "absolute_error": [100., 400., 500., 1500., 200., 100.],
        "signed_error": [100., -400., 500., -1500., 200., 100.],
        "state": ["AA", "AA", "BB", "BB", "AA", "BB"],
        "bedrooms": [1, 2, 3, 4, 1, np.nan],
    })


def test_training_thresholds_depend_only_on_training_targets():
    train = artificial_listings(80)
    first = evaluation.training_thresholds(train)
    heldout = artificial_listings(30).copy()
    heldout["price"] *= 100
    assert evaluation.training_thresholds(train) == first
    assert first["p50"] < first["p95"] < first["p99"]


def test_selected_model_must_be_frozen_training_winner():
    train = artificial_listings(120)
    holdout = artificial_listings(30).copy()
    with pytest.raises(ValueError, match="must match frozen"):
        evaluation.evaluate_partition(train, holdout, "primary", "ridge_log_rent")


def test_real_selected_model_and_baseline_evaluate_without_target_leakage():
    train = artificial_listings(120)
    test = artificial_listings(31)
    metrics, prediction, thresholds = evaluation.evaluate_partition(train, test, "primary")
    assert set(metrics["model"]) == {"median_baseline", "random_forest_log_rent"}
    assert set(metrics["evaluation"]) == {"primary"}
    assert metrics["heldout_listings"].eq(len(test)).all()
    assert prediction.shape[0] == len(test)
    assert prediction["predicted_rent"].ge(0).all()
    assert np.isfinite(prediction["signed_error"]).all()
    assert all(np.isfinite(list(thresholds.values())))
    assert "id" not in prediction


def test_subgroup_reporting_suppresses_tiny_groups():
    records = sample_residuals()
    result = evaluation.subgroup_errors(records, "state", minimum=4)
    assert result.empty
    result = evaluation.subgroup_errors(records, "state", minimum=3)
    assert result["listings"].sum() == len(records)
    assert set(result["state"]) == {"AA", "BB"}


def test_bedroom_labels_keep_missing_as_unknown():
    assert evaluation.bedroom_labels([0, 1, 2, 5, np.nan]).tolist() == ["Studio", "1", "2", "5+", "Unknown"]


def test_price_grouping_thresholds_from_train():
    records = sample_residuals()
    thresholds = {"p50": 1500., "p95": 3000., "p99": 4500.}
    results = evaluation.price_segments(records, thresholds, minimum=1)
    assert results["listings"].sum() == len(records)
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluation.price_segments(records, {"p50": 1500., "p95": 1500., "p99": 4500.})


def test_tail_sensitivity_includes_full_heldout_population():
    records = sample_residuals()
    rows = evaluation.sensitivity_rows(records, {"p99": 3000.}, "primary")
    full = rows.set_index("sensitivity_group").loc["full_holdout"]
    assert full["heldout_listings"] == len(records)
    assert float(full["mae_usd"]) == pytest.approx(records["absolute_error"].mean())
    assert rows.loc[rows["sensitivity_group"] != "full_holdout", "heldout_listings"].sum() == len(records)


def test_residual_quantiles_are_monotonic_and_safely_aggregated():
    result = evaluation.summarize_residuals(sample_residuals(), "primary")
    assert result["absolute_error_p50_usd"] <= result["absolute_error_p95_usd"] <= result["absolute_error_p99_usd"]
    assert 0 <= result["overprediction_percent"] <= 100
    assert "id" not in result


def test_isolated_partition_rejects_shared_groups(monkeypatch):
    frame = artificial_listings(8)
    assignments = pd.DataFrame({
        "primary_partition": ["train"] * 4 + ["test"] * 4,
        "geographic_partition": ["train"] * 4 + ["test"] * 4,
        "property_group": ["same"] * 8,
        "state": ["AA"] * 4 + ["BB"] * 4,
    })
    monkeypatch.setattr(evaluation, "EXPECTED_TRAINING_ROWS", {"primary": 4, "geographic": 4})
    monkeypatch.setattr(evaluation, "EXPECTED_TEST_ROWS", {"primary": 4, "geographic": 4})
    with pytest.raises(ValueError, match="overlap"):
        evaluation.isolated_partition(frame, assignments, "primary")
    train, test = evaluation.isolated_partition(frame, assignments, "geographic")
    assert len(train) == len(test) == 4


def test_frozen_design_fails_if_missing_source(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.load_frozen_design(tmp_path)


def test_report_contains_true_computed_metrics_and_attribution(tmp_path):
    metrics = pd.DataFrame([
        {"evaluation": region, "model": model, "heldout_listings": 30, "mae_usd": mae, "rmse_usd": mae * 1.5, "r2": .2}
        for region in ("primary", "geographic")
        for model, mae in (("median_baseline", 500.), (evaluation.EXPECTED_SELECTED_MODEL, 300.))
    ])
    dest = tmp_path / "evaluation.md"
    selected = {"selected_mean_cv_mae_usd": 321.61}
    evaluation.export_report(dest, metrics, pd.DataFrame(), pd.DataFrame(), {"p99": 4200.}, {"p99": 4300.}, pd.DataFrame({"state": ["AA"]}), selected)
    report = dest.read_text(encoding="utf-8")
    assert "$300.00" in report
    assert "40.0%" in report
    assert "10.24432/C5X623" in report
    assert "Cyril James Lunzaga" in report


def test_charts_generate_without_per_listing_exports(tmp_path):
    residual = sample_residuals()
    evaluation.plot_actual_predicted(residual, tmp_path / "comparison.png")
    evaluation.plot_error_distribution(residual, tmp_path / "errors.png")
    evaluation.plot_state_errors(evaluation.subgroup_errors(residual, "state", minimum=3), tmp_path / "states.png", "Test chart")
    assert len(list(tmp_path.glob("*.png"))) == 3
