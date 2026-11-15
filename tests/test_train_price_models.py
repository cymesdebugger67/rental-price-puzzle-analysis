import numpy as np
import pandas as pd
import pytest

from market_study.model_readiness import MODEL_FEATURES
from market_study.train_price_models import (
    CANDIDATES, aggregate_scores, create_estimator,
    ensure_clean_features, evaluate_training_candidates, group_cv_splits,
    score_regression, preprocess
)


def sample(n=180):
    return pd.DataFrame({
        "square_feet": [400 + (i % 35) * 35 for i in range(n)],
        "bedrooms": [i % 4 for i in range(n)],
        "bathrooms": [1 if i % 2 else 2 for i in range(n)],
        "category": ["apartment" if i % 2 else "studio" for i in range(n)],
        "state": [f"S{i % 12}" for i in range(n)],
        "cityname": [f"City{i % 25}" for i in range(n)],
        "has_photo": ["yes" if i % 3 else "no" for i in range(n)],
        "fee": ["yes" if i % 7 else "no" for i in range(n)],
        "pets_allowed": [None if i % 4 else "cats" for i in range(n)],
        "price": [850 + 25 * (i % 35) + 65 * (i % 4) for i in range(n)],
        "price_display": ["$1000"] * n,
        "id": list(range(n)),
    })


def test_feature_matrix_excludes_outcome_and_private_identifiers():
    source = sample()
    frame = ensure_clean_features(source)
    assert set(frame.columns) == set(MODEL_FEATURES)
    assert "id" not in frame and "price" not in frame and "price_display" not in frame
    assert frame.loc[1, "pets_allowed"] == "Unknown"


def test_missing_candidate_feature_raises():
    with pytest.raises(ValueError, match="Missing model features"):
        ensure_clean_features(sample().drop(columns="fee"))


def test_group_crossvalidation_does_not_mix_groups():
    groups = pd.Series([f"g{i // 3}" for i in range(180)])
    pieces = group_cv_splits(groups, 3)
    assert len(pieces) == 3
    for train_idx, val_idx in pieces:
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[val_idx]))
        assert len(set(train_idx) & set(val_idx)) == 0


def test_too_few_groups_are_rejected():
    with pytest.raises(ValueError, match="Not enough distinct"):
        group_cv_splits(["same"] * 30)


def test_regression_metrics_are_in_usd():
    results = score_regression([100, 200, 300], [100, 150, 300])
    assert results["mae_usd"] == pytest.approx(50 / 3)
    assert results["rmse_usd"] == pytest.approx((2500 / 3) ** .5)
    with pytest.raises(ValueError, match="Non-finite"):
        score_regression([1, 2], [np.nan, 2])


def test_estimator_registry_rejects_unknown():
    with pytest.raises(ValueError, match="Unrecognized"):
        create_estimator("unknown")
    assert all(create_estimator(name) is not None for name in CANDIDATES)


def test_cv_routine_trains_without_heldout_dataset():
    data = sample()
    x = ensure_clean_features(data)
    y = data["price"].astype(float)
    groups = pd.Series([f"g{i // 3}" for i in range(len(data))])
    scores, sizes = evaluate_training_candidates(x, y, groups)
    assert len(scores) == len(CANDIDATES) * 3
    assert scores["mae_usd"].ge(0).all()
    assert len(sizes) == 3
    assert sizes["shared_property_groups"].eq(0).all()
    summary = aggregate_scores(scores)
    assert set(summary["model"]) == set(CANDIDATES)
    assert summary.iloc[0]["mean_cv_mae_usd"] <= summary.iloc[-1]["mean_cv_mae_usd"]


def test_summary_refuses_incomplete_candidate_results():
    missing = pd.DataFrame({
        "model": ["median_baseline"] * 3,
        "fold": [1, 2, 3],
        "mae_usd": [100, 150, 120],
        "rmse_usd": [150, 200, 160],
        "r2": [0.1, 0.2, 0.3],
        "max_prediction_usd": [1200, 1300, 1400],
        "predictions_over_100k": [0, 0, 0],
    })
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_scores(missing)


def test_cv_fails_on_nonpositive_targets():
    frame = sample()
    x = ensure_clean_features(frame)
    y = frame["price"].astype(float)
    y.iloc[0] = -1
    groups = [f"g{i // 3}" for i in range(len(frame))]
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_training_candidates(x, y, groups)


def test_ridge_uses_bounded_rank_inputs_but_random_forest_unchanged():
    ridge = create_estimator("ridge_log_rent").regressor.named_steps["features"]
    forest = create_estimator("random_forest_log_rent").regressor.named_steps["features"]
    assert "rank" in ridge.transformers[0][1].named_steps
    assert "scale" in forest.transformers[0][1].named_steps
    assert "rank" not in forest.transformers[0][1].named_steps


def test_ridge_prediction_is_stable_on_extreme_area_outside_fit_range():
    train = sample(n=360)
    x_train = ensure_clean_features(train)
    model = create_estimator("ridge_log_rent")
    model.fit(x_train, train["price"].astype(float))
    normal = train.iloc[[0]].copy()
    extreme = train.iloc[[0]].copy()
    extreme.loc[:, "square_feet"] = 50000
    extreme.loc[:, "bedrooms"] = 80
    extreme.loc[:, "bathrooms"] = 50
    predictions = model.predict(ensure_clean_features(pd.concat([normal, extreme], ignore_index=True)))
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
    assert predictions.max() < 100000
    assert predictions[1] >= predictions[0]


def test_bounded_transform_uses_training_only_distribution():
    model = preprocess("bounded_rank")
    train = ensure_clean_features(sample(200))
    model.fit(train)
    ranker = model.named_transformers_["numeric"].named_steps["rank"]
    cutpoints = ranker.quantiles_.copy()
    extreme = train.iloc[[0]].copy()
    extreme.loc[:, "square_feet"] = 50000
    model.transform(extreme)
    np.testing.assert_array_equal(ranker.quantiles_, cutpoints)
    with pytest.raises(ValueError, match="Unrecognized numeric"):
        preprocess("arbitrary")
