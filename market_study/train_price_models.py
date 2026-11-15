from pathlib import Path
from datetime import datetime, timezone
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from market_study.model_readiness import MODEL_FEATURES
from market_study.prepare_modeling_data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FORBIDDEN_MODEL_FIELDS
from market_study.statistical_evidence import primary_training

FOLDS = 3
TRAINING_SEED = 2026
CANDIDATES = ("median_baseline", "ridge_log_rent", "random_forest_log_rent")


def ensure_clean_features(data):
    forbidden = set(MODEL_FEATURES) & set(FORBIDDEN_MODEL_FIELDS)
    if forbidden or "price" in MODEL_FEATURES:
        raise ValueError(f"Leaking predictors detected: {sorted(forbidden)}")
    absent = set(MODEL_FEATURES) - set(data.columns)
    if absent:
        raise ValueError(f"Missing model features: {sorted(absent)}")
    frame = pd.DataFrame(index=data.index)
    for col in NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(data[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        strings = data[col].astype("string").str.strip().fillna("Unknown")
        frame[col] = strings.mask(strings.eq(""), "Unknown").astype(str)
    if set(frame.columns) != set(MODEL_FEATURES):
        raise ValueError("Model input columns diverge from approved feature policy")
    return frame.loc[:, list(MODEL_FEATURES)]


def preprocess(numeric_strategy="standard"):
    if numeric_strategy not in ("standard", "bounded_rank"):
        raise ValueError(f"Unrecognized numeric preprocessing: {numeric_strategy}")
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if numeric_strategy == "bounded_rank":
        numeric_steps.append(("rank", QuantileTransformer(
            n_quantiles=128,
            output_distribution="uniform",
            random_state=TRAINING_SEED,
        )))
    else:
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = OneHotEncoder(handle_unknown="ignore", min_frequency=25, max_categories=35, sparse_output=True)
    return ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ], sparse_threshold=.3)


def create_estimator(name):
    if name == "median_baseline":
        return DummyRegressor(strategy="median")
    if name == "ridge_log_rent":
        regression = Ridge(alpha=30.0)
        numeric_strategy = "bounded_rank"
    elif name == "random_forest_log_rent":
        numeric_strategy = "standard"
        regression = RandomForestRegressor(
            n_estimators=48,
            max_depth=16,
            min_samples_leaf=6,
            max_features=.7,
            random_state=TRAINING_SEED,
            n_jobs=2,
        )
    else:
        raise ValueError(f"Unrecognized model: {name}")
    return TransformedTargetRegressor(
        regressor=Pipeline([("features", preprocess(numeric_strategy)), ("regression", regression)]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def group_cv_splits(groups, folds=FOLDS):
    if folds < 2 or pd.Series(groups).nunique() < folds:
        raise ValueError("Not enough distinct property groups for grouped cross-validation")
    position = np.arange(len(groups))
    group_values = pd.Series(groups).astype(str).to_numpy()
    pieces = []
    for fit_idx, valid_idx in GroupKFold(n_splits=folds).split(position, groups=group_values):
        if set(group_values[fit_idx]) & set(group_values[valid_idx]):
            raise ValueError("Property group crosses cross-validation fold")
        pieces.append((fit_idx, valid_idx))
    return pieces


def score_regression(y_true, predictions):
    truth = np.asarray(y_true, dtype=float)
    prediction = np.maximum(np.asarray(predictions, dtype=float), 0)
    if not np.isfinite(prediction).all() or len(truth) != len(prediction):
        raise ValueError("Non-finite predictions or mismatched prediction shape")
    return {
        "mae_usd": float(mean_absolute_error(truth, prediction)),
        "rmse_usd": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
    }


def evaluate_training_candidates(x, y, groups, candidates=CANDIDATES, folds=FOLDS):
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError("Inconsistent training sample sizes")
    if not pd.Series(y).gt(0).all():
        raise ValueError("Price targets must be positive")
    splits = group_cv_splits(groups, folds)
    records = []
    fold_sizes = []
    for fold_num, (fit_idx, valid_idx) in enumerate(splits, start=1):
        fold_sizes.append({"fold": fold_num, "training_listings": len(fit_idx), "validation_listings": len(valid_idx), "shared_property_groups": 0})
        for candidate in candidates:
            model = create_estimator(candidate)
            model.fit(x.iloc[fit_idx], y.iloc[fit_idx])
            predicted = model.predict(x.iloc[valid_idx])
            metrics = score_regression(y.iloc[valid_idx], predicted)
            records.append({"model": candidate, "fold": fold_num, "training_listings": len(fit_idx), "validation_listings": len(valid_idx), "max_prediction_usd": float(np.max(predicted)), "predictions_over_100k": int(np.sum(np.asarray(predicted) > 100000)), **metrics})
    return pd.DataFrame(records), pd.DataFrame(fold_sizes)


def aggregate_scores(scores):
    summary = scores.groupby("model", observed=True).agg(
        folds=("fold", "nunique"),
        mean_cv_mae_usd=("mae_usd", "mean"),
        sd_cv_mae_usd=("mae_usd", "std"),
        mean_cv_rmse_usd=("rmse_usd", "mean"),
        mean_cv_r2=("r2", "mean"),
        max_cv_prediction_usd=("max_prediction_usd", "max"),
        predictions_over_100k=("predictions_over_100k", "sum"),
    ).reset_index()
    if set(summary["model"]) != set(CANDIDATES):
        raise ValueError("Candidate comparison is incomplete")
    if summary["folds"].nunique() != 1 or summary["folds"].iloc[0] != FOLDS:
        raise ValueError("Cross-validation fold coverage is incomplete")
    if not np.isfinite(summary[["mean_cv_mae_usd", "mean_cv_rmse_usd", "mean_cv_r2", "max_cv_prediction_usd"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite aggregate validation metric")
    return summary.sort_values(["mean_cv_mae_usd", "model"]).reset_index(drop=True)


def draw_cv_comparison(summary, path):
    data = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(data["model"].str.replace("_", " "), data["mean_cv_mae_usd"], color="#335c75")
    ax.set(title="Training-only grouped cross-validation", xlabel="Mean absolute error (USD per month; lower is better)")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(root):
    frame, groups, signature = primary_training(root)
    x = ensure_clean_features(frame)
    y = pd.to_numeric(frame["price"], errors="raise").astype(float)
    scores, fold_sizes = evaluate_training_candidates(x, y, groups)
    summary = aggregate_scores(scores)
    selected = str(summary.iloc[0]["model"])
    if selected not in CANDIDATES:
        raise ValueError("Selected estimator is not permitted")
    directory = root / "outputs" / "market_research" / "training"
    directory.mkdir(parents=True, exist_ok=True)
    reports = {
        "grouped_cv_fold_scores.csv": scores,
        "grouped_cv_model_comparison.csv": summary,
        "grouped_cv_sample_sizes.csv": fold_sizes,
    }
    for name, table in reports.items():
        table.to_csv(directory / name, index=False, float_format="%.6g")
    draw_cv_comparison(summary, directory / "grouped_cv_mae_comparison.png")
    baseline_mae = float(summary.loc[summary["model"].eq("median_baseline"), "mean_cv_mae_usd"].iloc[0])
    selected_mae = float(summary.iloc[0]["mean_cv_mae_usd"])
    ridge_mae = float(summary.loc[summary["model"].eq("ridge_log_rent"), "mean_cv_mae_usd"].iloc[0])
    if ridge_mae > baseline_mae * 10:
        raise ValueError("Ridge remains numerically unstable relative to the training baseline")
    decision = {
        "scope": "training_only_grouped_cross_validation",
        "data_sha256": "6adeae31e5507e39feb03acf811547a5331d2329c5a24ed1b1f1455fc1835916",
        "assignment_sha256": signature,
        "training_listings": int(len(frame)),
        "training_property_groups": int(groups.nunique()),
        "n_cv_folds": FOLDS,
        "model_candidates": list(CANDIDATES),
        "selection_metric": "lowest_mean_cv_mae_usd",
        "selected_model": selected,
        "selected_mean_cv_mae_usd": selected_mae,
        "median_baseline_mean_cv_mae_usd": baseline_mae,
        "ridge_numeric_transformation": "training_fold_quantile_uniform_128",
        "ridge_instability_review": "revised_before_holdout_evaluation",
        "candidate_max_cv_predictions_usd": {
            str(row["model"]): float(row["max_cv_prediction_usd"])
            for row in summary.to_dict(orient="records")
        },
        "random_seed": TRAINING_SEED,
        "holdout_score_computed": False,
        "source": "UCI Apartment for Rent Classified, DOI 10.24432/C5X623, CC BY 4.0",
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    (directory / "model_selection.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    note = f"""# Regression modeling: training-only comparison

This experiment compares models using only the **{len(frame):,}** training observations from the frozen property-specification grouped split. Source: UCI Apartment for Rent Classified (2018–2019), DOI https://doi.org/10.24432/C5X623, CC BY 4.0. Model inputs are floor area, bedrooms, bathrooms, listing category, state, city, photo indicator, fee indicator, and pet-policy text; `price_display`, price-per-square-foot, IDs, addresses, listing descriptions, and timestamps are excluded.

## Training design

- The primary holdout of **19,905** listings has not been used to fit, tune, select, or score models in this phase.
- Each of the **{FOLDS}** grouped validation folds isolates exact property-specification groups between its training and validation sides. Similar units can still share nonidentical specifications.
- Missing numeric predictors are imputed and categorical values are one-hot encoded **inside each training fold**. Unknown category labels in validation are ignored; missing categorical details are represented as Unknown, not treated as a negative answer.
- The baseline predicts each fold's training median asking rent. Ridge regression and Random Forest predict log1p(price) internally and convert predictions back to monthly USD. Price-derived features are excluded.
- The initial training-only Ridge diagnostic suffered severe extrapolation under unbounded standardized numeric inputs followed by inverse exponential target transformation. Before examining any final test outcomes, Ridge numeric inputs were changed to a bounded rank-quantile mapping fitted separately within each CV training fold. This is a documented training-stage model repair, not a change to the held-out partitions or target records. The Random Forest preprocessing and algorithm configuration were not changed.
- Model selection uses the lowest mean grouped cross-validation **MAE**, without selecting on the held-out test data. RMSE and R² are supplementary diagnostics.

## Training outcome

- Selected algorithm: **{selected.replace('_', ' ')}**.
- Selected training-only mean grouped-CV MAE: **${selected_mae:,.2f}** per month.
- Median-baseline training-only mean grouped-CV MAE: **${baseline_mae:,.2f}** per month.

These are cross-validated training results, **not** out-of-sample scores on the final holdout. The next evaluation phase will refit the selected specification on the complete primary training split, then report errors on the untouched primary holdout and separately refit the same specification for the unseen-state stress test. Geographic and high-rent subgroup diagnostics will accompany aggregate scores.

This retrospective classifieds study is not a deployment-ready rent estimator. It neither measures actual executed rents nor establishes causal rental premiums.
"""
    (directory / "model_development.md").write_text(note, encoding="utf-8")
    return decision


def main():
    result = run(Path.cwd())
    print("PASS - Grouped training-only model comparison completed")
    print("CV training records:", result["training_listings"])
    print("Model selected by training-only MAE:", result["selected_model"])
    print("Mean CV MAE (USD):", round(result["selected_mean_cv_mae_usd"], 2))
    print("PASS - Three tables, one figure, and model decision report generated")
    print("PASS - Primary holdout never scored in this stage")


if __name__ == "__main__":
    main()
