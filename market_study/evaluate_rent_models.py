from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from market_study.market_evidence import EXPECTED_SOURCE_SHA256, cohorts, load_listings, locate_source
from market_study.prepare_modeling_data import assignments_digest, make_partitions
from market_study.train_price_models import create_estimator, ensure_clean_features, score_regression

SPLIT_DIGEST = "eed0e59e56f4c457841a882f10af66d3828782b131f829b9e5c1627cfb3e135d"
EXPECTED_TRAINING_ROWS = {"primary": 79196, "geographic": 82247}
EXPECTED_TEST_ROWS = {"primary": 19905, "geographic": 16854}
EXPECTED_SELECTED_MODEL = "random_forest_log_rent"
MIN_SUBGROUP_SIZE = 30


def load_frozen_design(root):
    source = load_listings(locate_source(root))
    _, _, _, cohort = cohorts(source)
    assignments = make_partitions(cohort)
    digest = assignments_digest(assignments)
    if len(source) != 99492 or len(cohort) != 99101 or digest != SPLIT_DIGEST:
        raise ValueError("Source and partition fingerprint differ from frozen Phase 3")
    preparation_path = root / "outputs" / "market_research" / "preparation" / "preparation_manifest.json"
    decision_path = root / "outputs" / "market_research" / "training" / "model_selection.json"
    if not preparation_path.exists() or not decision_path.exists():
        raise ValueError("Preparation and training-only model decision must exist before evaluation")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    selection = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        preparation.get("source_sha256") != EXPECTED_SOURCE_SHA256
        or preparation.get("assignment_sha256") != digest
        or preparation.get("modeling_rows") != len(cohort)
        or preparation.get("primary_train_rows") != EXPECTED_TRAINING_ROWS["primary"]
        or preparation.get("primary_test_rows") != EXPECTED_TEST_ROWS["primary"]
        or preparation.get("geo_train_rows") != EXPECTED_TRAINING_ROWS["geographic"]
        or preparation.get("geo_test_rows") != EXPECTED_TEST_ROWS["geographic"]
        or preparation.get("primary_group_overlap") != 0
        or preparation.get("geo_state_overlap") != 0
    ):
        raise ValueError("Preparation manifest disagrees with frozen data")
    if (
        selection.get("data_sha256") != EXPECTED_SOURCE_SHA256
        or selection.get("assignment_sha256") != digest
        or selection.get("training_listings") != EXPECTED_TRAINING_ROWS["primary"]
        or selection.get("n_cv_folds") != 3
        or selection.get("holdout_score_computed") is not False
        or selection.get("selected_model") != EXPECTED_SELECTED_MODEL
        or selection.get("selection_metric") != "lowest_mean_cv_mae_usd"
        or selection.get("ridge_numeric_transformation") != "training_fold_quantile_uniform_128"
    ):
        raise ValueError("Frozen training-only model decision cannot be verified")
    return cohort, assignments, selection


def isolated_partition(cohort, assignments, strategy):
    field = {"primary": "primary_partition", "geographic": "geographic_partition"}.get(strategy)
    if field is None:
        raise ValueError("Unknown evaluation strategy")
    training_mask = assignments[field].eq("train")
    testing_mask = assignments[field].eq("test")
    if not (training_mask | testing_mask).all() or (training_mask & testing_mask).any():
        raise ValueError("Invalid evaluation partition labels")
    training = cohort.loc[training_mask].copy()
    testing = cohort.loc[testing_mask].copy()
    if len(training) != EXPECTED_TRAINING_ROWS[strategy] or len(testing) != EXPECTED_TEST_ROWS[strategy]:
        raise ValueError("Frozen partition size mismatch")
    key = "property_group" if strategy == "primary" else "state"
    fit_keys = set(assignments.loc[training_mask, key])
    test_keys = set(assignments.loc[testing_mask, key])
    if fit_keys & test_keys:
        raise ValueError("Training and held-out partitions overlap")
    if training["id"].isin(testing["id"]).any():
        raise ValueError("Listing ID overlap detected")
    return training, testing


def training_thresholds(training):
    prices = pd.to_numeric(training["price"], errors="raise").astype(float)
    if prices.empty or not prices.gt(0).all():
        raise ValueError("Training rent distribution is invalid")
    return {"p50": float(prices.quantile(.5)), "p95": float(prices.quantile(.95)), "p99": float(prices.quantile(.99))}


def evaluate_partition(training, testing, strategy, selected_model=EXPECTED_SELECTED_MODEL):
    if selected_model != EXPECTED_SELECTED_MODEL:
        raise ValueError("Evaluation model must match frozen training decision")
    train_x = ensure_clean_features(training)
    test_x = ensure_clean_features(testing)
    train_y = pd.to_numeric(training["price"], errors="raise").astype(float)
    test_y = pd.to_numeric(testing["price"], errors="raise").astype(float)
    if not train_y.gt(0).all() or not test_y.gt(0).all():
        raise ValueError("Positive monthly target constraint violated")
    baseline = create_estimator("median_baseline")
    selected = create_estimator(selected_model)
    baseline.fit(train_x, train_y)
    selected.fit(train_x, train_y)
    predictions = {
        "median_baseline": np.asarray(baseline.predict(test_x), dtype=float),
        selected_model: np.asarray(selected.predict(test_x), dtype=float),
    }
    metrics = []
    for name, guess in predictions.items():
        scores = score_regression(test_y, guess)
        metrics.append({
            "evaluation": strategy,
            "model": name,
            "training_listings": len(training),
            "heldout_listings": len(testing),
            **scores,
        })
    selected_predictions = np.maximum(predictions[selected_model], 0)
    outcomes = pd.DataFrame({
        "actual_rent": test_y.to_numpy(dtype=float),
        "predicted_rent": selected_predictions,
        "absolute_error": np.abs(test_y.to_numpy(dtype=float) - selected_predictions),
        "signed_error": selected_predictions - test_y.to_numpy(dtype=float),
        "state": testing["state"].astype(str).to_numpy(),
        "bedrooms": pd.to_numeric(testing["bedrooms"], errors="coerce").to_numpy(dtype=float),
    })
    return pd.DataFrame(metrics), outcomes, training_thresholds(training)


def subgroup_errors(outcomes, column, minimum=MIN_SUBGROUP_SIZE):
    if column not in outcomes or minimum < 1:
        raise ValueError("Invalid subgroup definition")
    rows = []
    for label, sample in outcomes.groupby(column, dropna=False, observed=True):
        if len(sample) < minimum:
            continue
        rows.append({
            column: str(label),
            "listings": len(sample),
            "mae_usd": float(sample["absolute_error"].mean()),
            "mean_signed_error_usd": float(sample["signed_error"].mean()),
            "median_absolute_error_usd": float(sample["absolute_error"].median()),
            "median_actual_rent_usd": float(sample["actual_rent"].median()),
        })
    if not rows:
        return pd.DataFrame(columns=[column, "listings", "mae_usd", "mean_signed_error_usd", "median_absolute_error_usd", "median_actual_rent_usd"])
    return pd.DataFrame(rows).sort_values(["listings", column], ascending=[False, True]).reset_index(drop=True)


def bedroom_labels(values):
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    labels = np.select(
        [numeric.eq(0), numeric.eq(1), numeric.eq(2), numeric.eq(3), numeric.eq(4), numeric.ge(5)],
        ["Studio", "1", "2", "3", "4", "5+"],
        default="Unknown",
    )
    return pd.Series(labels)


def price_segments(outcomes, thresholds, minimum=MIN_SUBGROUP_SIZE):
    boundaries = [0, thresholds["p50"], thresholds["p95"], thresholds["p99"], float("inf")]
    if sorted(boundaries) != boundaries or len(set(boundaries)) != 5:
        raise ValueError("Training-derived price thresholds must be strictly increasing")
    labeled = outcomes.copy()
    labeled["training_defined_price_band"] = pd.cut(
        labeled["actual_rent"],
        bins=boundaries,
        labels=["At or below train p50", "Train p50 to p95", "Train p95 to p99", "Above train p99"],
        include_lowest=True,
    )
    return subgroup_errors(labeled, "training_defined_price_band", minimum=minimum)


def summarize_residuals(outcomes, strategy):
    errors = outcomes["absolute_error"]
    signed = outcomes["signed_error"]
    return {
        "evaluation": strategy,
        "listings": int(len(outcomes)),
        "absolute_error_p50_usd": float(errors.quantile(.5)),
        "absolute_error_p90_usd": float(errors.quantile(.9)),
        "absolute_error_p95_usd": float(errors.quantile(.95)),
        "absolute_error_p99_usd": float(errors.quantile(.99)),
        "mean_signed_error_usd": float(signed.mean()),
        "overprediction_percent": float(100 * signed.gt(0).mean()),
    }


def sensitivity_rows(outcomes, thresholds, strategy):
    p99 = thresholds["p99"]
    masks = {
        "full_holdout": pd.Series(True, index=outcomes.index),
        "at_or_below_train_p99": outcomes["actual_rent"].le(p99),
        "above_train_p99": outcomes["actual_rent"].gt(p99),
    }
    rows = []
    for label, mask in masks.items():
        sample = outcomes.loc[mask]
        if sample.empty:
            continue
        rows.append({
            "evaluation": strategy,
            "sensitivity_group": label,
            "train_p99_usd": p99,
            "heldout_listings": int(len(sample)),
            "mae_usd": float(sample["absolute_error"].mean()),
            "median_absolute_error_usd": float(sample["absolute_error"].median()),
        })
    return pd.DataFrame(rows)


def plot_actual_predicted(outcomes, path):
    maximum = float(np.quantile(np.concatenate([
        outcomes["actual_rent"].to_numpy(), outcomes["predicted_rent"].to_numpy()
    ]), .99))
    sample = outcomes.loc[outcomes["actual_rent"].le(maximum) & outcomes["predicted_rent"].le(maximum)]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    mesh = ax.hexbin(sample["actual_rent"], sample["predicted_rent"], gridsize=45, bins="log", mincnt=1, cmap="Blues")
    ax.plot([0, maximum], [0, maximum], color="#b56d3c", linewidth=1.7, linestyle="--")
    fig.colorbar(mesh, ax=ax, label="Listings (log scale)")
    ax.set(xlim=(0, maximum), ylim=(0, maximum), xlabel="Observed monthly asking rent (USD)", ylabel="Predicted asking rent (USD)", title="Primary held-out listings: predicted versus observed")
    ax.text(.02, .98, "Display bounded at joint 99th percentile", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_error_distribution(outcomes, path):
    maximum = float(outcomes["absolute_error"].quantile(.99))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(outcomes.loc[outcomes["absolute_error"].le(maximum), "absolute_error"], bins=45, color="#335c75", edgecolor="white", linewidth=.3)
    ax.axvline(outcomes["absolute_error"].median(), color="#b56d3c", linewidth=2, label="Full-holdout median error")
    ax.set(xlabel="Absolute prediction error (USD per month)", ylabel="Held-out listings", title="Distribution of absolute prediction errors")
    ax.text(.98, .94, "Displayed through the 99th percentile", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_state_errors(states, path, title):
    shown = states.head(12).sort_values("mae_usd")
    if shown.empty:
        raise ValueError("Cannot chart an empty state error table")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(shown["state"].astype(str), shown["mae_usd"], color="#335c75")
    ax.set(xlabel="MAE (USD per month)", title=title)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_holdout_comparison(comparison, path):
    piv = comparison.pivot(index="evaluation", columns="model", values="mae_usd").reindex(["primary", "geographic"])
    x = np.arange(2)
    width = .35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, piv["median_baseline"], width, label="Median baseline", color="#9db4c2")
    ax.bar(x + width / 2, piv[EXPECTED_SELECTED_MODEL], width, label="Selected random forest", color="#335c75")
    ax.set(xticks=x, xticklabels=["Property-grouped", "Unseen-state"], ylabel="Held-out MAE (USD per month)", title="Regression evaluation: two independently fitted experiments")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export_report(path, comparison, residual, sensitivity, primary_thresholds, geographic_thresholds, unseen_states, selection):
    pivot = comparison.pivot(index="evaluation", columns="model", values="mae_usd")
    p_mae = float(pivot.loc["primary", EXPECTED_SELECTED_MODEL])
    p_baseline = float(pivot.loc["primary", "median_baseline"])
    g_mae = float(pivot.loc["geographic", EXPECTED_SELECTED_MODEL])
    g_baseline = float(pivot.loc["geographic", "median_baseline"])
    p_gain = 100 * (p_baseline - p_mae) / p_baseline
    g_gain = 100 * (g_baseline - g_mae) / g_baseline
    geographic_mae_comparison = f"**{abs(g_gain):.1f}% {'lower' if g_gain >= 0 else 'higher'} MAE**"
    p_rmse = float(comparison.loc[comparison["evaluation"].eq("primary") & comparison["model"].eq(EXPECTED_SELECTED_MODEL), "rmse_usd"].iloc[0])
    p_r2 = float(comparison.loc[comparison["evaluation"].eq("primary") & comparison["model"].eq(EXPECTED_SELECTED_MODEL), "r2"].iloc[0])
    g_rmse = float(comparison.loc[comparison["evaluation"].eq("geographic") & comparison["model"].eq(EXPECTED_SELECTED_MODEL), "rmse_usd"].iloc[0])
    g_r2 = float(comparison.loc[comparison["evaluation"].eq("geographic") & comparison["model"].eq(EXPECTED_SELECTED_MODEL), "r2"].iloc[0])
    all_stats = comparison.sort_values(["evaluation", "model"])
    metrics_table = "\n".join(["| Experiment | Model | Listings | MAE | RMSE | R² |", "|---|---|---:|---:|---:|---:|"] + [
        f"| {row.evaluation} | {row.model.replace('_', ' ')} | {row.heldout_listings:,} | ${row.mae_usd:,.2f} | ${row.rmse_usd:,.2f} | {row.r2:.3f} |" for row in all_stats.itertuples(index=False)
    ])
    text = f"""# Held-out rental price evaluation

## Final model decision

Random Forest regression with a log1p monthly asking-rent target was selected **before** either held-out evaluation, by the lowest mean property-grouped training cross-validation MAE (${selection['selected_mean_cv_mae_usd']:,.2f}). The training-only model decision is preserved in `training/model_selection.json`. This evaluation does **not** retune features, model parameters, partitions, or price thresholds using the final tests.

## Results

{metrics_table}

The primary property-grouped experiment refits the selected model on 79,196 listings and evaluates it on 19,905 listings with no shared property-group fingerprints. The selected model's test MAE is ${p_mae:,.2f} versus ${p_baseline:,.2f} for its own training-median baseline, a **{p_gain:.1f}%** relative reduction; RMSE is ${p_rmse:,.2f}, and R² is {p_r2:.3f}. These are historical advertised asking-rent prediction errors, not actual leasing errors.

For geographic generalization, the same frozen model specification is independently refitted on 82,247 records from 43 states and scored on 16,854 listings from eight withheld states. Its MAE is ${g_mae:,.2f} versus ${g_baseline:,.2f} for the geographic training-median baseline, {geographic_mae_comparison}; RMSE is ${g_rmse:,.2f}, and R² is {g_r2:.3f}. Different training/test samples mean this experiment is a stress test, not a strictly controlled paired comparison of the two held-out metrics.

## Diagnostic interpretation

Primary-training p99 rent threshold: **${primary_thresholds['p99']:,.2f}**. Geographic-training p99 threshold: **${geographic_thresholds['p99']:,.2f}**. Both are determined using only the corresponding training partitions. The primary reported MAE, RMSE and R² retain **all** held-out listings, including high-rent properties. The separate tail table reports conditional errors above and below each training-defined p99; it does not present a replacement main metric.

Residual and subgroup tables include only aggregates, without individual listing predictions or addresses. State- and bedroom-group summaries suppress groups with fewer than {MIN_SUBGROUP_SIZE} held-out records. Prices used to label diagnostic bands are **observed test targets**, for evaluation after selection only, not predictors or model-selection information. Missing pet-policy and amenity information cannot be interpreted as absence of those features. No causal price premiums are inferred.

The unseen-state table contains {len(unseen_states)} state groups meeting the minimum reporting threshold. Its state-level error variation reflects both true market heterogeneity and the source's unequal, nonrepresentative listing coverage. The geographic holdout does not establish a national generalization bound.

## Limitations and provenance

The source is the UCI *Apartment for Rent Classified* dataset (historical US classified records from 2018–2019), DOI [10.24432/C5X623](https://doi.org/10.24432/C5X623), CC BY 4.0. The CSV SHA-256 is `{EXPECTED_SOURCE_SHA256}`. These are advertised prices rather than signed rental agreements; sample selection, missing fields, reposts, and property-specification collisions limit interpretation. The random split groups by state, city, category, size, room counts and coordinates but cannot establish unique physical properties. The state holdout is one deterministic partition rather than a full geographic cross-validation study.

The analytical project is authored by **Cyril James Lunzaga**. Numerical results in this report are generated from the local source data, not manually assumed.
"""
    path.write_text(text, encoding="utf-8")


def run(root):
    cohort, assignments, selection = load_frozen_design(root)
    outputs = {}
    metrics_frames = []
    residuals = []
    sensitivities = []
    for strategy in ("primary", "geographic"):
        training, testing = isolated_partition(cohort, assignments, strategy)
        metrics, outcomes, thresholds = evaluate_partition(training, testing, strategy, selection["selected_model"])
        outputs[strategy] = {"metrics": metrics, "outcomes": outcomes, "thresholds": thresholds, "training": len(training), "testing": len(testing)}
        metrics_frames.append(metrics)
        residuals.append(summarize_residuals(outcomes, strategy))
        sensitivities.append(sensitivity_rows(outcomes, thresholds, strategy))
    comparison = pd.concat(metrics_frames, ignore_index=True)
    p = outputs["primary"]["outcomes"]
    g = outputs["geographic"]["outcomes"]
    primary_states = subgroup_errors(p, "state")
    primary_bedrooms = p.assign(bedroom_group=bedroom_labels(p["bedrooms"]).to_numpy())
    bedroom_table = subgroup_errors(primary_bedrooms, "bedroom_group")
    price_table = price_segments(p, outputs["primary"]["thresholds"])
    geographic_states = subgroup_errors(g, "state")
    if not len(geographic_states) or not len(primary_states):
        raise ValueError("State error summary unexpectedly empty")
    summary = pd.DataFrame(residuals)
    sensitivity = pd.concat(sensitivities, ignore_index=True)
    path = root / "outputs" / "market_research" / "evaluation"
    path.mkdir(parents=True, exist_ok=True)
    tables = {
        "heldout_model_comparison.csv": comparison,
        "primary_state_errors.csv": primary_states,
        "primary_bedroom_errors.csv": bedroom_table,
        "primary_training_defined_price_bands.csv": price_table,
        "unseen_state_errors.csv": geographic_states,
        "absolute_error_distribution.csv": summary,
        "training_defined_tail_sensitivity.csv": sensitivity,
    }
    for name, table in tables.items():
        table.to_csv(path / name, index=False, float_format="%.6g")
    figures = {
        "primary_actual_vs_predicted.png": lambda f: plot_actual_predicted(p, f),
        "primary_absolute_error_distribution.png": lambda f: plot_error_distribution(p, f),
        "primary_state_mae.png": lambda f: plot_state_errors(primary_states, f, "Primary holdout: largest state samples"),
        "unseen_state_mae.png": lambda f: plot_state_errors(geographic_states, f, "Unseen-state holdout: error by state"),
        "holdout_mae_comparison.png": lambda f: plot_holdout_comparison(comparison, f),
    }
    for name, chart in figures.items():
        chart(path / name)
    export_report(path / "heldout_evaluation.md", comparison, summary, sensitivity, outputs["primary"]["thresholds"], outputs["geographic"]["thresholds"], geographic_states, selection)
    comparison_rows = comparison.to_dict(orient="records")
    manifest = {
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "assignment_sha256": SPLIT_DIGEST,
        "selection_source": "training_only_grouped_cross_validation",
        "selected_model": selection["selected_model"],
        "primary_train_rows": outputs["primary"]["training"],
        "primary_test_rows": outputs["primary"]["testing"],
        "geo_train_rows": outputs["geographic"]["training"],
        "geo_test_rows": outputs["geographic"]["testing"],
        "primary_group_overlap": 0,
        "geo_state_overlap": 0,
        "full_holdout_scoring": True,
        "training_only_thresholds": {k: v["thresholds"] for k, v in outputs.items()},
        "comparisons": comparison_rows,
        "tables": list(tables),
        "figures": list(figures),
        "no_listing_level_exports": True,
    }
    (path / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    report = run(Path.cwd())
    print("PASS - Selected model evaluated without retuning")
    print("PASS - Frozen source and split fingerprints verified")
    print("PASS - Primary holdout:", report["primary_test_rows"], "listings")
    print("PASS - Unseen-state holdout:", report["geo_test_rows"], "listings")
    for row in report["comparisons"]:
        print(row["evaluation"], row["model"], "MAE", round(row["mae_usd"], 2), "RMSE", round(row["rmse_usd"], 2), "R2", round(row["r2"], 3))
    print("PASS - Seven aggregate tables, five charts, narrative evaluation and manifest generated")


if __name__ == "__main__":
    main()
