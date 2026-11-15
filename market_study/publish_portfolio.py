from hashlib import sha256
from pathlib import Path
from shutil import copyfile
import json
import re

import pandas as pd

from market_study.market_evidence import EXPECTED_SOURCE_SHA256, locate_source, file_digest
from market_study.evaluate_rent_models import SPLIT_DIGEST, EXPECTED_SELECTED_MODEL

AUTHOR = "Cyril James Lunzaga"
UCI_CITATION = "Apartment for Rent Classified [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5X623."
DATA_URL = "https://archive.ics.uci.edu/dataset/555/apartment"
BASE = Path("outputs/market_research")
FIGURES = {
    "asking_rent_distribution.png": "figures/asking_rent_distribution.png",
    "rent_vs_floor_area.png": "figures/rent_vs_floor_area.png",
    "state_rent_comparison.png": "figures/state_rent_comparison.png",
    "bedroom_rent_comparison.png": "figures/bedroom_rent_comparison.png",
    "state_sample_concentration.png": "readiness/state_sample_concentration.png",
    "rent_area_tail_diagnostics.png": "readiness/rent_area_tail_diagnostics.png",
    "bedroom_median_intervals.png": "statistics/bedroom_median_intervals.png",
    "grouped_cv_mae_comparison.png": "training/grouped_cv_mae_comparison.png",
    "holdout_mae_comparison.png": "evaluation/holdout_mae_comparison.png",
    "primary_actual_vs_predicted.png": "evaluation/primary_actual_vs_predicted.png",
    "primary_absolute_error_distribution.png": "evaluation/primary_absolute_error_distribution.png",
    "unseen_state_mae.png": "evaluation/unseen_state_mae.png",
}
TABLES = {
    "cohort_flow.csv": "tables/cohort_flow.csv",
    "field_missingness.csv": "tables/field_missingness.csv",
    "rent_quantiles.csv": "tables/rent_quantiles.csv",
    "state_rent_summary.csv": "tables/state_rent_summary.csv",
    "city_rent_summary.csv": "tables/city_rent_summary.csv",
    "split_summary.csv": "preparation/split_summary.csv",
    "bedroom_median_contrasts.csv": "statistics/bedroom_median_contrasts.csv",
    "grouped_cv_model_comparison.csv": "training/grouped_cv_model_comparison.csv",
    "heldout_model_comparison.csv": "evaluation/heldout_model_comparison.csv",
    "primary_state_errors.csv": "evaluation/primary_state_errors.csv",
    "unseen_state_errors.csv": "evaluation/unseen_state_errors.csv",
    "training_defined_tail_sensitivity.csv": "evaluation/training_defined_tail_sensitivity.csv",
}
REPORTS = {
    "source_assessment.md": "source_assessment.md",
    "model_readiness.md": "readiness/model_readiness.md",
    "preparation_review.md": "preparation/preparation_review.md",
    "statistical_note.md": "statistics/statistical_note.md",
    "model_development.md": "training/model_development.md",
    "heldout_evaluation.md": "evaluation/heldout_evaluation.md",
}
EXPECTED_ROW_COUNTS = {
    "source_rows": 99492,
    "eligible_rows": 99487,
    "distinct_eligible_rows": 99403,
    "modeling_rows": 99101,
    "primary_train_rows": 79196,
    "primary_test_rows": 19905,
    "geo_train_rows": 82247,
    "geo_test_rows": 16854,
}


def load_json(path):
    if not path.is_file():
        raise ValueError(f"Required evidence missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_evidence(root):
    if file_digest(locate_source(root)) != EXPECTED_SOURCE_SHA256:
        raise ValueError("Source hash mismatch")
    path = root / BASE
    preparation = load_json(path / "preparation/preparation_manifest.json")
    training = load_json(path / "training/model_selection.json")
    evaluation = load_json(path / "evaluation/evaluation_manifest.json")
    for key, expected in EXPECTED_ROW_COUNTS.items():
        owning = preparation if key in {"source_rows", "eligible_rows", "distinct_eligible_rows", "modeling_rows"} else evaluation
        if owning.get(key) != expected:
            raise ValueError(f"Incorrect {key}: {owning.get(key)}")
    if (
        training.get("selected_model") != EXPECTED_SELECTED_MODEL
        or evaluation.get("selected_model") != EXPECTED_SELECTED_MODEL
        or training.get("holdout_score_computed") is not False
        or evaluation.get("full_holdout_scoring") is not True
        or evaluation.get("no_listing_level_exports") is not True
        or training.get("n_cv_folds") != 3
        or training.get("training_listings") != 79196
    ):
        raise ValueError("Frozen modeling and evaluation agreement failed")
    if any(manifest.get("assignment_sha256") != SPLIT_DIGEST for manifest in (preparation, training, evaluation)):
        raise ValueError("Frozen assignment hash mismatch")
    if any(manifest.get("source_sha256", manifest.get("data_sha256")) != EXPECTED_SOURCE_SHA256 for manifest in (preparation, training, evaluation)):
        raise ValueError("Source provenance mismatch")
    if any(evaluation.get(field) != 0 for field in ("primary_group_overlap", "geo_state_overlap")):
        raise ValueError("Holdout overlap detected")
    comparison = pd.read_csv(path / "evaluation/heldout_model_comparison.csv")
    cv = pd.read_csv(path / "training/grouped_cv_model_comparison.csv")
    if len(comparison) != 4 or len(cv) != 3:
        raise ValueError("Evaluation or cross-validation model table incomplete")
    if len(comparison.drop_duplicates(["evaluation", "model"])) != 4:
        raise ValueError("Duplicate evaluation comparison")
    if set(zip(comparison["evaluation"], comparison["model"])) != {
        (experiment, model)
        for experiment in ("primary", "geographic")
        for model in ("median_baseline", EXPECTED_SELECTED_MODEL)
    }:
        raise ValueError("Unexpected evaluation comparison")
    if not set(cv["model"]) == {"median_baseline", "ridge_log_rent", EXPECTED_SELECTED_MODEL}:
        raise ValueError("Cross-validation comparison incomplete")
    if not pd.to_numeric(comparison["mae_usd"], errors="raise").ge(0).all():
        raise ValueError("MAE cannot be negative")
    if not pd.to_numeric(cv["mean_cv_mae_usd"], errors="raise").ge(0).all():
        raise ValueError("CV MAE cannot be negative")
    if cv.loc[cv["mean_cv_mae_usd"].idxmin(), "model"] != EXPECTED_SELECTED_MODEL:
        raise ValueError("Selected model does not agree with CV ranking")
    if abs(float(cv.loc[cv["model"].eq(EXPECTED_SELECTED_MODEL), "mean_cv_mae_usd"].iloc[0]) - training["selected_mean_cv_mae_usd"]) > .01:
        raise ValueError("Training model decision does not match CV table")
    for row in comparison.itertuples(index=False):
        expected_test = 19905 if row.evaluation == "primary" else 16854
        if row.heldout_listings != expected_test:
            raise ValueError("Held-out observation count mismatch")
    evidence_comparisons = {(r["evaluation"], r["model"]): r for r in evaluation["comparisons"]}
    for row in comparison.itertuples(index=False):
        other = evidence_comparisons.get((row.evaluation, row.model))
        if other is None or abs(float(row.mae_usd) - other["mae_usd"]) > .01:
            raise ValueError("Evaluation manifest and CSV differ")
    return preparation, training, evaluation, comparison, cv


def compact_score(comparison, experiment, model):
    values = comparison.loc[comparison["evaluation"].eq(experiment) & comparison["model"].eq(model)]
    if len(values) != 1:
        raise ValueError("Missing comparison row")
    return values.iloc[0]


def metric_table(comparison):
    lines = ["| Evaluation | Candidate | Holdout n | MAE (USD) | RMSE (USD) | R² |", "|---|---|---:|---:|---:|---:|"]
    for experiment in ("primary", "geographic"):
        for model in ("median_baseline", EXPECTED_SELECTED_MODEL):
            row = compact_score(comparison, experiment, model)
            label = "Median benchmark" if model == "median_baseline" else "Random forest"
            lines.append(f"| {'Property-grouped' if experiment == 'primary' else 'Unseen-state'} | {label} | {int(row['heldout_listings']):,} | ${row['mae_usd']:,.2f} | ${row['rmse_usd']:,.2f} | {row['r2']:.3f} |")
    return "\n".join(lines)


def generate_readme(preparation, training, evaluation, comparison, cv):
    primary_rf = compact_score(comparison, "primary", EXPECTED_SELECTED_MODEL)
    primary_base = compact_score(comparison, "primary", "median_baseline")
    geo_rf = compact_score(comparison, "geographic", EXPECTED_SELECTED_MODEL)
    geo_base = compact_score(comparison, "geographic", "median_baseline")
    improvement = 100 * (primary_base["mae_usd"] - primary_rf["mae_usd"]) / primary_base["mae_usd"]
    geo_gap = 100 * (geo_rf["mae_usd"] - geo_base["mae_usd"]) / geo_base["mae_usd"]
    geo_statement = (
        f"**Geographic transfer falls short on the preselected MAE criterion.** The forest's ${geo_rf['mae_usd']:,.2f} error is {geo_gap:.1f}% higher than the regional experiment's ${geo_base['mae_usd']:,.2f} median baseline."
        if geo_gap > 0 else
        f"**Geographic stress-test result.** The forest's ${geo_rf['mae_usd']:,.2f} error is {abs(geo_gap):.1f}% lower than the regional experiment's ${geo_base['mae_usd']:,.2f} median baseline."
    )
    return f"""# The Rental Price Puzzle

**A historical US rental-listings study of price variation, property attributes, and model generalization**

*Real-estate analytics · CRISP-DM · 2018–2019 listings · Reproducible Python research*

**Author — {AUTHOR}**

## Executive finding

Across **99,101** eligible listings with known city and state, a random forest trained using group-separated cross-validation estimated advertised monthly rent with **${primary_rf['mae_usd']:,.2f} MAE** on **19,905 held-out listings**, versus **${primary_base['mae_usd']:,.2f}** for the training-median baseline. This is a **{improvement:.1f}% reduction in average absolute error** on the primary evaluation. The median listed rent in the broader distinct eligible cohort was **$1,350/month**.

{geo_statement} The geographic stress test withholds entire state codes, so the stronger primary result must **not** be generalized to previously unseen state markets. This contrast—not a claim of universally reliable rent prediction—is the project's main takeaway.

![Model evaluation across two holdout designs](portfolio/figures/holdout_mae_comparison.png)

## The question

How much of the variation in historical **advertised asking rent** is associated with property size, room counts, listing features, and location? Can a model predict rents for listings separated from similar property specifications, and does the same model design transfer to **states excluded from training**?

We distinguish descriptive patterns from causal effects and model estimates from actual signed lease prices. No affordability or present-day US rental-index claim is made.

## Research design

The study follows a six-phase CRISP-DM workflow: problem definition, source understanding, data preparation, modeling, evaluation, and delivery. The original data source includes **99,492 records across 22 fields**; **99,487** meet basic monthly-USD positive-price and area conditions. Removing **84 exact row duplicates** leaves **99,403**, of which **99,101** contain both city and state. All retained high rents and large units remain in the primary analysis; the tail analysis is reported separately.

| Evaluation decision | Frozen protocol |
|---|---|
| Target | Historical advertised monthly asking rent, USD |
| Predictors | Square feet, bedrooms, bathrooms, category, state, city, photo, fee, pet-policy indication |
| Excluded | `price_display`, identifiers, listing text, addresses, source coordinates, timestamp, price-derived rent per sq ft |
| Selection | 3-fold training-only property-grouped cross-validation; minimum mean MAE |
| Candidates | Median baseline, regularized Ridge, Random Forest |
| Primary test | 79,196 training / 19,905 held-out listings; no matching property-group fingerprint across partitions |
| Geographic test | Independent refit on 82,247 listings from 43 state codes; 16,854 listings from 8 withheld codes |
| Transformations | Numeric imputation and category encoding fitted within training folds only; Ridge uses training-fold bounded quantiles |
| Outliers | No blanket rent or area winsorization in primary results; tail sensitivity reported separately |

The fingerprint of the frozen partition design is `{SPLIT_DIGEST}`. Property grouping reduces near-specification leakage but cannot establish physical-unit identity. The geographic experiment is a stress test using a separate fitted model, not another score on the same estimator.

## Key numerical results

{metric_table(comparison)}

The Random Forest was selected **before either held-out evaluation**, with training-only grouped-CV MAE of **${training['selected_mean_cv_mae_usd']:,.2f}**. Ridge was stabilized during training analysis before test scoring; the selected forest's specification was unchanged. Geographic Random Forest RMSE was **${geo_rf['rmse_usd']:,.2f}** versus **${geo_base['rmse_usd']:,.2f}** for the baseline, but the MAE ranking reverses and both geographic R² values are negative. This mixed evidence matters: success depends on the metric and deployment population.

<details><summary>Training-only cross-validation results</summary>

| Candidate | Mean CV MAE (USD) | Mean CV R² |
|---|---:|---:|
{chr(10).join(f'| {str(row.model).replace("_", " ")} | ${row.mean_cv_mae_usd:,.2f} | {row.mean_cv_r2:.3f} |' for row in cv.itertuples(index=False))}

</details>

## Visual evidence

| Asking rents | Predictive fit |
|---|---|
| ![Asking-rent distribution](portfolio/figures/asking_rent_distribution.png) | ![Primary actual versus predicted rent](portfolio/figures/primary_actual_vs_predicted.png) |

| State concentration | Transfer to withheld states |
|---|---|
| ![Geographic source composition](portfolio/figures/state_sample_concentration.png) | ![Unseen-state error](portfolio/figures/unseen_state_mae.png) |

All charts are computed from the source data or aggregate evaluation artifacts. Distribution figures may limit the **display range** for readability; their displayed bounds do not remove high-price observations from the underlying analysis.

## Project assets and reading order

| Where | What to read or run |
|---|---|
| [`notebooks/01_market_context.ipynb`](notebooks/01_market_context.ipynb) | Audited cohort, missingness and market comparisons |
| [`notebooks/02_model_comparison.ipynb`](notebooks/02_model_comparison.ipynb) | Training-only CV, primary holdout and error reductions |
| [`notebooks/03_geographic_generalization.ipynb`](notebooks/03_geographic_generalization.ipynb) | Why unseen-state results limit the claim |
| [`market_study/`](market_study/) | Dedicated preparation, statistical, modeling, evaluation and release code |
| [`portfolio/tables/`](portfolio/tables/) | Reviewed, aggregate result tables |
| [`portfolio/reports/heldout_evaluation.md`](portfolio/reports/heldout_evaluation.md) | Extended final model evaluation |
| [`docs/portfolio_methodology.md`](docs/portfolio_methodology.md) | Research definitions, audit methods and evaluation interpretation |
| [`tests/`](tests/) | Unit tests and reproducibility/coverage gates |

## Reproduce the analysis

**Prerequisites:** Python 3.15 on Windows, Git, and the original UCI 100K source archive. Download the [dataset from UCI]({DATA_URL}), extract `apartments_for_rent_classified_100K.csv`, and place it anywhere under `data/raw/`. Source CSV SHA-256 must equal:

`{EXPECTED_SOURCE_SHA256}`

From the project root in PowerShell:

```powershell
py -3.15 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\\.venv\\Scripts\\python.exe analysis.py
.\\.venv\\Scripts\\python.exe -m pytest tests -q --basetemp=.venv/pytest-tmp
.\\.venv\\Scripts\\python.exe -m market_study.verify_portfolio
```

`analysis.py` regenerates the research stages in order, including model training and the two held-out evaluations, then rebuilds the curated publication assets from verified evidence. Notebook sources run against the checked-in aggregate `portfolio/` tables and figures; no raw listing export is needed to read them.

## Interpretation and limitations

The original data are self-selected classified advertisements from **December 2018 through December 2019** rather than signed leases, comprehensive coverage, or current rental inventory. Missing amenities and pet-policy fields cannot be treated as explicit negatives. Source sampling is geographically uneven, and visually matching property specifications are not proof of identical underlying properties. This is a historical, observational prediction study; no causal price effect or policy recommendation follows from its association tests. The unseen-state result makes a general-purpose national price estimator inappropriate without further geography-aware development and independent validation.

**Conclusion.** On the frozen primary property-grouped holdout, a nonlinear model predicts historical asking rent substantially more accurately than a simple median. On geographically unseen state codes, its absolute-dollar error does not transfer comparably. The evidence supports a **scope-limited historical rental model**, not reliable prediction for unknown US states.

## Dataset attribution

{UCI_CITATION} Dataset page: [{DATA_URL}]({DATA_URL}). License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The project applies original analysis and derived summary visualizations; the underlying UCI dataset is credited to its original contributors. Raw listings, addresses, individual predictions and row-level assignments are not included in this repository.

---

**Project author:** {AUTHOR}
"""


def run(root):
    preparation, training, evaluation, comparison, cv = assert_evidence(root)
    destination = root / "portfolio"
    for folder, definitions in (("figures", FIGURES), ("tables", TABLES), ("reports", REPORTS)):
        target = destination / folder
        target.mkdir(parents=True, exist_ok=True)
        for filename, original in definitions.items():
            source = root / BASE / original
            if not source.is_file():
                raise ValueError(f"Missing curated publication artifact: {source}")
            if folder == "tables":
                columns = pd.read_csv(source, nrows=0).columns
                forbidden = {"id", "listing_id", "address", "latitude", "longitude", "body", "title", "time", "price", "price_display", "actual_rent", "predicted_rent", "property_group"}
                if set(columns.str.lower()) & forbidden:
                    raise ValueError(f"Listing-level or target data in curated table: {filename}")
            if folder == "figures" and source.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Invalid chart PNG: {filename}")
            copyfile(source, target / filename)
    manifest = {
        "author": AUTHOR,
        "dataset_citation": UCI_CITATION,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "assignment_sha256": SPLIT_DIGEST,
        "selected_model": EXPECTED_SELECTED_MODEL,
        "primary_mae_usd": float(compact_score(comparison, "primary", EXPECTED_SELECTED_MODEL)["mae_usd"]),
        "geographic_mae_usd": float(compact_score(comparison, "geographic", EXPECTED_SELECTED_MODEL)["mae_usd"]),
        "figures": list(FIGURES),
        "tables": list(TABLES),
        "reports": list(REPORTS),
    }
    (destination / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text(generate_readme(preparation, training, evaluation, comparison, cv), encoding="utf-8")
    return manifest


def main():
    result = run(Path.cwd())
    print("PASS - Final README generated from source-matched evidence")
    print("PASS - Curated portfolio figures:", len(result["figures"]))
    print("PASS - Curated portfolio tables:", len(result["tables"]))
    print("PASS - Curated technical reports:", len(result["reports"]))
    print("PASS - No raw listing or individual prediction export")


if __name__ == "__main__":
    main()
