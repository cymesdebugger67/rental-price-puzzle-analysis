# Data preparation and split-design review

## Purpose and scope

This report fixes the modeling cohort and the evaluation boundaries for a historical US apartment classified-listings study. It does not present a fitted model or report predictive performance. Original source: UCI Apartment for Rent Classified, DOI https://doi.org/10.24432/C5X623, CC BY 4.0, 2018–2019 listings.

## Modeling cohort

- Source rows verified against SHA-256 `6adeae31e5507e39feb03acf811547a5331d2329c5a24ed1b1f1455fc1835916`: **99,492**.
- Provisional monthly USD listings with positive asking rent and advertised area: **99,487**.
- Exact complete-row duplicates removed: **84**.
- Distinct eligible listings: **99,403**.
- Modeling cohort requiring both city and state: **99,101**.
- Repeated listing IDs in modeling cohort: **0**.

The modeling cohort does not discard listings merely for high asking rent or large area. These extremes are retained in the primary analysis. The 1st and 99th percentile thresholds in `training_tail_reference.csv` are calculated **from the primary training portion only** and are reserved for sensitivity reporting. No target winsorization, filtering, or feature selection has been fitted on held-out records.

## Predeclared evaluation design

- Primary comparison: seeded SHA-256 group assignment with an approximately 80/20 target, grouping by normalized state, city, category, area, bedrooms, bathrooms, latitude, and longitude, **not by price**. Actual partitions: **79,196 train**, **19,905 test**. Group-overlap check: **passed**.
- Geographic stress test: separate deterministic holdout of entire state codes. Actual partitions: **82,247 train**, **16,854 test**. State-overlap check: **passed**. This is a separate experiment, not a second score on the primary fitted model.
- Assignment seed: **2026**. Assignment audit digest: `eed0e59e56f4c457841a882f10af66d3828782b131f829b9e5c1627cfb3e135d`. Re-running source preparation reproduces the same partitions.
- The grouped holdout may conservatively group distinct apartments sharing advertised specifications or coordinates. It reduces potential near-duplicate leakage but cannot prove unique-property identification.

## Candidate modeling policy

Permitted predictors are `square_feet`, `bedrooms`, `bathrooms`, `category`, `state`, `cityname`, `has_photo`, `fee`, `pets_allowed`. The target is advertised monthly asking rent `price`. No other source fields enter the baseline feature matrix by default. Identifiers, addresses, free text, exact coordinates, source platform, timestamps, `price_display`, and target-derived rent per square foot are excluded from predictors. The grouping coordinates are used **only to construct split boundaries**, not as predictors.

Numeric missingness and scaling, categorical missingness and encoding, model selection, and any target transformation must be fitted exclusively on the relevant training fold within a modeling pipeline. No imputed or encoded feature matrix has been exported here.

## Next evaluation contract

Compare a training-median baseline, regularized linear regression, and a nonlinear ensemble. Report held-out MAE (primary), RMSE, R², subgroup errors, and sample sizes in USD. Tune with training-only cross-validation; evaluate the reserved test once after final model selection. Treat extreme-value sensitivity and unseen-state evaluation separately. Report limitations of self-selected classified listings, noncausal associations, duplicate property uncertainty, and geographical imbalance. Do not interpret differences as a present-day market index.

## Reproducibility and release

Generated partition summaries and threshold references are committed only after review. The original row-level data stay in `data/raw/` and are ignored by Git. Raw listing-level split assignments are not exported publicly. Source references and final chart/table links belong in the portfolio README at release.
