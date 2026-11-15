# Held-out rental price evaluation

## Final model decision

Random Forest regression with a log1p monthly asking-rent target was selected **before** either held-out evaluation, by the lowest mean property-grouped training cross-validation MAE ($321.61). The training-only model decision is preserved in `training/model_selection.json`. This evaluation does **not** retune features, model parameters, partitions, or price thresholds using the final tests.

## Results

| Experiment | Model | Listings | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| geographic | median baseline | 16,854 | $911.14 | $1,674.89 | -0.333 |
| geographic | random forest log rent | 16,854 | $1,036.30 | $1,644.82 | -0.285 |
| primary | median baseline | 19,905 | $520.32 | $970.16 | -0.034 |
| primary | random forest log rent | 19,905 | $312.37 | $628.83 | 0.565 |

The primary property-grouped experiment refits the selected model on 79,196 listings and evaluates it on 19,905 listings with no shared property-group fingerprints. The selected model's test MAE is $312.37 versus $520.32 for its own training-median baseline, a **40.0%** relative reduction; RMSE is $628.83, and R² is 0.565. These are historical advertised asking-rent prediction errors, not actual leasing errors.

For geographic generalization, the same frozen model specification is independently refitted on 82,247 records from 43 states and scored on 16,854 listings from eight withheld states. Its MAE is $1,036.30 versus $911.14 for the geographic training-median baseline, **13.7% higher MAE**; RMSE is $1,644.82, and R² is -0.285. Different training/test samples mean this experiment is a stress test, not a strictly controlled paired comparison of the two held-out metrics.

## Diagnostic interpretation

Primary-training p99 rent threshold: **$4,238.05**. Geographic-training p99 threshold: **$3,646.08**. Both are determined using only the corresponding training partitions. The primary reported MAE, RMSE and R² retain **all** held-out listings, including high-rent properties. The separate tail table reports conditional errors above and below each training-defined p99; it does not present a replacement main metric.

Residual and subgroup tables include only aggregates, without individual listing predictions or addresses. State- and bedroom-group summaries suppress groups with fewer than 30 held-out records. Prices used to label diagnostic bands are **observed test targets**, for evaluation after selection only, not predictors or model-selection information. Missing pet-policy and amenity information cannot be interpreted as absence of those features. No causal price premiums are inferred.

The unseen-state table contains 7 state groups meeting the minimum reporting threshold. Its state-level error variation reflects both true market heterogeneity and the source's unequal, nonrepresentative listing coverage. The geographic holdout does not establish a national generalization bound.

## Limitations and provenance

The source is the UCI *Apartment for Rent Classified* dataset (historical US classified records from 2018–2019), DOI [10.24432/C5X623](https://doi.org/10.24432/C5X623), CC BY 4.0. The CSV SHA-256 is `6adeae31e5507e39feb03acf811547a5331d2329c5a24ed1b1f1455fc1835916`. These are advertised prices rather than signed rental agreements; sample selection, missing fields, reposts, and property-specification collisions limit interpretation. The random split groups by state, city, category, size, room counts and coordinates but cannot establish unique physical properties. The state holdout is one deterministic partition rather than a full geographic cross-validation study.

The analytical project is authored by **Cyril James Lunzaga**. Numerical results in this report are generated from the local source data, not manually assumed.
