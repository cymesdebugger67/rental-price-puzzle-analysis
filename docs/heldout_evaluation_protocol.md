# Held-out evaluation protocol

**Project owner and author:** Cyril James Lunzaga

## Status and decision boundary

The Phase 4 Ridge stability correction was made using only grouped training-fold outcomes. The final model selection was subsequently rerun: Random Forest regression using log1p asking rent achieved the smallest three-fold property-grouped mean cross-validation MAE of approximately $321.61. The median baseline was approximately $523.98, and the repaired Ridge candidate was approximately $336.86. These values belong to training-only cross-validation, not either held-out test.

Phase 5 begins after this decision is frozen. No algorithm, preprocessing, predictor, threshold, or partition is selected using either held-out target. The script refuses evaluation if the dataset hash, saved partition digest, cohort counts, or training-only model decision differ from the approved values.

## Two independent evaluations

1. **Primary property-grouped holdout:** refit the preselected Random Forest specification using all 79,196 primary training rows, then evaluate 19,905 held-out records. The comparator is a newly fitted training-median baseline. Exact property-group hashes must be disjoint across the split.
2. **Geographic stress test:** independently refit that same specification on 82,247 training listings drawn from 43 state codes and evaluate 16,854 records from eight unseen state codes. Compare it with that experiment's own training-median baseline. No state code may appear on both sides.

Both experiments report USD MAE, USD RMSE, and R². Scores are calculated on all records in the frozen held-out cohort, including extremely high asking rents. The geographic holdout is a predefined secondary diagnostic rather than a second opportunity to tune.

## Diagnostic reporting

- State and bedroom errors summarize aggregate subgroup MAE and signed bias, with groups smaller than 30 listings excluded from publication.
- Price-band diagnostics condition on observed held-out asking rents after fitting, with boundary values determined solely from each experiment's training portion. They inform interpretation, never feature engineering or tuning.
- Error quantiles describe the full held-out error distribution. High-rent sensitivity reports results on records above and below each **training-derived** 99th-percentile asking-rent threshold.
- Display-limited charts label the plotted percentile cap; all original held-out records remain in headline metrics and tables.
- Individual listing predictions, IDs, raw addresses, raw coordinates and free-text descriptions are not exported.

## Methodological limits

These 2018–2019 US rental classifieds are a nonrepresentative historical convenience sample. Advertised asking prices do not necessarily equal signed rents. Listings with different IDs may refer to the same property, and exact matching specifications need not imply identical units. Performance on eight withheld states cannot establish generalization to every US market. Statistical associations do not identify causal rental premiums.

Source: [UCI Apartment for Rent Classified](https://doi.org/10.24432/C5X623), DOI 10.24432/C5X623, CC BY 4.0. The source CSV is retained outside the Git index and verified against SHA-256 `6adeae31e5507e39feb03acf811547a5331d2329c5a24ed1b1f1455fc1835916`.
