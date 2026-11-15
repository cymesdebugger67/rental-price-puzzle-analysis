# Regression modeling: training-only comparison

This experiment compares models using only the **79,196** training observations from the frozen property-specification grouped split. Source: UCI Apartment for Rent Classified (2018–2019), DOI https://doi.org/10.24432/C5X623, CC BY 4.0. Model inputs are floor area, bedrooms, bathrooms, listing category, state, city, photo indicator, fee indicator, and pet-policy text; `price_display`, price-per-square-foot, IDs, addresses, listing descriptions, and timestamps are excluded.

## Training design

- The primary holdout of **19,905** listings has not been used to fit, tune, select, or score models in this phase.
- Each of the **3** grouped validation folds isolates exact property-specification groups between its training and validation sides. Similar units can still share nonidentical specifications.
- Missing numeric predictors are imputed and categorical values are one-hot encoded **inside each training fold**. Unknown category labels in validation are ignored; missing categorical details are represented as Unknown, not treated as a negative answer.
- The baseline predicts each fold's training median asking rent. Ridge regression and Random Forest predict log1p(price) internally and convert predictions back to monthly USD. Price-derived features are excluded.
- The initial training-only Ridge diagnostic suffered severe extrapolation under unbounded standardized numeric inputs followed by inverse exponential target transformation. Before examining any final test outcomes, Ridge numeric inputs were changed to a bounded rank-quantile mapping fitted separately within each CV training fold. This is a documented training-stage model repair, not a change to the held-out partitions or target records. The Random Forest preprocessing and algorithm configuration were not changed.
- Model selection uses the lowest mean grouped cross-validation **MAE**, without selecting on the held-out test data. RMSE and R² are supplementary diagnostics.

## Training outcome

- Selected algorithm: **random forest log rent**.
- Selected training-only mean grouped-CV MAE: **$321.61** per month.
- Median-baseline training-only mean grouped-CV MAE: **$523.98** per month.

These are cross-validated training results, **not** out-of-sample scores on the final holdout. The next evaluation phase will refit the selected specification on the complete primary training split, then report errors on the untouched primary holdout and separately refit the same specification for the unseen-state stress test. Geographic and high-rent subgroup diagnostics will accompany aggregate scores.

This retrospective classifieds study is not a deployment-ready rent estimator. It neither measures actual executed rents nor establishes causal rental premiums.
