# Regression and Statistical Evaluation Protocol

**Project owner and author:** Cyril James Lunzaga

## Research contract

The Rental Price Puzzle uses the UCI Machine Learning Repository's historical *Apartment for Rent Classified* data (DOI [10.24432/C5X623](https://doi.org/10.24432/C5X623), CC BY 4.0). The predictive target is advertised monthly rent in USD, not signed rent or current market prices. The study is observational.

## Fixed study population and partitions

The preparation checkpoint verified 99,101 distinct, positive-area, positive-rent monthly USD listings with known state and city. The primary holdout allocates whole property-specification groups: 79,196 development listings and 19,905 final-test listings. A separate unseen-state holdout contains 82,247 development and 16,854 evaluation listings. Both assignments are hashed and frozen in `preparation_manifest.json`; each experiment must fit preprocessing and estimators exclusively on its own training portion.

## Phase 4: Training-only research

Run `python -m market_study.statistical_evidence` to produce exploratory rank correlations, group-specific medians, bootstrap intervals and geographic sample summaries on the primary **training** records only. Bootstrap intervals are listing-level descriptive, not cluster-adjusted or nationally representative. The observational differences do not establish causal effects.

Run `python -m market_study.train_price_models` to compare a training-median baseline, Ridge regression with log target, and Random Forest regression with log target. Use three property-group-isolated cross-validation folds on the 79,196 training records. All numeric imputation and categorical encoding are fitted within each CV training fold through a scikit-learn Pipeline. The price target, outcome-derived fields, IDs, individual addresses, latitude/longitude, listing text, and timestamp are excluded from prediction inputs. Categorical unknowns are not interpreted as the absence of an amenity or pet allowance.

Freeze the algorithm with the smallest **mean grouped CV MAE** in USD, reporting baseline, RMSE, R² and variation across folds. No final held-out scores are calculated during model selection. Compare substantive differences as diagnostics, not as causes.

## Training-stage numerical stability review

During the first real-data grouped-CV run, the Ridge candidate reported mean MAE of approximately $402,168, far above the training-median benchmark of approximately $524. The combination of StandardScaler on unbounded area/room counts, linear extrapolation on the log-rent target, and inverse exponentiation can amplify extreme predictor values. The resulting mean is not a useful indication of typical rental prediction error. This failure was identified using training-only CV before the final holdout was scored.

The model registry was corrected without changing cohort, split fingerprint, target observations, or the Random Forest configuration: Ridge now applies a monotonic uniform quantile mapping to numeric inputs, fitted inside each CV training fold. This bounds numeric predictors for out-of-range validation observations. Categorical encoding and missing-value imputation remain fold-fitted. The repaired candidate comparison must be rerun and its reported figures must replace the earlier diagnostic numbers before final test evaluation. The resulting comparison is an iterative development outcome, not a claim of fully preregistered algorithm selection.

## Phase 5: Final evaluation contract

Refit the selected Phase 4 specification on all 79,196 primary training records and compute held-out USD MAE, RMSE, R², baseline comparison, error quantiles, high-rent sensitivity and relevant subgroup diagnostics on all 19,905 primary test records. Avoid deleting high-rent cases from primary scoring; any central-range scores should be labeled sensitivity analyses and use training-defined thresholds.

For geographic stress testing, refit the **same selected specification** using only the 82,247 state-training records and evaluate on the 16,854 records from eight unseen state codes. Do not tune the model using the geographic evaluation target. Report whether unseen-state performance weakens, but do not generalize this particular held-out set to all states.

## Reproducibility and release

The source dataset is retained locally and excluded from Git. Generated aggregate evidence may be included in Git after source/figure review, excluding raw listing rows and identifying address text. Model comparison and evaluation reports must disclose actual split counts, random seed, feature policy, algorithm settings, Python versions, source attribution, and limitations. The full project is ready for GitHub main integration only after tests, notebook execution, report review, and privacy checks pass.
