# Study design and interpretation

## Problem and estimand

The Rental Price Puzzle investigates the distribution and supervised prediction of **monthly advertised asking rent** in historical US apartment classified listings dated December 2018–December 2019. It does not estimate signed contract rent, current rental supply, causal rental premiums, or housing affordability. Analyses follow the six CRISP-DM stages, with software tests and controlled publication checks.

## Population and integrity

Original UCI Apartment for Rent Classified 100K archive: 99,492 rows and 22 columns. Source DOI [10.24432/C5X623](https://doi.org/10.24432/C5X623); license [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Raw source SHA-256 `6adeae31e5507e39feb03acf811547a5331d2329c5a24ed1b1f1455fc1835916`.

The cohort selects monthly USD rents with positive price and floor area (99,487), removes 84 exact duplicate complete rows (99,403), and requires known city and state for the modeling population (99,101). High-rent and high-area listings remain in primary metrics. Group specifications and coordinates are used to reduce likely near-duplicate leakage at partition creation, not as estimator inputs. Exact physical units cannot be uniquely identified.

## Statistical investigation

Distributional summaries, state/city comparisons, rent per square foot, and sample imbalance are descriptive. Training-only Spearman correlations, Mann–Whitney comparisons and bootstrap median intervals are exploratory and are not causal estimates. With large samples, very small p-values may coexist with modest practical effects. Training-only inference also avoids using final test data to guide modeling decisions.

## Model design

A reproducible 80/20 property-group holdout is constructed by a deterministic hash assignment with a fixed seed, plus a separate unseen-state stress split. Three-fold grouped cross-validation on the primary training partition chooses among a training-median baseline, log-target Ridge and log-target Random Forest using mean MAE in USD. Preprocessing is fitted within training folds. No `price_display`, target-derived field, identifier, free text, or exact coordinate enters prediction features. Ridge's initial preprocessing instability was corrected during the training stage, before any held-out scoring. The model and both splits remain frozen thereafter.

## Independent evaluation

Primary evaluation fits the selected specification on 79,196 listings and scores 19,905 with no shared property-group key. Geographic stress evaluation independently refits on 82,247 listings across 43 state codes and scores 16,854 from eight other codes. Both experiments compare their own fitted median baseline and selected forest. MAE is the selection and headline evaluation metric; RMSE and R² provide additional insight. An observed-target-defined upper-tail diagnostic is permissible **only after selection** and must not replace the full held-out metrics. Tables suppress small subgroup reporting and do not export individual predictions.

The final geographic result shows a higher Random Forest MAE than the median baseline despite slightly better RMSE and R². This is a central finding about limited transfer, not a defect to conceal. It does not justify retraining on withheld states or tuning to those scores while calling the same holdout unbiased.

## Deliverable review

`analysis.py` regenerates all outputs using the raw source; `market_study.publish_portfolio` validates signed-off manifests and copies only a limited allowlist of aggregate figures/tables/reports into `portfolio/`. `market_study.verify_portfolio` validates release counts, figure signatures, notebook structure, summary metrics and README link integrity. Tests should be executed with a project-local pytest temporary directory on Windows. `portfolio/` is safe for selective version-control review; raw archives, row-level split assignments, private paths and listing-level predictions should remain untracked.

## Authorship and source acknowledgment

Project author: **Cyril James Lunzaga**. Data citation: *Apartment for Rent Classified [Dataset]. (2019). UCI Machine Learning Repository*. [DOI:10.24432/C5X623](https://doi.org/10.24432/C5X623), CC BY 4.0. Derived analysis and summaries are distinct from the source authors' contribution. The analytical conclusions are valid only for the documented historical, observed classified data and frozen evaluations.
