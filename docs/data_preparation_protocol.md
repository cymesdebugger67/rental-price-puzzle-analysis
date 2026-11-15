# Data Preparation Protocol

## Study population

The descriptive source includes every record in the original UCI 100K CSV. The provisional eligible cohort requires USD, monthly advertised rent greater than zero, and advertised square feet greater than zero. Remove exact whole-row duplicates only. Location-aware modeling requires both state and city. Keep missing bedrooms, bathrooms, pets, fees, and image indicators as missing until training-only preprocessing is fitted.

## Extreme observations

The 1st/99th percentile screens identify extremes, but do not establish errors or justify deleting all high-value apartments. Preserve the full eligible modeling cohort for primary held-out evaluation. Only calculate sensitivity cutoffs from the appropriate training partition; preserve results on the complete held-out sample and report central-range sensitivity separately. Outlier intervention should be justified by a source validity finding or declared sensitivity scenario.

## Primary partition

Use seeded SHA-256 assignment of non-target property-specification groups constructed from normalized state, city, category, square footage, bedrooms, bathrooms, latitude, and longitude. Rent and IDs are explicitly absent from the grouping key. Train/test grouping is deterministic and targets an 80/20 allocation by group; exact record fractions can differ. Nearby but non-identical records may still represent the same physical unit; the grouping cannot guarantee property identity.

## Geographic stress experiment

Separately assign entire state codes to a state-held-out test portion with the same hash seed. No state appears on both sides. This is a generalization diagnostic, and its fitted models and preprocessing must be independent of the primary split's test data.

## Model inputs

Candidate features: square_feet, bedrooms, bathrooms, category, state, cityname, has_photo, fee, pets_allowed. Target: price. Exclude id, price_display, title, body, address, direct latitude/longitude, time, source, and target-derived variables such as price per square foot from feature matrices. Lat/long is restricted to split grouping, not model predictors.

## Reproducibility

Run `python -m market_study.prepare_modeling_data` from the project root after acquiring the UCI dataset. Validate `outputs/market_research/preparation/preparation_manifest.json` and the five CSV summaries. The fingerprint of row-to-partition assignments is stored only as a SHA-256 digest, not exported as raw listing rows. No model is trained in this stage.

## Release control

Keep the raw archive and CSV excluded from Git. Review summary reports and charts before controlled Git staging. The current feature branch must remain separate from `main` until full project quality assurance and release approval.
