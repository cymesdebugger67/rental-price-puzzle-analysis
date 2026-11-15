# Modeling readiness — historical rental classifieds

## Review scope

The source is the UCI Apartment for Rent Classified collection (DOI: https://doi.org/10.24432/C5X623; CC BY 4.0). These are historic advertised rents, not current rents or executed leases. This checkpoint audits samples and candidate features without training or validating a model.

## Cohort reconciliation

- Original records: 99,492
- Monthly USD listings with positive rent and area: 99,487
- After removing exact complete-row duplicates: 99,403
- Distinct eligible listings with city and state: 99,101
- Repeated IDs after exact-row removal: 0

## Distribution and market coverage

- Monthly asking rent: median $1,350; 99th percentile $4,250; maximum $52,500.
- Advertised floor area: median 900 sq ft; 99th percentile 2,259 sq ft; maximum 50,000 sq ft.
- Rent per square foot: median $1.47; 99th percentile $4.68; maximum $37.02.
- Five highest-volume state codes account for 42.8% of records with known city and state. This reflects source coverage, not a population-weighted national rental market.
- Listing timestamps: 2018-12-07T09:20:18+00:00 to 2019-12-26T20:17:05+00:00.

## Decisions before regression

1. Retain the original positive-price and positive-area eligible population for descriptive evidence, documenting exact-row deduplication. The model cohort and any additional exclusions must be frozen before model evaluation.
2. Inspect extreme high-rent, large-area, and high rent-per-square-foot records. Quantile screening is diagnostic only: it must not silently erase rare units. Compare model errors on all cases and central-range subsets.
3. Preserve a missing category for pets and amenities; unknown is not equivalent to absence.
4. For a location-augmented model, compare an ordinary holdout with a separate geography-out holdout. Keep transformations and feature selection within the training data.
5. Exclude `price_display`, price-per-square-foot, ID, text descriptions, addresses, and any target-derived information from predictors. City and state can be evaluated as location features, but location shift and source imbalance must be acknowledged.
6. Interpret all statistical relationships as observational associations, not causal price effects. The source is not a representative census of available rentals.

## Review artifacts

Tables contain tail diagnostics, missing-field coverage, state and city concentration, and a candidate feature policy. Figures visualize geographic sample imbalance and the rent and floor-area upper tails. No raw listing rows or addresses are exported.
