# Training-only statistical exploration

This exploratory assessment uses **79,196** training listings from the fixed property-group holdout. All comparisons are descriptive and unadjusted, not causal estimates. The remaining **19,905** listings are excluded from statistical exploration here, protecting the primary held-out evaluation. Source: UCI Apartment for Rent Classified, DOI https://doi.org/10.24432/C5X623 (CC BY 4.0).

- Spearman's rank correlation of floor area and advertised rent in training: **0.366**. Rank correlation does not isolate the effects of location or property type.
- One-bedroom median advertised rent: **$1,220**; two-bedroom median: **$1,400**.
- Unadjusted two-minus-one bedroom median difference: **$180**, with an approximate percentile bootstrap 95% interval of **$174 to $191** (300 seeded replicates).
- A Mann–Whitney comparison and Spearman p-values are included in the associated tables as exploratory evidence, not as confirmation of a causal rental premium. With a large observational sample, very small p-values may correspond to modest practical differences.

The bootstrap intervals are listing-level descriptive intervals and do **not** account for the dependence of listings within property-specification groups, reposted listings, sampling bias, or multiple testing. They should not be treated as nationally representative uncertainty intervals. Property attributes, location, and listing coverage are confounded. Do not use these training-only association results to adjust model choices after viewing the held-out results.

The statistical figures present the observed training sample. They have not been geographically or compositionally weighted to represent all US apartments.
