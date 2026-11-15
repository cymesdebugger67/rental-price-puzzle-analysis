# The Rental Price Puzzle — Research Design

**Project owner and author:** Cyril James Lunzaga

## Objective

Describe differences in advertised monthly apartment rents across historical US listings, investigate associations with advertised property attributes and locations, and assess the reliability of predictive regression models. The work is a retrospective analytics case study, not a valuation service or a current rental price index.

## Questions

1. How are asking rents distributed, and how heavily are the listings concentrated in particular states and cities?
2. How do advertised area, bedroom count, bathroom count, and listing categories relate to observed rent differences?
3. What descriptive rent differences appear across major geographic markets, and how do source coverage and missing fields limit interpretation?
4. How much better do out-of-sample regression methods perform than a training-only median baseline?
5. Does performance deteriorate when evaluated in states excluded from training, or on high-rent observations?

## Source and unit of analysis

The unit is a **classified listing**, not a property, a household, or an executed rental contract. The study uses the UCI Machine Learning Repository's *Apartment for Rent Classified* dataset (DOI: https://doi.org/10.24432/C5X623, CC BY 4.0), comprising historical 2018–2019 listings. Data are raw and uncommitted in `data/raw`. Source CSV SHA-256: `6adeae31e5507e39feb03acf811547a5331d2329c5a24ed1b1f1455fc1835916`.

## Method and acceptance conditions

The analytical lifecycle follows CRISP-DM: business understanding, data understanding, data preparation, modeling, evaluation, and delivery. Phase-specific release checks add source integrity, deterministic execution, testing, provenance, and owner attribution.

- **Data understanding:** Verify source identity and SHA-256; report cohort selection, missingness, geographic coverage, duplicates, percentiles, and outlier diagnostics.
- **Preparation:** Use monthly USD listings with strictly positive advertised rent and area; remove exact duplicate complete rows; require both city and state for location-aware modeling. Do not silently discard rare high-rent properties.
- **Modeling:** Fit preprocessing and hyperparameter choices exclusively within the training portion. Compare training-median baseline, regularized regression, and nonlinear ensemble models. Keep price-derived fields and identifiers out of features.
- **Evaluation:** Report held-out USD MAE, RMSE, R² and data counts; evaluate geographic generalization separately; test subgroup performance and sensitivity to extreme asking rents. Never use held-out scores to tune features or thresholds.
- **Delivery:** Cite and attribute the source, link reviewed generated reports, provide commands to reproduce the analysis, verify Git exclusions, and document limitations and final outcomes. Public-facing author credit is Cyril James Lunzaga.

## Scope limitations

Observational correlations do not establish causality. Asking rents are not signed transaction prices. Listings may be reposted under different IDs, and source selection is not guaranteed nationally representative. Missing amenity or pet details cannot be interpreted as the absence of a feature. Geographic holdouts quantify an operationally difficult generalization case, not general housing-market accuracy.

## Study outcome and verification

The research stages are complete: the source was verified, the 99,101-listing modeling cohort and deterministic holdouts were prepared, three regression candidates were compared by training-only grouped cross-validation, and the selected Random Forest was evaluated on frozen primary and unseen-state holdouts. The primary holdout yielded $312.37 MAE versus the $520.32 median baseline; the unseen-state experiment yielded $1,036.30 versus $911.14, respectively. The geographic shortfall is explicitly retained as a central limitation. The final portfolio includes 12 curated figures, 12 aggregate tables, six analytical reports, three notebooks, and automated tests. Publication and Git repository status are independent release-management actions; this document does not assert a GitHub push or merge.
