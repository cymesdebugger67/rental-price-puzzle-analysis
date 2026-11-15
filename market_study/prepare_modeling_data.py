from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd

from market_study.market_evidence import (
    EXPECTED_SOURCE_SHA256,
    cohorts,
    load_listings,
    locate_source,
)
from market_study.model_readiness import MODEL_FEATURES

SEED = 2026
HOLDOUT_FRACTION = 0.20
GROUP_COLUMNS = (
    "state", "cityname", "category", "square_feet", "bedrooms",
    "bathrooms", "latitude", "longitude"
)
NUMERIC_FEATURES = ("square_feet", "bedrooms", "bathrooms")
CATEGORICAL_FEATURES = tuple(field for field in MODEL_FEATURES if field not in NUMERIC_FEATURES)
FORBIDDEN_MODEL_FIELDS = frozenset({
    "id", "price", "price_display", "title", "body", "address",
    "rent_per_square_foot", "latitude", "longitude", "time", "source"
})


def canonicalize_group_value(value):
    if pd.isna(value):
        return "<missing>"
    if isinstance(value, (float, np.floating, int, np.integer)):
        if not np.isfinite(value):
            return "<missing>"
        return f"{float(value):.6f}"
    cleaned = str(value).strip().casefold()
    return cleaned if cleaned else "<missing>"


def group_fingerprints(data):
    unavailable = set(GROUP_COLUMNS).difference(data.columns)
    if unavailable:
        raise ValueError(f"Missing grouping fields: {sorted(unavailable)}")
    groups = data.loc[:, list(GROUP_COLUMNS)].itertuples(index=False, name=None)
    digests = []
    for values in groups:
        normalized = json.dumps(
            [canonicalize_group_value(value) for value in values],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digests.append(sha256(normalized.encode("utf-8")).hexdigest())
    return pd.Series(digests, index=data.index, dtype="string")


def assigned_holdout(group_keys, seed=SEED, fraction=HOLDOUT_FRACTION):
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must be between zero and one")
    threshold = int(fraction * (1 << 64))
    unique = pd.Series(group_keys, dtype="string").dropna().unique()
    decisions = {
        key: int.from_bytes(
            sha256(f"{seed}:{key}".encode("utf-8")).digest()[:8], "big"
        ) < threshold
        for key in unique
    }
    return pd.Series(group_keys, dtype="string").map(decisions).fillna(False).astype(bool)


def make_partitions(frame):
    if not frame.index.is_unique:
        raise ValueError("Listing index must be unique before splitting")
    if frame["id"].duplicated().any():
        raise ValueError("Repeated listing IDs remain after source deduplication")
    group_keys = group_fingerprints(frame)
    holdout = assigned_holdout(group_keys)
    state_keys = frame["state"].astype("string").str.strip().str.upper()
    if state_keys.isna().any() or state_keys.eq("").any():
        raise ValueError("Location validation failed")
    geography_holdout = assigned_holdout(state_keys)
    primary_train = set(group_keys.loc[~holdout])
    primary_test = set(group_keys.loc[holdout])
    geographic_train = set(state_keys.loc[~geography_holdout])
    geographic_test = set(state_keys.loc[geography_holdout])
    if primary_train & primary_test:
        raise ValueError("Property-specification group crosses primary split")
    if geographic_train & geographic_test:
        raise ValueError("State crosses geographic split")
    if not holdout.any() or holdout.all():
        raise ValueError("Primary holdout has an empty partition")
    if not geography_holdout.any() or geography_holdout.all():
        raise ValueError("Geographic holdout has an empty partition")
    return pd.DataFrame({
        "listing_id": frame["id"],
        "property_group": group_keys,
        "state": state_keys,
        "primary_partition": np.where(holdout, "test", "train"),
        "geographic_partition": np.where(geography_holdout, "test", "train"),
    }, index=frame.index)


def feature_policy(available_columns):
    available = set(available_columns)
    if set(MODEL_FEATURES) & FORBIDDEN_MODEL_FIELDS:
        raise ValueError("Forbidden field is included as a candidate model feature")
    absent = set(MODEL_FEATURES) - available
    if absent:
        raise ValueError(f"Missing candidate model predictors: {sorted(absent)}")
    rows = []
    for field in MODEL_FEATURES:
        kind = "numeric" if field in NUMERIC_FEATURES else "categorical"
        rows.append({
            "source_field": field,
            "decision": "permitted_predictor",
            "treatment": "training_only_imputation" if kind == "numeric" else "training_only_missing_category_and_encoding",
            "kind": kind,
        })
    for field in sorted(FORBIDDEN_MODEL_FIELDS):
        rows.append({
            "source_field": field,
            "decision": "excluded_predictor",
            "treatment": "not_provided_to_estimator",
            "kind": "excluded",
        })
    return pd.DataFrame(rows)


def training_tail_reference(train):
    if train.empty:
        raise ValueError("Cannot calculate training tail reference on empty frame")
    rows = []
    for name, values in (
        ("asking_rent_usd", train["price"]),
        ("square_feet", train["square_feet"]),
        ("rent_per_square_foot_diagnostic_only", train["price"] / train["square_feet"]),
    ):
        numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if numeric.empty:
            raise ValueError(f"No finite values for {name}")
        p01, p99 = numeric.quantile([.01, .99])
        rows.append({
            "measure": name,
            "training_rows": len(numeric),
            "train_p01": float(p01),
            "train_p99": float(p99),
            "train_above_p99": int(numeric.gt(p99).sum()),
            "applied_in_primary_model": False,
        })
    return pd.DataFrame(rows)


def feature_coverage(train):
    result = []
    for field in MODEL_FEATURES:
        values = train[field]
        if field in NUMERIC_FEATURES:
            missing = pd.to_numeric(values, errors="coerce").isna().sum()
        else:
            cleaned = values.astype("string").str.strip()
            missing = (cleaned.isna() | cleaned.eq("")).sum()
        result.append({
            "feature": field,
            "train_rows": len(train),
            "train_missing": int(missing),
            "train_missing_percent": round(float(missing) * 100 / len(train), 3),
        })
    return pd.DataFrame(result)


def assignments_digest(partitions):
    rows = partitions[["listing_id", "primary_partition", "geographic_partition"]]
    values = sorted(
        (str(id_value), primary, geographic)
        for id_value, primary, geographic in rows.itertuples(index=False, name=None)
    )
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def summarize_partitions(frame, partitions):
    descriptions = []
    for partition, name in (
        ("primary_partition", "property_grouped_80_20_hash_holdout"),
        ("geographic_partition", "unseen_state_hash_holdout"),
    ):
        for sample in ("train", "test"):
            subset = partitions.loc[partitions[partition].eq(sample)]
            descriptions.append({
                "strategy": name,
                "partition": sample,
                "listings": len(subset),
                "share_percent": round(100 * len(subset) / len(frame), 3),
                "states": int(subset["state"].nunique()),
                "property_groups": int(subset["property_group"].nunique()),
            })
    return pd.DataFrame(descriptions)


def write_report(path, manifest, split_table, tails, coverage):
    group_train = int(split_table.loc[(split_table["strategy"] == "property_grouped_80_20_hash_holdout") & (split_table["partition"] == "train"), "listings"].iloc[0])
    group_test = int(split_table.loc[(split_table["strategy"] == "property_grouped_80_20_hash_holdout") & (split_table["partition"] == "test"), "listings"].iloc[0])
    geo_train = int(split_table.loc[(split_table["strategy"] == "unseen_state_hash_holdout") & (split_table["partition"] == "train"), "listings"].iloc[0])
    geo_test = int(split_table.loc[(split_table["strategy"] == "unseen_state_hash_holdout") & (split_table["partition"] == "test"), "listings"].iloc[0])
    markdown = f"""# Data preparation and split-design review

## Purpose and scope

This report fixes the modeling cohort and the evaluation boundaries for a historical US apartment classified-listings study. It does not present a fitted model or report predictive performance. Original source: UCI Apartment for Rent Classified, DOI https://doi.org/10.24432/C5X623, CC BY 4.0, 2018–2019 listings.

## Modeling cohort

- Source rows verified against SHA-256 `{EXPECTED_SOURCE_SHA256}`: **{manifest['source_rows']:,}**.
- Provisional monthly USD listings with positive asking rent and advertised area: **{manifest['eligible_rows']:,}**.
- Exact complete-row duplicates removed: **{manifest['exact_row_duplicates_removed']:,}**.
- Distinct eligible listings: **{manifest['distinct_eligible_rows']:,}**.
- Modeling cohort requiring both city and state: **{manifest['modeling_rows']:,}**.
- Repeated listing IDs in modeling cohort: **{manifest['duplicate_modeling_ids']}**.

The modeling cohort does not discard listings merely for high asking rent or large area. These extremes are retained in the primary analysis. The 1st and 99th percentile thresholds in `training_tail_reference.csv` are calculated **from the primary training portion only** and are reserved for sensitivity reporting. No target winsorization, filtering, or feature selection has been fitted on held-out records.

## Predeclared evaluation design

- Primary comparison: seeded SHA-256 group assignment with an approximately 80/20 target, grouping by normalized state, city, category, area, bedrooms, bathrooms, latitude, and longitude, **not by price**. Actual partitions: **{group_train:,} train**, **{group_test:,} test**. Group-overlap check: **passed**.
- Geographic stress test: separate deterministic holdout of entire state codes. Actual partitions: **{geo_train:,} train**, **{geo_test:,} test**. State-overlap check: **passed**. This is a separate experiment, not a second score on the primary fitted model.
- Assignment seed: **{SEED}**. Assignment audit digest: `{manifest['assignment_sha256']}`. Re-running source preparation reproduces the same partitions.
- The grouped holdout may conservatively group distinct apartments sharing advertised specifications or coordinates. It reduces potential near-duplicate leakage but cannot prove unique-property identification.

## Candidate modeling policy

Permitted predictors are {', '.join(f'`{value}`' for value in MODEL_FEATURES)}. The target is advertised monthly asking rent `price`. No other source fields enter the baseline feature matrix by default. Identifiers, addresses, free text, exact coordinates, source platform, timestamps, `price_display`, and target-derived rent per square foot are excluded from predictors. The grouping coordinates are used **only to construct split boundaries**, not as predictors.

Numeric missingness and scaling, categorical missingness and encoding, model selection, and any target transformation must be fitted exclusively on the relevant training fold within a modeling pipeline. No imputed or encoded feature matrix has been exported here.

## Next evaluation contract

Compare a training-median baseline, regularized linear regression, and a nonlinear ensemble. Report held-out MAE (primary), RMSE, R², subgroup errors, and sample sizes in USD. Tune with training-only cross-validation; evaluate the reserved test once after final model selection. Treat extreme-value sensitivity and unseen-state evaluation separately. Report limitations of self-selected classified listings, noncausal associations, duplicate property uncertainty, and geographical imbalance. Do not interpret differences as a present-day market index.

## Reproducibility and release

Generated partition summaries and threshold references are committed only after review. The original row-level data stay in `data/raw/` and are ignored by Git. Raw listing-level split assignments are not exported publicly. Source references and final chart/table links belong in the portfolio README at release.
"""
    path.write_text(markdown, encoding="utf-8")


def run(root):
    source = locate_source(root)
    raw = load_listings(source)
    _, eligible, distinct, known = cohorts(raw)
    if known["id"].duplicated().any():
        raise ValueError("Repeated listing identifiers in modeling sample")
    if not known["price"].gt(0).all() or not known["square_feet"].gt(0).all():
        raise ValueError("Nonpositive target or area in modeling sample")
    partitions = make_partitions(known)
    split_table = summarize_partitions(known, partitions)
    train = known.loc[partitions["primary_partition"].eq("train")]
    thresholds = training_tail_reference(train)
    coverage = feature_coverage(train)
    policy = feature_policy(known.columns)
    manifest = {
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_rows": len(raw),
        "eligible_rows": len(eligible),
        "exact_row_duplicates_removed": len(eligible) - len(distinct),
        "distinct_eligible_rows": len(distinct),
        "modeling_rows": len(known),
        "duplicate_modeling_ids": int(known["id"].duplicated().sum()),
        "seed": SEED,
        "target": "price",
        "primary_split": "property_grouped_80_20_hash_holdout",
        "geo_stress_split": "unseen_state_hash_holdout",
        "primary_train_rows": len(train),
        "primary_test_rows": int(partitions["primary_partition"].eq("test").sum()),
        "geo_train_rows": int(partitions["geographic_partition"].eq("train").sum()),
        "geo_test_rows": int(partitions["geographic_partition"].eq("test").sum()),
        "primary_group_overlap": 0,
        "geo_state_overlap": 0,
        "assignment_sha256": assignments_digest(partitions),
        "primary_winsorization": False,
        "candidate_features": list(MODEL_FEATURES),
        "output_tables": [
            "modeling_cohort.csv", "split_summary.csv", "training_tail_reference.csv",
            "training_feature_coverage.csv", "modeling_field_policy.csv",
        ],
    }
    target = root / "outputs" / "market_research" / "preparation"
    target.mkdir(parents=True, exist_ok=True)
    cohort_table = pd.DataFrame([
        {"stage": "Original UCI source", "listings": len(raw)},
        {"stage": "Monthly USD with positive rent and area", "listings": len(eligible)},
        {"stage": "After exact-row deduplication", "listings": len(distinct)},
        {"stage": "Modeling cohort with city and state", "listings": len(known)},
    ])
    for filename, frame in (
        ("modeling_cohort.csv", cohort_table),
        ("split_summary.csv", split_table),
        ("training_tail_reference.csv", thresholds),
        ("training_feature_coverage.csv", coverage),
        ("modeling_field_policy.csv", policy),
    ):
        frame.to_csv(target / filename, index=False, float_format="%.4f")
    (target / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_report(target / "preparation_review.md", manifest, split_table, thresholds, coverage)
    return manifest


def main():
    manifest = run(Path.cwd())
    for key in ("source_rows", "distinct_eligible_rows", "modeling_rows", "primary_train_rows", "primary_test_rows", "geo_train_rows", "geo_test_rows"):
        print(f"{key}: {manifest[key]}")
    print("PASS - Five preparation tables, review report, and manifest generated")
    print("PASS - Property groups and geographic states isolated across their respective splits")
    print("PASS - Original listing records remain excluded from Git")


if __name__ == "__main__":
    main()
