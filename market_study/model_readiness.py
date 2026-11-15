from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from market_study.market_evidence import cohorts, load_listings, locate_source, EXPECTED_SOURCE_SHA256


MODEL_FEATURES = (
    "square_feet", "bedrooms", "bathrooms", "category", "state",
    "cityname", "has_photo", "fee", "pets_allowed"
)
LEAKAGE_FIELDS = (
    "price", "price_display", "id", "title", "body", "address",
    "rent_per_square_foot"
)


def distribution_extremes(frame):
    measures = {
        "monthly_asking_rent_usd": frame["price"],
        "floor_area_square_feet": frame["square_feet"],
        "monthly_rent_per_square_foot": frame["price"] / frame["square_feet"],
    }
    rows = []
    for name, series in measures.items():
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            raise ValueError(f"No values available for {name}")
        low, high = values.quantile([.01, .99])
        central = values.loc[values.between(low, high, inclusive="both")]
        rows.append({
            "measurement": name,
            "listings": len(values),
            "minimum": float(values.min()),
            "p01": float(low),
            "median": float(values.median()),
            "p99": float(high),
            "maximum": float(values.max()),
            "above_p99": int(values.gt(high).sum()),
            "below_p01": int(values.lt(low).sum()),
            "untrimmed_mean": float(values.mean()),
            "central_98_percent_mean": float(central.mean()),
        })
    return pd.DataFrame(rows)


def coverage(frame):
    fields = [
        "cityname", "state", "bedrooms", "bathrooms", "amenities",
        "pets_allowed", "latitude", "longitude", "category", "has_photo", "fee"
    ]
    rows = []
    for field in fields:
        if field not in frame:
            continue
        values = frame[field]
        if pd.api.types.is_string_dtype(values) or values.dtype == object:
            values = values.astype("string").str.strip().replace("", pd.NA)
        missing = int(values.isna().sum())
        rows.append({
            "field": field,
            "listings": len(frame),
            "missing_count": missing,
            "missing_percent": round(100 * missing / len(frame), 3),
            "distinct_nonmissing_values": int(values.nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values("missing_count", ascending=False)


def geography(known):
    states = known.groupby("state", observed=True).size().rename("listings").reset_index()
    states = states.sort_values(["listings", "state"], ascending=[False, True]).reset_index(drop=True)
    states["share_percent"] = (100 * states["listings"] / len(known)).round(3)
    markets = known.groupby(["state", "cityname"], observed=True).size().rename("listings").reset_index()
    markets = markets.sort_values(["listings", "state", "cityname"], ascending=[False, True, True]).reset_index(drop=True)
    markets["share_percent"] = (100 * markets["listings"] / len(known)).round(3)
    return states, markets


def spec_collision_counts(known):
    key = ["state", "cityname", "price", "square_feet", "bedrooms", "bathrooms"]
    groups = known.groupby(key, dropna=False, observed=True).size()
    repeated = groups.loc[groups.gt(1)]
    return {
        "same_spec_groups": int(len(repeated)),
        "records_in_same_spec_groups": int(repeated.sum()),
        "excess_records_at_matching_specs": int((repeated - 1).sum()),
    }


def listing_time(raw):
    if "time" not in raw:
        return {"timestamp_column_present": False}
    number = pd.to_numeric(raw["time"], errors="coerce")
    times = pd.to_datetime(number, unit="s", errors="coerce", utc=True)
    valid = times.dropna()
    return {
        "timestamp_column_present": True,
        "valid_timestamps": int(len(valid)),
        "invalid_or_missing_timestamps": int(times.isna().sum()),
        "earliest_utc": valid.min().isoformat() if len(valid) else None,
        "latest_utc": valid.max().isoformat() if len(valid) else None,
        "counts_by_year": {str(k): int(v) for k, v in valid.dt.year.value_counts().sort_index().items()},
    }


def write_research_note(path, totals, extreme, state_table, missing_table, clock):
    rent = extreme.set_index("measurement").loc["monthly_asking_rent_usd"]
    area = extreme.set_index("measurement").loc["floor_area_square_feet"]
    ppsf = extreme.set_index("measurement").loc["monthly_rent_per_square_foot"]
    concentration = float(state_table.head(5)["share_percent"].sum())
    text = f"""# Modeling readiness — historical rental classifieds

## Review scope

The source is the UCI Apartment for Rent Classified collection (DOI: https://doi.org/10.24432/C5X623; CC BY 4.0). These are historic advertised rents, not current rents or executed leases. This checkpoint audits samples and candidate features without training or validating a model.

## Cohort reconciliation

- Original records: {totals['source_rows']:,}
- Monthly USD listings with positive rent and area: {totals['eligible_records']:,}
- After removing exact complete-row duplicates: {totals['distinct_eligible_records']:,}
- Distinct eligible listings with city and state: {totals['known_location_records']:,}
- Repeated IDs after exact-row removal: {totals['remaining_repeated_ids']:,}

## Distribution and market coverage

- Monthly asking rent: median ${rent['median']:,.0f}; 99th percentile ${rent['p99']:,.0f}; maximum ${rent['maximum']:,.0f}.
- Advertised floor area: median {area['median']:,.0f} sq ft; 99th percentile {area['p99']:,.0f} sq ft; maximum {area['maximum']:,.0f} sq ft.
- Rent per square foot: median ${ppsf['median']:,.2f}; 99th percentile ${ppsf['p99']:,.2f}; maximum ${ppsf['maximum']:,.2f}.
- Five highest-volume state codes account for {concentration:.1f}% of records with known city and state. This reflects source coverage, not a population-weighted national rental market.
- Listing timestamps: {clock['earliest_utc'] if clock.get('timestamp_column_present') else 'Not provided'} to {clock['latest_utc'] if clock.get('timestamp_column_present') else 'Not provided'}.

## Decisions before regression

1. Retain the original positive-price and positive-area eligible population for descriptive evidence, documenting exact-row deduplication. The model cohort and any additional exclusions must be frozen before model evaluation.
2. Inspect extreme high-rent, large-area, and high rent-per-square-foot records. Quantile screening is diagnostic only: it must not silently erase rare units. Compare model errors on all cases and central-range subsets.
3. Preserve a missing category for pets and amenities; unknown is not equivalent to absence.
4. For a location-augmented model, compare an ordinary holdout with a separate geography-out holdout. Keep transformations and feature selection within the training data.
5. Exclude `price_display`, price-per-square-foot, ID, text descriptions, addresses, and any target-derived information from predictors. City and state can be evaluated as location features, but location shift and source imbalance must be acknowledged.
6. Interpret all statistical relationships as observational associations, not causal price effects. The source is not a representative census of available rentals.

## Review artifacts

Tables contain tail diagnostics, missing-field coverage, state and city concentration, and a candidate feature policy. Figures visualize geographic sample imbalance and the rent and floor-area upper tails. No raw listing rows or addresses are exported.
"""
    path.write_text(text, encoding="utf-8")


def build_feature_policy(raw):
    rows = []
    for field in MODEL_FEATURES:
        rows.append({"field": field, "role": "candidate_predictor", "present": field in raw, "restriction": "Training-only transformations; handle missing values explicitly"})
    for field in LEAKAGE_FIELDS:
        rows.append({"field": field, "role": "excluded_predictor", "present": field in raw, "restriction": "Target leakage, identifier, unreviewed text, or sensitive address detail"})
    return pd.DataFrame(rows)


def draw_state_chart(states, path):
    subset = states.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(subset["state"], subset["share_percent"], color="#335c75")
    ax.set(title="Where the historical listing sample is concentrated", xlabel="Share of listings with known city and state (%)")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_tails(data, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, measure, title, label in (
        (axes[0], data["price"], "Asking rent", "Monthly USD"),
        (axes[1], data["square_feet"], "Floor area", "Square feet"),
    ):
        top = float(measure.quantile(.999))
        values = measure.loc[measure.le(top)]
        ax.hist(values, bins=55, color="#335c75")
        ax.axvline(float(measure.quantile(.99)), color="#d88b45", linewidth=2, label="Full-cohort 99th percentile")
        ax.set(title=title, xlabel=label, ylabel="Listings")
        ax.legend(frameon=False)
        ax.text(.98, .95, "Display limited to p99.9", transform=ax.transAxes, va="top", ha="right", fontsize=8)
    fig.suptitle("Upper-tail screening of historical rental listings")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(root):
    raw = load_listings(locate_source(root))
    _, eligible, distinct, known = cohorts(raw)
    extremes = distribution_extremes(distinct)
    missing = coverage(distinct)
    states, markets = geography(known)
    fingerprints = spec_collision_counts(known)
    time_info = listing_time(raw)
    policy = build_feature_policy(raw)
    target = root / "outputs" / "market_research" / "readiness"
    target.mkdir(parents=True, exist_ok=True)
    tables = {
        "tail_screening.csv": extremes,
        "feature_coverage.csv": missing,
        "state_listing_concentration.csv": states,
        "city_listing_concentration.csv": markets.head(40),
        "model_feature_policy.csv": policy,
    }
    for name, frame in tables.items():
        frame.to_csv(target / name, index=False, float_format="%.4f")
    draw_state_chart(states, target / "state_sample_concentration.png")
    draw_tails(distinct, target / "rent_area_tail_diagnostics.png")
    totals = {
        "source_rows": len(raw),
        "eligible_records": len(eligible),
        "distinct_eligible_records": len(distinct),
        "known_location_records": len(known),
        "remaining_repeated_ids": int(distinct["id"].duplicated().sum()),
    }
    write_research_note(target / "model_readiness.md", totals, extremes, states, missing, time_info)
    manifest = {
        "source_sha256": EXPECTED_SOURCE_SHA256,
        **totals,
        **fingerprints,
        "top_five_state_sample_share_percent": round(float(states.head(5)["share_percent"].sum()), 3),
        "listing_dates": time_info,
        "tables": list(tables),
        "charts": ["state_sample_concentration.png", "rent_area_tail_diagnostics.png"],
    }
    (target / "readiness_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    result = run(Path.cwd())
    for key in (
        "source_rows", "distinct_eligible_records", "known_location_records", "remaining_repeated_ids",
        "same_spec_groups", "records_in_same_spec_groups", "top_five_state_sample_share_percent"
    ):
        print(f"{key}: {result[key]}")
    print("PASS - Five diagnostic tables, two figures, and readiness note generated")


if __name__ == "__main__":
    main()
