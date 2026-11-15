import pandas as pd

from market_study.model_readiness import (
    distribution_extremes, coverage, geography,
    spec_collision_counts, listing_time, build_feature_policy
)


def sample():
    return pd.DataFrame({
        "id": [1, 2, 3, 4], "price": [800, 1000, 1000, 6000],
        "square_feet": [500, 500, 500, 1000],
        "bedrooms": [1, 1, 1, 3], "bathrooms": [1, 1, 1, 2],
        "cityname": ["A", "A", "A", "B"], "state": ["MA", "MA", "MA", "TX"],
        "amenities": [None, None, "Pool", "Gym"],
        "pets_allowed": [None] * 4, "category": ["Housing"] * 4,
        "time": [1570000000] * 4,
    })


def test_quantile_audit_does_not_discard_rows():
    result = distribution_extremes(sample())
    assert len(result) == 3
    assert result["listings"].eq(4).all()
    assert result.loc[result["measurement"].eq("monthly_asking_rent_usd"), "maximum"].iloc[0] == 6000


def test_missing_not_treated_as_negative():
    result = coverage(sample()).set_index("field")
    assert result.loc["pets_allowed", "missing_count"] == 4
    assert result.loc["amenities", "missing_count"] == 2


def test_location_totals_reconcile():
    states, cities = geography(sample())
    assert states["listings"].sum() == 4
    assert cities["listings"].sum() == 4
    assert states.iloc[0]["state"] == "MA"


def test_same_specs_not_removed_as_duplicates():
    result = spec_collision_counts(sample())
    assert result["same_spec_groups"] == 1
    assert result["records_in_same_spec_groups"] == 2


def test_time_summaries_are_utc():
    result = listing_time(sample())
    assert result["counts_by_year"] == {"2019": 4}
    assert result["invalid_or_missing_timestamps"] == 0


def test_target_leakage_exclusions():
    policy = build_feature_policy(sample()).set_index("field")
    assert policy.loc["price", "role"] == "excluded_predictor"
    assert policy.loc["price_display", "role"] == "excluded_predictor"
    assert policy.loc["square_feet", "role"] == "candidate_predictor"
