import pandas as pd

from market_study.market_evidence import (
    area_group, bedroom_group, cohorts, eligibility_flags,
    flow_table, grouped_rents, missingness_table, quantile_table
)


def sample():
    return pd.DataFrame({
        "id": [1, 1, 2, 3, 4, 5, 6],
        "price": [1200, 1200, 1800, 500, 1000, -50, 2100],
        "price_type": [
            "Monthly", "Monthly", " Monthly ",
            "Weekly", "Monthly", "Monthly", "Monthly"
        ],
        "currency": ["USD", "USD", "USD", "USD", "EUR", "USD", "USD"],
        "square_feet": [600, 600, 900, 500, 700, 400, 0],
        "bedrooms": [1, 1, 2, 0, 3, 2, None],
        "bathrooms": [1, 1, 2, 1, None, 1, None],
        "cityname": [
            "Boston", "Boston", None, "Boston",
            "Boston", "Boston", "Boston"
        ],
        "state": ["MA", "MA", None, "MA", "MA", "MA", "MA"],
        "amenities": [None] * 7,
        "category": ["housing"] * 7,
    })


def test_monthly_positive_usd_and_area_conditions():
    flags = eligibility_flags(sample())
    assert flags[
        ["monthly", "usd", "positive_price", "positive_area"]
    ].all(axis=1).sum() == 3


def test_preserve_location_missing_before_geo_cohort():
    flags, eligible, distinct, known = cohorts(sample())
    assert len(eligible) == 3
    assert len(distinct) == 2
    assert len(known) == 1
    assert not flags["known_location"].iloc[2]


def test_cohort_flow_is_monotone():
    raw = sample()
    flags, eligible, distinct, known = cohorts(raw)
    result = flow_table(raw, flags, eligible, distinct, known)
    assert list(result["listings"]) == sorted(
        result["listings"], reverse=True
    )
    assert result["listings"].iloc[-1] == 1


def test_missingness_does_not_impute():
    result = missingness_table(sample()).set_index("field")
    assert result.loc["amenities", "missing_count"] == 7
    assert result.loc["state", "missing_count"] == 1


def test_bedroom_and_area_categories():
    assert bedroom_group(
        pd.Series([0, 1, 5, None])
    ).tolist() == ["0", "1", "5+", "Unknown/invalid"]

    assert area_group(
        pd.Series([499, 500, 750, 1000, 1500])
    ).astype(str).tolist() == [
        "1–499", "500–749", "750–999", "1000–1499", "1500+"
    ]


def test_grouped_rent_only_shows_sufficient_groups():
    data = pd.DataFrame({
        "state": ["TX", "TX", "MA"],
        "price": [1000, 2000, 5000]
    })
    grouped = grouped_rents(data, ["state"], minimum=2)
    assert grouped["state"].tolist() == ["TX"]
    assert grouped["median_rent"].tolist() == [1500.0]


def test_quantiles_preserve_full_distinct_sample():
    _, _, distinct, _ = cohorts(sample())
    quantiles = quantile_table(distinct).set_index("measurement")
    assert quantiles.loc["asking_rent_usd_month", "listings"] == 2
    assert quantiles.loc["asking_rent_usd_month", "p50"] == 1500