from hashlib import sha256
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED_SOURCE_SHA256 = "6adeae31e5507e39feb03acf811547a5331d2329c5a24ed1b1f1455fc1835916"
SOURCE_NAME = "apartments_for_rent_classified_100K.csv"
SOURCE_DOI = "https://doi.org/10.24432/C5X623"
MIN_CITY_LISTINGS = 30


def file_digest(path):
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_source(root):
    matches = list((root / "data" / "raw").rglob(SOURCE_NAME))
    if len(matches) != 1:
        raise ValueError(f"Expected one {SOURCE_NAME}, found {len(matches)}")
    return matches[0]


def load_listings(path):
    digest = file_digest(path)
    if digest != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Unexpected source SHA256: {digest}")
    data = pd.read_csv(path, sep=";", encoding="cp1252", low_memory=False)
    needed = {
        "id", "price", "price_type", "currency", "square_feet",
        "bedrooms", "bathrooms", "cityname", "state", "amenities", "category"
    }
    absent = needed.difference(data.columns)
    if absent:
        raise ValueError(f"Required source columns missing: {sorted(absent)}")
    return data


def standardized_text(column):
    values = column.astype("string").str.strip()
    return values.mask(values.eq(""))


def eligibility_flags(raw):
    price = pd.to_numeric(raw["price"], errors="coerce")
    area = pd.to_numeric(raw["square_feet"], errors="coerce")
    price_type = standardized_text(raw["price_type"])
    currency = standardized_text(raw["currency"])
    city = standardized_text(raw["cityname"])
    state = standardized_text(raw["state"])

    flags = pd.DataFrame(index=raw.index)
    flags["monthly"] = price_type.str.casefold().eq("monthly").fillna(False)
    flags["usd"] = currency.str.upper().eq("USD").fillna(False)
    flags["positive_price"] = price.gt(0).fillna(False)
    flags["positive_area"] = area.gt(0).fillna(False)
    flags["known_location"] = city.notna() & state.notna()
    return flags


def cohorts(raw):
    flags = eligibility_flags(raw)
    conditions = ["monthly", "usd", "positive_price", "positive_area"]
    eligible = raw.loc[flags[conditions].all(axis=1)].copy()
    distinct = eligible.drop_duplicates().copy()
    known = distinct.loc[eligibility_flags(distinct)["known_location"]].copy()

    for frame in (eligible, distinct, known):
        for column in ("price", "square_feet", "bedrooms", "bathrooms"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["cityname"] = standardized_text(frame["cityname"])
        frame["state"] = standardized_text(frame["state"])

    return flags, eligible, distinct, known


def flow_table(raw, flags, eligible, distinct, known):
    stages = [
        ("All source records", len(raw)),
        ("Monthly price type", int(flags["monthly"].sum())),
        ("Monthly and USD", int(flags[["monthly", "usd"]].all(axis=1).sum())),
        ("Monthly USD with positive asking rent", int(flags[["monthly", "usd", "positive_price"]].all(axis=1).sum())),
        ("Monthly USD with positive rent and area", len(eligible)),
        ("After exact-row deduplication", len(distinct)),
        ("Distinct eligible listings with city and state", len(known)),
    ]
    counts = [count for _, count in stages]
    if counts != sorted(counts, reverse=True):
        raise ValueError("Cohort counts should not increase across selection stages")
    return pd.DataFrame(stages, columns=["stage", "listings"])


def missingness_table(raw):
    result = pd.DataFrame({
        "field": raw.columns,
        "missing_count": [int(raw[col].isna().sum()) for col in raw],
    })
    result["missing_percent"] = (result["missing_count"] / len(raw) * 100).round(3)
    return result.sort_values(
        ["missing_count", "field"], ascending=[False, True]
    ).reset_index(drop=True)


def grouped_rents(data, columns, minimum=1):
    result = (
        data.groupby(columns, dropna=False, observed=True)["price"]
        .agg(
            listings="size",
            median_rent="median",
            mean_rent="mean",
            lower_quartile=lambda s: s.quantile(.25),
            upper_quartile=lambda s: s.quantile(.75)
        )
        .reset_index()
    )
    return result.loc[result["listings"] >= minimum].sort_values(
        ["listings", "median_rent"], ascending=[False, True]
    ).reset_index(drop=True)


def bedroom_group(values):
    numeric = pd.to_numeric(values, errors="coerce")
    labels = np.select(
        [
            numeric.eq(0), numeric.eq(1), numeric.eq(2),
            numeric.eq(3), numeric.eq(4), numeric.ge(5)
        ],
        ["0", "1", "2", "3", "4", "5+"],
        default="Unknown/invalid"
    )
    return pd.Series(labels, index=values.index)


def area_group(values):
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[0, 500, 750, 1000, 1500, float("inf")],
        labels=["1–499", "500–749", "750–999", "1000–1499", "1500+"],
        right=False
    )


def quantile_table(distinct):
    rows = []
    measurements = (
        ("asking_rent_usd_month", distinct["price"]),
        ("floor_area_sqft", distinct["square_feet"]),
        ("rent_usd_per_sqft", distinct["price"] / distinct["square_feet"])
    )

    for label, series in measurements:
        row = {"measurement": label, "listings": int(series.notna().sum())}
        for quantile in (0, .01, .05, .25, .5, .75, .95, .99, 1):
            row[f"p{int(quantile * 100):02d}"] = float(series.quantile(quantile))
        rows.append(row)

    return pd.DataFrame(rows)


def save_table(table, path):
    table.to_csv(path, index=False, float_format="%.4f")


def plot_distribution(values, path, title, x_label):
    top = float(values.quantile(.99))
    if not np.isfinite(top) or top <= 0:
        raise ValueError("Invalid 99th percentile for distribution plot")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(
        values.loc[values.le(top)],
        bins=45,
        color="#335c75",
        edgecolor="white",
        linewidth=.3
    )
    ax.axvline(
        values.median(),
        color="#e89d50",
        linewidth=2,
        label="Full-cohort median"
    )
    ax.set(title=title, xlabel=x_label, ylabel="Listings")
    ax.legend(frameon=False)
    ax.text(
        .99, .95,
        "Display limited to the 99th percentile",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ranked(table, label_column, path, title, count=15):
    subset = table.head(count).copy().sort_values("median_rent")
    if subset.empty:
        raise ValueError(f"No rows available for {title}")

    labels = subset[label_column].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(9, max(5, len(labels) * .38)))
    ax.barh(labels, subset["median_rent"], color="#335c75")
    ax.set(
        title=title,
        xlabel="Median advertised monthly rent (USD)"
    )
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bedrooms(data, path):
    order = ["0", "1", "2", "3", "4", "5+", "Unknown/invalid"]
    grouped = grouped_rents(
        data, ["bedroom_band"], minimum=30
    ).set_index("bedroom_band")

    names = [name for name in order if name in grouped.index]
    if not names:
        raise ValueError("No bedroom groups for chart")

    selected = grouped.loc[names]
    x = np.arange(len(selected))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(
        x,
        selected["median_rent"],
        yerr=[
            selected["median_rent"] - selected["lower_quartile"],
            selected["upper_quartile"] - selected["median_rent"]
        ],
        fmt="o",
        color="#335c75",
        ecolor="#9cb8c9",
        capsize=4
    )
    ax.set(
        xticks=x,
        xticklabels=names,
        ylabel="Monthly rent (USD)",
        xlabel="Number of bedrooms",
        title="Advertised rent by bedroom count (median and IQR)"
    )
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_area_bands(table, path):
    order = ["1–499", "500–749", "750–999", "1000–1499", "1500+"]
    selected = table.set_index("area_band").reindex(order)
    selected = selected.dropna(subset=["median_rent"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        selected.index.astype(str),
        selected["median_rent"],
        color="#335c75"
    )
    ax.set(
        title="Median asking rent by advertised floor area",
        xlabel="Square feet",
        ylabel="Median monthly rent (USD)"
    )
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_joint(data, path):
    xlimit = float(data["square_feet"].quantile(.99))
    ylimit = float(data["price"].quantile(.99))
    subset = data.loc[
        data["square_feet"].le(xlimit) &
        data["price"].le(ylimit)
    ]

    if subset.empty:
        raise ValueError("No values for joint distribution")

    fig, ax = plt.subplots(figsize=(9, 6))
    hb = ax.hexbin(
        subset["square_feet"],
        subset["price"],
        gridsize=48,
        mincnt=1,
        bins="log",
        cmap="Blues"
    )
    fig.colorbar(hb, ax=ax, label="Listings (log color scale)")
    ax.set(
        title="Floor area and asking rent",
        xlabel="Advertised area (square feet)",
        ylabel="Monthly rent (USD)"
    )
    ax.text(
        .02, .98,
        "Both axes display up to the 99th percentile",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_missingness(table, path):
    subset = table.loc[
        table["missing_count"].gt(0)
    ].head(12).sort_values("missing_percent")

    fig, ax = plt.subplots(figsize=(9, max(4, len(subset) * .45)))
    ax.barh(
        subset["field"],
        subset["missing_percent"],
        color="#335c75"
    )
    ax.set(
        title="Missing information in the source listings",
        xlabel="Percent of source rows missing"
    )
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(path, raw, eligible, distinct, known, flow):
    median_rent = float(distinct["price"].median())
    iqr = distinct["price"].quantile([.25, .75]).tolist()
    p99 = float(distinct["price"].quantile(.99))
    flow_markdown = "\n".join(
        ["| Stage | Listings |", "|---|---:|"] +
        [
            f"| {item.stage} | {item.listings:,} |"
            for item in flow.itertuples(index=False)
        ]
    )

    text = f"""# Rental market source assessment

## Source and reproducibility

The analysis uses the historical Apartment for Rent Classified collection from the UCI Machine Learning Repository (2019). Source DOI: [{SOURCE_DOI}]({SOURCE_DOI}). The dataset is distributed under CC BY 4.0 and is credited to its original repository. The original CSV is excluded from version control.

Source CSV SHA-256: `{EXPECTED_SOURCE_SHA256}`.

## Audited populations

{flow_markdown}

The {len(eligible):,} initial eligible records are positive-rent, positive-area, USD-denominated monthly classified listings. Removing exact row duplicates produces {len(distinct):,} provisional distinct observations, {len(known):,} of which include both city and state. These are **exploratory cohorts**, not a finalized model-training dataset. Duplicate removal is based only on exact complete-row identity. No price or area outlier has been removed at this stage.

## Preliminary market portrait

The provisional distinct cohort has median advertised monthly rent ${median_rent:,.0f}, an interquartile range of ${iqr[0]:,.0f}–${iqr[1]:,.0f}, and a 99th percentile of ${p99:,.0f}. These are asking rents, not signed leases or current market prices. Location comparisons are descriptive and not adjusted for unit attributes or geographical sampling imbalances.

The source contains {len(raw):,} rows. The provisional city-and-state cohort covers {known['state'].nunique():,} state codes and {known['cityname'].nunique():,} city names. State-level outputs summarize all represented groups; the public city table shows locations with at least {MIN_CITY_LISTINGS} eligible distinct listings to avoid unstable tiny-group comparisons.

## Interpretation guardrails

- Geographic differences are descriptive and must not be described as causal regional premiums.
- Amenities and pet policy contain substantial missingness. Missing values cannot be equated to absence.
- Records may contain reposts, duplicate units under different IDs, and nonrepresentative source coverage.
- Listing dates describe historical classified records and do not establish present-day rents.
- `price_display` directly restates the outcome and must not be used as a price-prediction feature.
- Figures depicting prices or square footage may restrict displayed ranges to the 99th percentile; those display limits do not remove records from tables.
- Raw addresses, free text, and individual listing records are not exported to the public summary.

## Artifacts

The `tables` folder contains source quality, cohort flow, quantiles, state and city summaries, bedroom and area comparisons, and correlations. The `figures` folder contains eight research charts. All are generated by `python -m market_study.market_evidence` from the original uncommitted CSV.

## Next research decision

Review the upper tails of price and area, duplicate-ID behavior, geographic concentration, and the handling of incomplete listings before freezing the modeling sample. No prediction performance or causal conclusions are claimed in this phase.
"""
    path.write_text(text, encoding="utf-8")


def run(root):
    raw_path = locate_source(root)
    raw = load_listings(raw_path)
    flags, eligible, distinct, known = cohorts(raw)

    flow = flow_table(raw, flags, eligible, distinct, known)
    missing = missingness_table(raw)
    quantiles = quantile_table(distinct)

    state = grouped_rents(known, ["state"])
    city = grouped_rents(
        known, ["state", "cityname"], minimum=MIN_CITY_LISTINGS
    )
    city["market"] = (
        city["cityname"].astype(str) + ", " +
        city["state"].astype(str)
    )

    distinct["bedroom_band"] = bedroom_group(distinct["bedrooms"])
    distinct["area_band"] = area_group(distinct["square_feet"])

    bedroom = grouped_rents(distinct, ["bedroom_band"])
    area = grouped_rents(distinct, ["area_band"])
    category = grouped_rents(distinct, ["category"])

    complete = distinct[
        ["price", "square_feet", "bedrooms", "bathrooms"]
    ]
    associations = (
        complete.corr(method="spearman")["price"]
        .rename("spearman_with_rent")
        .rename_axis("measurement")
        .reset_index()
    )

    evidence = root / "outputs" / "market_research"
    tables = evidence / "tables"
    figures = evidence / "figures"

    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    exports = {
        "cohort_flow.csv": flow,
        "field_missingness.csv": missing,
        "rent_quantiles.csv": quantiles,
        "state_rent_summary.csv": state,
        "city_rent_summary.csv": city.head(30),
        "bedroom_rent_summary.csv": bedroom,
        "floor_area_rent_summary.csv": area,
        "category_rent_summary.csv": category,
        "numeric_associations.csv": associations,
    }

    for name, table in exports.items():
        save_table(table, tables / name)

    plot_distribution(
        distinct["price"],
        figures / "asking_rent_distribution.png",
        "Distribution of advertised monthly rent",
        "Asking rent (USD per month)"
    )
    plot_distribution(
        distinct["price"] / distinct["square_feet"],
        figures / "rent_per_square_foot.png",
        "Distribution of rent per square foot",
        "USD per square foot per month"
    )
    plot_ranked(
        state,
        "state",
        figures / "state_rent_comparison.png",
        "Largest state samples: median advertised rent"
    )
    plot_ranked(
        city,
        "market",
        figures / "city_rent_comparison.png",
        "Largest city samples: median advertised rent",
        count=12
    )
    plot_bedrooms(
        distinct,
        figures / "bedroom_rent_comparison.png"
    )
    plot_area_bands(
        area,
        figures / "area_band_rent_comparison.png"
    )
    plot_joint(
        distinct,
        figures / "rent_vs_floor_area.png"
    )
    plot_missingness(
        missing,
        figures / "missing_information.png"
    )

    write_report(
        evidence / "source_assessment.md",
        raw, eligible, distinct, known, flow
    )

    manifest = {
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_rows": len(raw),
        "eligible_rows": len(eligible),
        "distinct_eligible_rows": len(distinct),
        "distinct_with_known_location": len(known),
        "exact_duplicate_source_rows": int(raw.duplicated().sum()),
        "repeated_source_ids": int(raw["id"].duplicated().sum()),
        "generated_tables": len(exports),
        "generated_figures": 8,
    }

    (evidence / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8"
    )

    return manifest


def main():
    manifest = run(Path.cwd())

    print("PASS - Historical rental source verified")
    print("PASS - Source rows:", manifest["source_rows"])
    print("PASS - Preliminary eligible listings:", manifest["eligible_rows"])
    print("PASS - Distinct preliminary eligible listings:", manifest["distinct_eligible_rows"])
    print("PASS - Distinct listings with known city and state:", manifest["distinct_with_known_location"])
    print("PASS - Research tables:", manifest["generated_tables"])
    print("PASS - Research figures:", manifest["generated_figures"])
    print("PASS - Reports in outputs/market_research")


if __name__ == "__main__":
    main()