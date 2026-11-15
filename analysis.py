# %% Imports and paths
"""NYC Airbnb 2019 listing-price analysis. Run: python analysis.py"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/raw/AB_NYC_2019.csv"
OUT = ROOT / "outputs"


def summarize(df, by):
    return (df.groupby(by, dropna=False)
            .agg(listings=("price", "size"), median_usd=("price", "median"),
                 mean_usd=("price", "mean"))
            .reset_index().round({"median_usd": 2, "mean_usd": 2}))


def chart(data, group, value, title, filename):
    d = data.sort_values(value).dropna(subset=[group])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(d[group].astype(str), d[value])
    ax.set(xlabel="USD" if value == "median_usd" else "Listings", title=title)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / filename, dpi=150)
    plt.close(fig)


def main():
    if not SOURCE.exists():
        raise SystemExit(f"Dataset missing: {SOURCE}\nDownload AB_NYC_2019.csv from the README link.")
    df = pd.read_csv(SOURCE)
    required = {"id", "price", "neighbourhood_group", "neighbourhood", "room_type"}
    if required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    raw_rows = len(df)
    df = df.drop_duplicates(subset="id").copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.loc[df["price"].gt(0)].copy()  # non-positive prices cannot be compared as valid asking prices
    if df.empty:
        raise ValueError("No listings with positive prices")
    for c in ["neighbourhood_group", "neighbourhood", "room_type"]:
        df[c] = df[c].astype("string").str.strip()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    boroughs = summarize(df.dropna(subset=["neighbourhood_group"]), "neighbourhood_group")
    rooms = summarize(df.dropna(subset=["room_type"]), "room_type")
    neighborhoods = summarize(df.dropna(subset=["neighbourhood"]), "neighbourhood")
    neighborhoods = neighborhoods[neighborhoods["listings"] >= 30].sort_values("median_usd", ascending=False)
    top_volume = summarize(df.dropna(subset=["neighbourhood"]), "neighbourhood").nlargest(10, "listings")
    for name, table in [("prices_by_borough", boroughs), ("prices_by_room_type", rooms),
                        ("neighborhoods_at_least_30_listings", neighborhoods), ("top_neighborhoods_by_listings", top_volume)]:
        table.to_csv(OUT / "tables" / f"{name}.csv", index=False)
    chart(boroughs, "neighbourhood_group", "median_usd", "Median asking price by borough", "median_by_borough.png")
    chart(rooms, "room_type", "median_usd", "Median asking price by room type", "median_by_room_type.png")
    chart(top_volume, "neighbourhood", "listings", "Neighborhoods with most listings", "listings_by_neighborhood.png")
    highest = boroughs.sort_values("median_usd", ascending=False).iloc[0]
    report = f"""# Findings — The Rental Price Puzzle

- Source rows: {raw_rows:,}; unique listings with positive prices: {len(df):,}.
- Overall median advertised nightly price: **${df['price'].median():,.2f}**.
- Highest borough median: **{highest['neighbourhood_group']}** (${highest['median_usd']:,.2f}, n={int(highest['listings']):,}).

## Interpretation and next step
This is a **2019 listing snapshot**, not today's market. Advertised prices are not realized bookings. The source's `availability_365` must not be treated as occupancy. Inspect extreme prices and subgroup sample sizes before making claims.
"""
    (OUT / "findings.md").write_text(report, encoding="utf-8")
    print(f"Done. {len(df):,} listings analyzed. Results: {OUT}")


if __name__ == "__main__":
    main()
