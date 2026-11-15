# The Rental Price Puzzle

**Domain:** Real Estate / Travel
**Level:** Intermediate / medium-size analytics project
**Data status:** Public dataset referenced; **data not bundled**.

## Business question

How did Airbnb listing prices differ by borough and room type in NYC in 2019?

## Scope (keep it manageable)

- Median listing price by borough
- Median price by room type
- Neighborhood price comparisons with minimum samples
- Listing distribution

**Done means:** One clean run, at least three charts, summary tables, a factual findings report, and a README updated with your own conclusions. No dashboard, predictive modeling, or web scraping required.

## Get the dataset

1. Visit the [dataset page](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data).
2. Download the source data and extract it if zipped.
3. Place **`AB_NYC_2019.csv`** at **`data/raw/AB_NYC_2019.csv`** exactly (case-sensitive on some systems).
4. Review the original dataset's attribution and redistribution terms. Raw data is intentionally excluded from Git.

## Run in VS Code

Open **this project folder** in VS Code. In the VS Code terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python analysis.py
```

On macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`. If PowerShell activation is blocked, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe analysis.py`.

The script checks the expected columns, handles selected missing/invalid values, computes summaries, and writes results under `outputs/`. Open `outputs/findings.md` first, then the generated `outputs/tables/*.csv` and `outputs/charts/*.png`. Outputs are ignored by Git by default; commit **selected, reviewed** figures or findings only if you wish to publish them.

## Evidence / caveats

2019 snapshot, not present-day prices. Listing prices are advertised, not booked prices; availability_365 does not measure occupancy.

Do not present automatically generated descriptive metrics as causal findings, and do not invent numbers if the input is missing.

## Portfolio completion checklist

- [ ] Data source and limitations documented
- [ ] Script runs with original dataset
- [ ] Three useful visualizations reviewed
- [ ] Findings rewritten in your own language with numbers from actual output
- [ ] GitHub README updated with 2–3 clear insights and a suggested next step
- [ ] Personal credit added before publishing (your chosen public display name)
