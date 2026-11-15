from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

from market_study.market_evidence import EXPECTED_SOURCE_SHA256, cohorts, load_listings, locate_source
from market_study.prepare_modeling_data import make_partitions, assignments_digest

SEED = 2026
BOOTSTRAP_REPLICATES = 300


def primary_training(root):
    raw = load_listings(locate_source(root))
    _, _, _, known = cohorts(raw)
    parts = make_partitions(known)
    train_mask = parts["primary_partition"].eq("train")
    train = known.loc[train_mask].copy()
    groups = parts.loc[train_mask, "property_group"].copy()
    if train.empty or parts.loc[~train_mask, "property_group"].isin(groups).any():
        raise ValueError("Primary training partition is invalid")
    if len(known) != 99101 or len(train) != 79196:
        raise ValueError("Actual analytical cohort differs from the frozen Phase 3 design")
    signature = assignments_digest(parts)
    expected = "eed0e59e56f4c457841a882f10af66d3828782b131f829b9e5c1627cfb3e135d"
    if signature != expected:
        raise ValueError("Frozen assignment fingerprint mismatch")
    saved_path = root / "outputs" / "market_research" / "preparation" / "preparation_manifest.json"
    if not saved_path.exists():
        raise ValueError("Rebuild the Phase 3 preparation manifest before Phase 4")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if saved.get("assignment_sha256") != signature or saved.get("modeling_rows") != len(known):
        raise ValueError("Phase 3 manifest does not match the frozen partition")
    return train, groups, signature


def safe_spearman(frame):
    rows = []
    for field in ("square_feet", "bedrooms", "bathrooms"):
        a = pd.to_numeric(frame[field], errors="coerce")
        b = pd.to_numeric(frame["price"], errors="coerce")
        valid = a.notna() & b.notna()
        if valid.sum() < 3 or a.loc[valid].nunique() < 2 or b.loc[valid].nunique() < 2:
            coefficient, p_value = float("nan"), float("nan")
        else:
            result = spearmanr(a.loc[valid].to_numpy(), b.loc[valid].to_numpy())
            coefficient, p_value = float(result.statistic), float(result.pvalue)
        rows.append({"attribute": field, "observations": int(valid.sum()), "spearman_rho": coefficient, "exploratory_p_value": p_value})
    return pd.DataFrame(rows)


def bootstrap_median(values, rng, replicates=BOOTSTRAP_REPLICATES):
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        raise ValueError("At least two finite observations required")
    if replicates < 20:
        raise ValueError("At least 20 bootstrap replicates required")
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        estimates[index] = np.median(data[rng.integers(0, len(data), size=len(data))])
    return float(np.median(data)), float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975)), estimates


def bedroom_contrasts(frame, seed=SEED, replicates=BOOTSTRAP_REPLICATES):
    rng = np.random.default_rng(seed)
    one = pd.to_numeric(frame.loc[frame["bedrooms"].eq(1), "price"], errors="coerce").dropna().to_numpy()
    two = pd.to_numeric(frame.loc[frame["bedrooms"].eq(2), "price"], errors="coerce").dropna().to_numpy()
    if len(one) < 30 or len(two) < 30:
        raise ValueError("Bedroom comparison requires at least 30 observations per category")
    median_one, low_one, high_one, reps_one = bootstrap_median(one, rng, replicates)
    median_two, low_two, high_two, reps_two = bootstrap_median(two, rng, replicates)
    differences = reps_two - reps_one
    test = mannwhitneyu(two, one, alternative="two-sided", method="asymptotic")
    rows = [
        {"bedroom_group": "1 bedroom", "listings": len(one), "median_rent": median_one, "bootstrap_ci_low": low_one, "bootstrap_ci_high": high_one, "contrast_two_minus_one": float("nan"), "contrast_ci_low": float("nan"), "contrast_ci_high": float("nan"), "exploratory_mann_whitney_p_value": float("nan")},
        {"bedroom_group": "2 bedrooms", "listings": len(two), "median_rent": median_two, "bootstrap_ci_low": low_two, "bootstrap_ci_high": high_two, "contrast_two_minus_one": median_two - median_one, "contrast_ci_low": float(np.quantile(differences, .025)), "contrast_ci_high": float(np.quantile(differences, .975)), "exploratory_mann_whitney_p_value": float(test.pvalue)},
    ]
    return pd.DataFrame(rows)


def state_summary(frame):
    result = frame.groupby("state", observed=True)["price"].agg(listings="size", median_asking_rent="median").reset_index()
    result = result.sort_values(["listings", "state"], ascending=[False, True]).reset_index(drop=True)
    result["train_sample_percent"] = 100 * result["listings"] / len(frame)
    return result


def plot_bedroom_comparison(contrasts, path):
    centers = contrasts["median_rent"].to_numpy(dtype=float)
    lows = contrasts["bootstrap_ci_low"].to_numpy(dtype=float)
    highs = contrasts["bootstrap_ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(range(len(centers)), centers, yerr=[centers - lows, highs - centers], fmt="o", markersize=8, capsize=5, color="#335c75")
    ax.set(xticks=list(range(len(centers))), xticklabels=contrasts["bedroom_group"], ylabel="Monthly asking rent (USD)", title="Training listings: unadjusted bedroom-group medians")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_state_samples(states, path):
    sample = states.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(sample["state"], sample["train_sample_percent"], color="#335c75")
    ax.set(xlabel="Share of primary training records (%)", title="Geographic concentration in the primary training sample")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(root):
    train, groups, signature = primary_training(root)
    correlations = safe_spearman(train)
    contrast = bedroom_contrasts(train)
    states = state_summary(train)
    target = root / "outputs" / "market_research" / "statistics"
    target.mkdir(parents=True, exist_ok=True)
    for filename, table in (("training_correlations.csv", correlations), ("bedroom_median_contrasts.csv", contrast), ("training_state_composition.csv", states)):
        table.to_csv(target / filename, index=False, float_format="%.6g")
    plot_bedroom_comparison(contrast, target / "bedroom_median_intervals.png")
    plot_state_samples(states, target / "training_state_distribution.png")
    mean_rho = correlations.set_index("attribute").loc["square_feet", "spearman_rho"]
    difference = contrast.iloc[1]
    manifest = {
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "assignment_sha256": signature,
        "sample": "primary_train_only",
        "training_rows": len(train),
        "distinct_property_groups": int(groups.nunique()),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "tables": ["training_correlations.csv", "bedroom_median_contrasts.csv", "training_state_composition.csv"],
        "figures": ["bedroom_median_intervals.png", "training_state_distribution.png"],
    }
    report = f"""# Training-only statistical exploration

This exploratory assessment uses **{len(train):,}** training listings from the fixed property-group holdout. All comparisons are descriptive and unadjusted, not causal estimates. The remaining **19,905** listings are excluded from statistical exploration here, protecting the primary held-out evaluation. Source: UCI Apartment for Rent Classified, DOI https://doi.org/10.24432/C5X623 (CC BY 4.0).

- Spearman's rank correlation of floor area and advertised rent in training: **{mean_rho:.3f}**. Rank correlation does not isolate the effects of location or property type.
- One-bedroom median advertised rent: **${contrast.iloc[0]['median_rent']:,.0f}**; two-bedroom median: **${difference['median_rent']:,.0f}**.
- Unadjusted two-minus-one bedroom median difference: **${difference['contrast_two_minus_one']:,.0f}**, with an approximate percentile bootstrap 95% interval of **${difference['contrast_ci_low']:,.0f} to ${difference['contrast_ci_high']:,.0f}** ({BOOTSTRAP_REPLICATES} seeded replicates).
- A Mann–Whitney comparison and Spearman p-values are included in the associated tables as exploratory evidence, not as confirmation of a causal rental premium. With a large observational sample, very small p-values may correspond to modest practical differences.

The bootstrap intervals are listing-level descriptive intervals and do **not** account for the dependence of listings within property-specification groups, reposted listings, sampling bias, or multiple testing. They should not be treated as nationally representative uncertainty intervals. Property attributes, location, and listing coverage are confounded. Do not use these training-only association results to adjust model choices after viewing the held-out results.

The statistical figures present the observed training sample. They have not been geographically or compositionally weighted to represent all US apartments.
"""
    (target / "statistical_note.md").write_text(report, encoding="utf-8")
    (target / "statistics_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    report = run(Path.cwd())
    print("PASS - Training-only statistical exploration completed")
    print("Training listings:", report["training_rows"])
    print("Training property groups:", report["distinct_property_groups"])
    print("PASS - Three tables, two figures, and statistical note generated")


if __name__ == "__main__":
    main()
