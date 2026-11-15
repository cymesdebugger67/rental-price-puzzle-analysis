import numpy as np
import pandas as pd
import pytest

from market_study.statistical_evidence import (
    bedroom_contrasts, bootstrap_median, safe_spearman, state_summary
)


def sample():
    return pd.DataFrame({
        "price": [1000.0 + (i % 30) * 40 + (i % 2) * 300 for i in range(300)],
        "square_feet": [500 + 20 * (i % 30) for i in range(300)],
        "bedrooms": [1 if i % 2 else 2 for i in range(300)],
        "bathrooms": [1 + i % 2 for i in range(300)],
        "state": ["TX" if i % 3 else "CA" for i in range(300)],
    })


def test_bootstrap_seed_is_reproducible():
    values = np.arange(100, dtype=float)
    first = bootstrap_median(values, np.random.default_rng(7), 40)
    second = bootstrap_median(values, np.random.default_rng(7), 40)
    assert first[:3] == second[:3]
    np.testing.assert_array_equal(first[3], second[3])


def test_bootstrap_requires_data_and_replicates():
    with pytest.raises(ValueError, match="two finite"):
        bootstrap_median([1], np.random.default_rng(1))
    with pytest.raises(ValueError, match="20 bootstrap"):
        bootstrap_median([1, 2, 3], np.random.default_rng(1), 10)


def test_bedroom_contrast_is_positive_and_seeded():
    first = bedroom_contrasts(sample(), seed=15, replicates=40)
    second = bedroom_contrasts(sample(), seed=15, replicates=40)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 2
    assert first["listings"].tolist() == [150, 150]
    assert first.iloc[1]["contrast_two_minus_one"] < 0


def test_rank_correlation_and_state_reconciliation():
    frame = sample()
    corr = safe_spearman(frame)
    assert set(corr["attribute"]) == {"square_feet", "bedrooms", "bathrooms"}
    assert corr["observations"].eq(300).all()
    states = state_summary(frame)
    assert states["listings"].sum() == 300
    assert states["train_sample_percent"].sum() == pytest.approx(100)
