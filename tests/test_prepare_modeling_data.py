import json

import pandas as pd
import pytest

from market_study.prepare_modeling_data import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_MODEL_FIELDS,
    GROUP_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    assigned_holdout,
    assignments_digest,
    canonicalize_group_value,
    feature_coverage,
    feature_policy,
    group_fingerprints,
    make_partitions,
    summarize_partitions,
    training_tail_reference,
)


def records():
    return pd.DataFrame({
        "id": list(range(100)),
        "price": [750 + i * 15 for i in range(100)],
        "square_feet": [450 + i * 10 for i in range(100)],
        "bedrooms": [i % 4 for i in range(100)],
        "bathrooms": [1 + i % 2 for i in range(100)],
        "state": [f"S{i % 20:02d}" for i in range(100)],
        "cityname": [f"City{i % 20:02d}" for i in range(100)],
        "category": ["housing/rent/apartment"] * 100,
        "latitude": [40 + i / 500 for i in range(100)],
        "longitude": [-70 - i / 500 for i in range(100)],
        "has_photo": ["yes"] * 100,
        "fee": ["no"] * 100,
        "pets_allowed": [None if i % 5 else "cats" for i in range(100)],
    })


def test_feature_policy_blocks_target_and_sensitive_fields():
    frame = records()
    policy = feature_policy(frame.columns)
    selected = set(policy.loc[policy["decision"].eq("permitted_predictor"), "source_field"])
    assert selected == set(MODEL_FEATURES)
    assert not selected & FORBIDDEN_MODEL_FIELDS
    assert "price" not in selected
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_missing_model_predictor_is_rejected():
    with pytest.raises(ValueError, match="Missing candidate"):
        feature_policy(records().drop(columns=["fee"]).columns)


def test_canonicalization_is_case_and_whitespace_stable():
    assert canonicalize_group_value(" Boston ") == canonicalize_group_value("boston")
    assert canonicalize_group_value(None) == "<missing>"
    assert canonicalize_group_value(100.0) == canonicalize_group_value(100)


def test_grouping_excludes_price_and_identifiers():
    frame = records()
    modified = frame.copy()
    modified.loc[0, "price"] = 999999
    modified.loc[0, "id"] = -100
    assert "price" not in GROUP_COLUMNS
    assert "id" not in GROUP_COLUMNS
    assert group_fingerprints(frame).tolist() == group_fingerprints(modified).tolist()


def test_same_property_specifications_cannot_cross_primary_holdout():
    frame = records()
    duplicate_spec = frame.loc[[0]].copy()
    duplicate_spec.index = [101]
    duplicate_spec.loc[101, "id"] = 101
    duplicate_spec.loc[101, "price"] = 2900
    data = pd.concat([frame, duplicate_spec])
    partition = make_partitions(data)
    assert partition.loc[0, "property_group"] == partition.loc[101, "property_group"]
    assert partition.loc[0, "primary_partition"] == partition.loc[101, "primary_partition"]
    for key in ("primary_partition", "geographic_partition"):
        assert set(partition[key]) == {"train", "test"}


def test_geographic_holdout_has_no_state_overlap():
    partition = make_partitions(records())
    train = set(partition.loc[partition["geographic_partition"].eq("train"), "state"])
    test = set(partition.loc[partition["geographic_partition"].eq("test"), "state"])
    assert train.isdisjoint(test)


def test_assignments_are_reproducible_independent_of_row_order():
    frame = records()
    first = make_partitions(frame)
    reordered = frame.sample(frac=1.0, random_state=7)
    second = make_partitions(reordered)
    assert assignments_digest(first) == assignments_digest(second)
    pd.testing.assert_frame_equal(first.sort_index(), second.sort_index())


def test_seed_and_rate_control_holdout():
    keys = pd.Series([f"group-{i}" for i in range(1000)])
    results = assigned_holdout(keys)
    assert results.dtype == bool
    assert 0.1 < results.mean() < 0.3
    assert assigned_holdout(keys).equals(results)
    with pytest.raises(ValueError, match="fraction"):
        assigned_holdout(keys, fraction=1)


def test_training_thresholds_are_training_only_and_do_not_change_on_test():
    train = records().iloc[:40]
    tails = training_tail_reference(train)
    mutated_test = records().iloc[40:].copy()
    mutated_test.loc[40, "price"] = 99999999
    assert tails.loc[tails["measure"].eq("asking_rent_usd"), "train_p99"].iloc[0] < 2000
    assert not tails["applied_in_primary_model"].any()


def test_training_missing_categories_are_not_changed_to_no():
    train = records()
    coverage = feature_coverage(train).set_index("feature")
    assert coverage.loc["pets_allowed", "train_missing"] == 80
    assert coverage.loc["square_feet", "train_missing"] == 0


def test_summary_reconciles_each_holdout():
    data = records()
    summary = summarize_partitions(data, make_partitions(data))
    assert len(summary) == 4
    assert summary.groupby("strategy")["listings"].sum().eq(len(data)).all()
    assert summary.groupby("strategy")["share_percent"].sum().between(99.99, 100.01).all()


def test_rejects_duplicate_identifiers_and_indices():
    frame = records()
    frame.loc[1, "id"] = 0
    with pytest.raises(ValueError, match="Repeated listing IDs"):
        make_partitions(frame)
    frame = records()
    frame.index = [0] * len(frame)
    with pytest.raises(ValueError, match="index"):
        make_partitions(frame)
