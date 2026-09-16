import numpy as np
import pandas as pd
import pytest

from src.features import (SITE_COL, SiteTargetEncoder, add_target_context, base_frame, build,
                          tier_columns, upstream_map)


@pytest.fixture(scope="module")
def df():
    return base_frame()


def test_tiers_are_nested():
    t1, t2, t3 = (set(tier_columns(k)) for k in (1, 2, 3))
    assert t1 < t2 < t3
    assert SITE_COL in t2 and SITE_COL not in t1


def test_upstream_map(df):
    up = upstream_map(df)
    assert up["M5"] == "M4" and up["M18"] == "M17" and "M1" not in up
    assert up["7"] == "6"            # Gumi settled <- Gumi raw
    assert up["12"] == "M9"          # Goryeong settled <- Goryeong raw (M9)
    assert up["18"] == "M14"         # Bansong settled <- Bansong raw (M14)
    assert up["17"] == "18"          # Bansong ozonated <- settled
    assert up["19"] == "17" and up["20"] == "19"
    assert all(s not in up for s in ["T1", "T3", "1", "5"])


def test_lag_features_use_only_earlier_dates(df):
    d = add_target_context(df, "THMFP")
    for site, sub in d.groupby("Site"):
        sub = sub.sort_values("Date")
        # lag1_y at row i equals y at row i-1 (same site, previous month)
        expected = sub["y_THMFP"].shift(1)
        assert np.allclose(sub["lag1_y"].fillna(-99), expected.fillna(-99))
        # hist_mean_y never includes the current or later rows
        for i in range(len(sub)):
            prev = sub["y_THMFP"].iloc[:i].dropna()
            hm = sub["hist_mean_y"].iloc[i]
            if prev.empty:
                assert np.isnan(hm)
            else:
                assert np.isclose(hm, prev.mean())


def test_upstream_features_same_month_other_site(df):
    d = add_target_context(df, "HAAFP")
    row = d[(d.Site == "M5") & (d.Date == "2017-05-01")].iloc[0]
    src = d[(d.Site == "M4") & (d.Date == "2017-05-01")].iloc[0]
    assert np.isclose(row["up_y"], src["y_HAAFP"]) and np.isclose(row["up_log_TOC"], src["log_TOC"])
    assert d.loc[d.Site == "T1", "up_y"].isna().all()


def test_build_shapes():
    for t in ("THMFP", "HAAFP"):
        X, y, meta = build(t, 3)
        assert len(X) == len(y) == len(meta) and y.notna().all()
        assert len(X) >= 680
        assert not X[tier_columns(1)].isna().all(axis=1).any()


def test_site_target_encoder_is_fold_aware():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], SITE_COL: ["s1", "s1", "s2", "s2"]})
    y = np.array([0.0, 1.0, 10.0, 11.0])
    enc = SiteTargetEncoder(m=0.0).fit(X, y)
    out = enc.transform(pd.DataFrame({"a": [0.0, 0.0], SITE_COL: ["s1", "unseen"]}))
    assert SITE_COL not in out.columns
    assert np.isclose(out["site_te"].iloc[0], 0.5)          # mean of s1 in training
    assert np.isclose(out["site_te"].iloc[1], y.mean())      # unseen site -> global mean
    # smoothing pulls toward global mean
    enc2 = SiteTargetEncoder(m=2.0).fit(X, y)
    assert 0.5 < enc2.mapping_["s1"] < y.mean()
