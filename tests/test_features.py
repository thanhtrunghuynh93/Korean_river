import numpy as np
import pandas as pd
import pytest

from src.experiments import EXPERIMENTS, EXPERIMENT_ORDER
from src.features import (BASIN_COLS, CHEM_EXTRA_COLS, SITE_COL, XLAG_COLS, SiteTargetEncoder, add_cross_target,
                          add_target_context, base_frame, build, downstream_map, feature_columns, tier_columns,
                          upstream_map)


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


def test_downstream_map_is_inverse_of_upstream(df):
    up, down = upstream_map(df), downstream_map(df)
    assert down["M4"] == "M5" and down["M17"] == "M18" and "M18" not in down
    assert down["6"] == "7" and down["7"] == "8" and down["10"] == "11" and "11" not in down   # Gumi train
    assert down["18"] == "17" and down["17"] == "19" and down["19"] == "20"                    # Bansong train
    assert down["M9"] == "M10" and down["M14"] == "M15"        # raw intakes keep their river neighbour
    for a, b in down.items():
        assert up[b] == a
    assert all(s not in down for s in ["T1", "T6", "1", "5"])


def test_basin_mean_excludes_own_row_and_other_dates(df):
    d = add_target_context(df, "THMFP")
    river = d[d.group.astype(str) != "treatment"]
    row = river[(river.Site == "M5") & (river.Date == "2017-05-01")].iloc[0]
    others = river[(river.Date == "2017-05-01") & (river.Site != "M5")]["y_THMFP"].dropna()
    assert np.isclose(row["basin_y"], others.mean())
    # treatment rows get the plain river mean of that month (they are not members)
    trow = d[(d.Site == "7") & (d.Date == "2017-05-01")].iloc[0]
    assert np.isclose(trow["basin_y"], river[river.Date == "2017-05-01"]["y_THMFP"].dropna().mean())
    # group mean is leave-one-out within (Date, group)
    grp = d[(d.Date == "2017-05-01") & (d.group.astype(str) == "main_stem") & (d.Site != "M5")]["y_THMFP"].dropna()
    assert np.isclose(row["grp_y"], grp.mean())
    # a different month never leaks in: perturbing another month's values leaves this row unchanged
    d2 = df.copy(); d2.loc[d2.Date == "2017-06-01", "y_THMFP"] += 10
    assert np.isclose(add_target_context(d2, "THMFP").loc[row.name, "basin_y"], row["basin_y"])


def test_plant_raw_and_downstream_features(df):
    d = add_target_context(df, "HAAFP")
    raw = d[(d.Site == "M9") & (d.Date == "2017-05-01")].iloc[0]           # Goryeong raw intake
    stage = d[(d.Site == "13") & (d.Date == "2017-05-01")].iloc[0]          # Goryeong filtered
    assert np.isclose(stage["plant_raw_y"], raw["y_HAAFP"])
    assert np.isnan(raw["plant_raw_y"])                                      # never its own target
    m4 = d[(d.Site == "M4") & (d.Date == "2017-05-01")].iloc[0]
    m5 = d[(d.Site == "M5") & (d.Date == "2017-05-01")].iloc[0]
    assert np.isclose(m4["down_y"], m5["y_HAAFP"]) and np.isclose(m4["down_log_TOC"], m5["log_TOC"])
    # upstream site's lag-1: M5 in June sees M4's May value
    assert np.isclose(d[(d.Site == "M5") & (d.Date == "2017-06-01")].iloc[0]["up_lag1_y"], m4["y_HAAFP"])
    assert np.isnan(m5["up_lag1_y"])                                         # first month has no lag


def test_cross_target_lags_use_previous_month_only(df):
    d = add_cross_target(add_target_context(df, "THMFP"), "THMFP")
    sub = d[d.Site == "M3"].sort_values("Date")
    assert np.allclose(sub["lag1_y_other"].fillna(-99), sub["y_HAAFP"].shift(1).fillna(-99))
    assert np.allclose(sub["lag2_y"].fillna(-99), sub["y_THMFP"].shift(2).fillna(-99))


def test_experiment_columns_exist_and_extend_tier3():
    t3 = set(tier_columns(3))
    for name in EXPERIMENT_ORDER:
        assert t3 <= set(EXPERIMENTS[name])
        X, y, _ = build("HAAFP", name)
        assert list(X.columns) == feature_columns(name) and len(X) == len(y)
    assert set(EXPERIMENTS["t4_all"]) == t3 | set(CHEM_EXTRA_COLS) | set(XLAG_COLS) | set(BASIN_COLS)


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
