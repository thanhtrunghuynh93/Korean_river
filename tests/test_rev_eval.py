"""Withholding tests for the revision engine (plan section 12)."""
import numpy as np
import pytest

from src.features import base_frame
from src.rev_eval import XGB_CONFIGS, design, frame, nested_fold, site_level, splits


@pytest.fixture(scope="module")
def d():
    return frame("THMFP", base_frame())


def _fold(d, scheme="month", k=0):
    return list(splits(scheme, d, 0))[k]


def test_xgb_configs_are_fixed_and_valid():
    assert len(XGB_CONFIGS) == 4
    assert all(2 <= c["max_depth"] <= 5 and 0.6 <= c["subsample"] <= 1.0 for c in XGB_CONFIGS)


def test_site_level_uses_training_rows_only(d):
    _, tr, te = _fold(d)
    a = site_level(d, tr)
    d2 = d.copy(); d2.loc[te, "y"] += 5.0
    assert np.allclose(site_level(d2, tr), a)


@pytest.mark.parametrize("scen", ["S1a", "S1b", "S3b"])
def test_withheld_target_does_not_change_its_prediction_or_selection(d, scen):
    fold, tr, te = _fold(d)
    rows, chosen = nested_fold(d, scen, "month", tr, te, fold, 0)
    d2 = d.copy()
    j = te[0]
    d2.loc[j, "y"] += 3.0                                  # perturb one withheld evaluation target
    rows2, chosen2 = nested_fold(d2, scen, "month", tr, te, fold, 0)
    r1 = rows.set_index("row").loc[j]; r2 = rows2.set_index("row").loc[j]
    for fam in ("ridge", "xgb", "blend", "selected"):
        assert np.isclose(r1[f"yhat_{fam}"], r2[f"yhat_{fam}"])
        assert np.isclose(r1[f"q90_{fam}"], r2[f"q90_{fam}"])
    assert {k: v for k, v in chosen.items() if k.startswith("chosen")} == {k: v for k, v in chosen2.items() if k.startswith("chosen")}


def test_plant_holdout_keeps_intake_with_its_train(d):
    folds = {f: (tr, te) for f, tr, te in splits("plant", d)}
    tr, te = folds["plant_Goryeong"]
    assert set(d.Site.iloc[te]) == {"M9", "12", "13", "14", "15", "16"}
    assert not set(d.Site.iloc[tr]) & set(d.Site.iloc[te])


def test_rolling_origin_is_strictly_forward(d):
    for f, tr, te in splits("rolling", d):
        assert d.Date.iloc[tr].max() < d.Date.iloc[te].min()
