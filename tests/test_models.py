import numpy as np
import pandas as pd
import pytest

from src.features import SITE_COL, base_frame, build, target_col
from src.models import OTHER_Y_COL, ROUND2_MODELS, ChainedRegressor, MixedLMRegressor, OffsetRegressor, make_model


@pytest.fixture(scope="module")
def data():
    df = base_frame()
    X, y, meta = build("THMFP", 2, df)
    X = X.reset_index(drop=True); y = y.reset_index(drop=True); meta = meta.reset_index(drop=True)
    return df, X, y, meta


@pytest.mark.parametrize("name", ROUND2_MODELS)
def test_round2_models_fit_and_predict(data, name):
    df, X, y, meta = data
    X = X.copy()
    if name == "chained":
        X[OTHER_Y_COL] = np.random.default_rng(0).normal(size=len(X))
    tr, te = np.arange(0, 500), np.arange(500, len(X))
    pipe, params = make_model(name, 2, seed=0)
    p = pipe.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    assert p.shape == (len(te),) and np.isfinite(p).all()


def test_gapfill_spec_builds_for_every_round2_model():
    for name in ROUND2_MODELS:
        make_model(name, "t4_all", seed=0)
    with pytest.raises(ValueError):
        make_model("mixed", 1)                                      # tier 1 carries no Site column


def test_chained_ignores_other_target_truth_at_prediction(data):
    df, X, y, meta = data
    X = X.copy(); X[OTHER_Y_COL] = df.loc[build("THMFP", 2, df)[0].index, target_col("HAAFP")].to_numpy()
    pipe, _ = make_model("chained", 2, seed=0)
    pipe.fit(X.iloc[:500], y.iloc[:500])
    te = X.iloc[500:].copy()
    a = pipe.predict(te)
    te[OTHER_Y_COL] = 99.0
    b = pipe.predict(te.drop(columns=[OTHER_Y_COL]))
    assert np.allclose(a, pipe.predict(te)) and np.allclose(a, b)


def test_offset_regressor_adds_offset_back():
    from sklearn.dummy import DummyRegressor
    X = pd.DataFrame({"log_TOC": [0.1, 0.2, 0.3, np.nan], "a": [1, 2, 3, 4]})
    y = np.array([1.1, 1.2, 1.3, 1.25])
    m = OffsetRegressor(DummyRegressor(strategy="mean"), "log_TOC").fit(X, y)
    # dummy learns mean(y - offset) = 1.0 (offset NaN filled by median 0.2 -> 1.05) ...
    expected_const = np.mean(y - X.log_TOC.fillna(0.2).to_numpy())
    assert np.allclose(m.predict(X), expected_const + X.log_TOC.fillna(0.2).to_numpy())


def test_mixed_model_unseen_site_gets_fixed_part_only():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n), SITE_COL: np.repeat(list("abcd"), n // 4)})
    offset = X[SITE_COL].map({"a": -1.0, "b": 0.0, "c": 1.0, "d": 2.0})
    y = 0.5 * X.x + offset + rng.normal(scale=0.05, size=n)
    m = MixedLMRegressor().fit(X, y)
    known = m.predict(pd.DataFrame({"x": [0.0], SITE_COL: ["d"]}))[0]
    unseen = m.predict(pd.DataFrame({"x": [0.0], SITE_COL: ["zz"]}))[0]
    assert known > unseen + 1.0                                     # site d's high intercept is used
    assert abs(unseen - offset.mean()) < 0.3                        # unseen site falls back to the mean level
