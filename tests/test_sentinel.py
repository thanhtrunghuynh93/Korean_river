import numpy as np
import pytest

from src.features import base_frame
from src.sentinel import _score, candidate_sites, greedy_select, prepare, sentinel_features


@pytest.fixture(scope="module")
def d():
    return prepare("THMFP", base_frame())


def _split(d, test_dates=("2018-06-01", "2018-07-01")):
    is_te = d.Date.astype(str).str[:10].isin(test_dates).to_numpy()
    idx = np.arange(len(d))
    return idx[~is_te], idx[is_te]


def test_non_sentinel_targets_never_enter_features(d):
    tr, te = _split(d)
    sent = ["M5", "M14"]
    base = sentinel_features(d, tr, sent)
    d2 = d.copy()
    ns_te = te[~d.Site.iloc[te].isin(sent).to_numpy()]
    d2.loc[ns_te, "y"] += 5.0; d2.loc[ns_te, "y_other"] += 5.0
    assert np.allclose(sentinel_features(d2, tr, sent).fillna(-99), base.fillna(-99))


def test_site_means_use_training_rows_only(d):
    tr, te = _split(d)
    sent = ["M5"]
    base = sentinel_features(d, tr, sent)
    d2 = d.copy(); d2.loc[te, "y"] += 5.0
    f2 = sentinel_features(d2, tr, sent)
    assert np.allclose(f2.site_mean, base.site_mean)
    # sentinel anomalies on test dates DO change: the sentinel is measured that month by design
    m5_june = (d.Date.astype(str).str[:10] == "2018-06-01").to_numpy()
    assert not np.allclose(f2.sent_anom[m5_june], base.sent_anom[m5_june])


def test_sentinel_anomaly_is_same_month_mean(d):
    tr, _ = _split(d)
    sent = ["M5", "M14"]
    f = sentinel_features(d, tr, sent)
    row = d.index[(d.Site == "M10") & (d.Date == "2017-08-01")][0]
    sm = d.iloc[tr].groupby("Site").y.mean()
    s_rows = d[(d.Site.isin(sent)) & (d.Date == "2017-08-01")]
    expected = (s_rows.y - s_rows.Site.map(sm)).mean()
    assert np.isclose(f.at[row, "sent_anom"], expected)
    assert f.at[row, "sent_near_dist"] == 4                     # M10: nearest of M5 (5 away) / M14 (4 away)


def test_evaluation_excludes_sentinel_rows(d):
    tr, te = _split(d)
    p, y = _score(d, tr, te, ["M5", "M14"])
    n_ns = (~d.Site.iloc[te].isin(["M5", "M14"])).sum()
    assert len(p) == len(y) == n_ns


def test_greedy_selection_ignores_test_months(d):
    tr, te = _split(d)
    cands = candidate_sites(d)[:8]
    a = greedy_select(d, tr, 2, cands)
    d2 = d.copy(); d2.loc[te, "y"] = d2.loc[te, "y"] * -3 + 1
    assert greedy_select(d2, tr, 2, cands) == a
    assert "7" not in candidate_sites(d)                        # treatment stages are never sentinels


def test_qa_screen_drops_an_outlying_sentinel_reading(d):
    tr, te = _split(d)
    sent = ["M5", "M14"]
    d2 = d.copy()
    bad = d2.index[(d2.Site == "M5") & (d2.Date == "2018-06-01")][0]
    d2.loc[bad, "y"] -= 1.5                                    # simulated lab error on a test month
    row = d2.index[(d2.Site == "M10") & (d2.Date == "2018-06-01")][0]
    sm = d2.iloc[tr].groupby("Site").y.mean()
    m14 = d2[(d2.Site == "M14") & (d2.Date == "2018-06-01")].iloc[0]
    screened = sentinel_features(d2, tr, sent, qa=True).at[row, "sent_anom"]
    assert np.isclose(screened, m14.y - sm["M14"])            # only the good sentinel is averaged
    raw = sentinel_features(d2, tr, sent, qa=False).at[row, "sent_anom"]
    assert raw < screened - 0.5                                # without QA the error drags the estimate
