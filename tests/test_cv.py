import numpy as np
import pytest

from src.cv import FORWARD_TRAIN_END, N_MONTH_FOLDS, N_SITE_FOLDS, metrics, splits
from src.features import build


@pytest.fixture(scope="module")
def meta():
    _, _, meta = build("THMFP", 1)
    return meta.reset_index(drop=True)


def test_month_folds_do_not_share_dates(meta):
    folds = list(splits("month", meta))
    assert len(folds) == N_MONTH_FOLDS
    for _, tr, te in folds:
        assert not set(meta.Date.iloc[tr]) & set(meta.Date.iloc[te])
        assert meta.Date.iloc[te].nunique() == 2


def test_site_folds_do_not_share_sites_and_mix_groups(meta):
    folds = list(splits("site", meta, seed=0))
    assert len(folds) == N_SITE_FOLDS
    covered = []
    for _, tr, te in folds:
        assert not set(meta.Site.iloc[tr]) & set(meta.Site.iloc[te])
        assert 3 <= meta.Site.iloc[te].nunique() <= 5
        covered += list(meta.Site.iloc[te].unique())
    assert sorted(covered) == sorted(meta.Site.unique())


def test_forward_holdout_is_strictly_later(meta):
    (_, tr, te), = list(splits("forward", meta))
    assert meta.Date.iloc[tr].max() <= FORWARD_TRAIN_END < meta.Date.iloc[te].min()
    assert meta.Date.iloc[te].nunique() == 4


def test_random_folds_cover_everything(meta):
    folds = list(splits("random", meta))
    assert len(folds) == 10
    for _, tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(meta)


def test_metrics_perfect_and_medape():
    y = np.array([-1.0, -0.5, 0.0])
    m = metrics(y, y)
    assert m["rmse_log"] == 0 and m["r2_log"] == 1 and m["medape"] == 0
    m2 = metrics(y, y + np.log10(1.1))
    assert np.isclose(m2["medape"], 0.1)
