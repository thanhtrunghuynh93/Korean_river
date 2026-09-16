"""Cross-validation schemes and metrics.

random   RepeatedKFold 5 folds x 2 repeats                 optimistic reference
month    GroupKFold by Date, 8 folds (2 months each)       new months, known sites
site     11 folds of 4 sites, stratified by site group     new sites, known months
forward  single split: train <= 2018-04, test 2018-05..08  strict temporal holdout

`splits()` yields (fold_name, train_idx, test_idx) as positional indices into meta.
`inner_groups()` gives the grouping used for nested hyper-parameter search under each scheme.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, RepeatedKFold

SCHEMES = ("random", "month", "site", "forward")
FORWARD_TRAIN_END = pd.Timestamp("2018-04-01")
N_SITE_FOLDS = 11
N_MONTH_FOLDS = 8


def _site_folds(meta: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Site -> fold id, round-robin within each group so every fold mixes groups."""
    rng = np.random.default_rng(seed)
    sites = meta[["Site", "group"]].drop_duplicates().sort_values(["group", "Site"])
    fold_of: dict[str, int] = {}
    k = 0
    for _, sub in sites.groupby("group", sort=True):
        order = rng.permutation(sub["Site"].to_numpy())
        for s in order:
            fold_of[s] = k % n_folds
            k += 1
    return meta["Site"].map(fold_of)


def splits(scheme: str, meta: pd.DataFrame, seed: int = 0) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    n = len(meta)
    idx = np.arange(n)
    if scheme == "random":
        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)
        for i, (tr, te) in enumerate(rkf.split(idx)):
            yield f"random_{i}", tr, te
    elif scheme == "month":
        dates = meta["Date"].to_numpy()
        for i, (tr, te) in enumerate(GroupKFold(n_splits=N_MONTH_FOLDS).split(idx, groups=dates)):
            yield f"month_{i}", tr, te
    elif scheme == "site":
        fold = _site_folds(meta, N_SITE_FOLDS, seed).to_numpy()
        for k in range(N_SITE_FOLDS):
            te = idx[fold == k]
            yield f"site_{k}", idx[fold != k], te
    elif scheme == "forward":
        is_test = (meta["Date"] > FORWARD_TRAIN_END).to_numpy()
        yield "forward_0", idx[~is_test], idx[is_test]
    else:
        raise ValueError(f"scheme must be one of {SCHEMES}")


def inner_groups(scheme: str, meta: pd.DataFrame) -> np.ndarray | None:
    """Groups for the nested search: same leakage structure as the outer scheme."""
    if scheme in ("month", "forward"):
        return meta["Date"].astype(str).to_numpy()
    if scheme == "site":
        return meta["Site"].astype(str).to_numpy()
    return None


def inner_cv(scheme: str, n_splits: int = 3):
    from sklearn.model_selection import KFold
    return GroupKFold(n_splits=n_splits) if scheme != "random" else KFold(n_splits=n_splits, shuffle=True, random_state=0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Log-scale RMSE / MAE / R2 plus median absolute percentage error on the original scale."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ape = np.abs(10 ** y_pred - 10 ** y_true) / (10 ** y_true)
    return {
        "rmse_log": float(root_mean_squared_error(y_true, y_pred)),
        "mae_log": float(mean_absolute_error(y_true, y_pred)),
        "r2_log": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "medape": float(np.median(ape)),
    }
