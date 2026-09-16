"""Train / evaluate one (target, tier, model, scheme) cell, or the whole grid.

    uv run python -m src.train --target THMFP --tier 3 --model xgb --scheme site
    uv run python -m src.train --all            # 2 targets x 3 tiers x 3 models x 4 schemes + baselines

Every outer fold runs a nested RandomizedSearchCV whose inner folds share the outer scheme's grouping
(months or sites), so tuning never sees held-out months / sites. Out-of-fold predictions are appended to
results/predictions.csv and aggregated metrics to results/experiments.csv (existing rows for the same key
are replaced, so cells can be rerun individually).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from preprocessing.load import TARGETS
from src.cv import SCHEMES, inner_cv, inner_groups, metrics, splits
from src.features import TIERS, base_frame, build
from src.models import BASELINES, MODELS, N_SEARCH_ITER, make_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRED_PATH = RESULTS / "predictions.csv"
EXP_PATH = RESULTS / "experiments.csv"
KEY = ["target", "tier", "model", "scheme"]
SEED = 0
N_JOBS = -1


def _baseline_predict(name: str, y_tr: pd.Series, meta_tr: pd.DataFrame, meta_te: pd.DataFrame) -> np.ndarray:
    global_med = float(y_tr.median())
    site_med = y_tr.groupby(meta_tr["Site"].to_numpy()).median()
    site_pred = meta_te["Site"].map(site_med).fillna(global_med).to_numpy()
    if name == "median":
        return np.full(len(meta_te), global_med)
    if name == "site_median":
        return site_pred
    if name == "persistence":                       # previous-month same-site value, else site median
        return meta_te["lag1_y"].fillna(pd.Series(site_pred, index=meta_te.index)).to_numpy()
    raise ValueError(name)


def run_cell(target: str, tier: int | str, model: str, scheme: str, df: pd.DataFrame | None = None,
             seed: int = SEED, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    is_baseline = model in BASELINES
    X, y, meta = build(target, 1 if is_baseline else int(tier), df)
    X, y, meta = X.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True)
    tier_label = "base" if is_baseline else int(tier)

    preds, best_params = [], []
    for fold, tr, te in splits(scheme, meta, seed):
        if is_baseline:
            p = _baseline_predict(model, y.iloc[tr], meta.iloc[tr], meta.iloc[te])
        else:
            pipe, params = make_model(model, int(tier), seed)
            if params:
                search = RandomizedSearchCV(pipe, params, n_iter=N_SEARCH_ITER, cv=inner_cv(scheme),
                                            scoring="neg_root_mean_squared_error", random_state=seed,
                                            n_jobs=N_JOBS, refit=True)
                g = inner_groups(scheme, meta.iloc[tr])
                search.fit(X.iloc[tr], y.iloc[tr], groups=g)
                est = search.best_estimator_
                best_params.append({k.replace("model__", ""): v for k, v in search.best_params_.items()})
            else:
                est = pipe.fit(X.iloc[tr], y.iloc[tr])
            p = est.predict(X.iloc[te])
        preds.append(pd.DataFrame({"target": target, "tier": tier_label, "model": model, "scheme": scheme,
                                   "fold": fold, "row": te, "Site": meta.Site.iloc[te].to_numpy(),
                                   "Date": meta.Date.iloc[te].to_numpy(), "group": meta.group.iloc[te].to_numpy(),
                                   "y": y.iloc[te].to_numpy(), "y_pred": p}))
    pred = pd.concat(preds, ignore_index=True)

    # pooled metrics + per-fold spread + per-group R2
    row = {"target": target, "tier": tier_label, "model": model, "scheme": scheme, "n": len(pred)}
    row.update(metrics(pred.y, pred.y_pred))
    per_fold = pred.groupby("fold").apply(lambda d: pd.Series(metrics(d.y, d.y_pred)), include_groups=False)
    row["r2_fold_mean"], row["r2_fold_sd"] = float(per_fold.r2_log.mean()), float(per_fold.r2_log.std(ddof=0))
    row["rmse_fold_mean"], row["rmse_fold_sd"] = float(per_fold.rmse_log.mean()), float(per_fold.rmse_log.std(ddof=0))
    for grp, d in pred.groupby("group"):
        row[f"r2_{grp}"] = metrics(d.y, d.y_pred)["r2_log"]
        row[f"rmse_{grp}"] = metrics(d.y, d.y_pred)["rmse_log"]
    row["best_params"] = str(pd.DataFrame(best_params).median(numeric_only=True).round(3).to_dict()) if best_params else ""
    row["seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"{target:6s} tier={tier_label!s:4s} {model:12s} {scheme:8s} "
              f"R2={row['r2_log']:.3f} (folds {row['r2_fold_mean']:.3f}±{row['r2_fold_sd']:.3f}) "
              f"RMSE={row['rmse_log']:.3f} medAPE={row['medape']:.2f}  [{row['seconds']}s]")
    return pred, row


def _upsert(path: Path, new: pd.DataFrame) -> None:
    RESULTS.mkdir(exist_ok=True)
    if "Date" in new.columns:                                   # one text format, so re-reads parse cleanly
        new = new.assign(Date=pd.to_datetime(new["Date"]).dt.strftime("%Y-%m-%d"))
    if path.exists():
        old = pd.read_csv(path)
        old["tier"] = old["tier"].astype(str)
        new = new.copy(); new["tier"] = new["tier"].astype(str)
        key_new = new[KEY].drop_duplicates()
        keep = ~old[KEY].astype(str).apply(tuple, axis=1).isin(key_new.astype(str).apply(tuple, axis=1))
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(path, index=False)


def save(pred: pd.DataFrame, row: dict) -> None:
    _upsert(PRED_PATH, pred)
    _upsert(EXP_PATH, pd.DataFrame([row]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS)
    ap.add_argument("--tier", type=int, choices=TIERS)
    ap.add_argument("--model", choices=MODELS + BASELINES)
    ap.add_argument("--scheme", choices=SCHEMES)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    df = base_frame()
    if a.all:
        cells = [(t, "base", b, s) for t in TARGETS for s in SCHEMES for b in BASELINES]
        cells += [(t, tier, m, s) for t in TARGETS for s in SCHEMES for tier in TIERS for m in MODELS]
    else:
        if not (a.target and a.model and a.scheme) or (a.model in MODELS and a.tier is None):
            ap.error("need --target, --model, --scheme (and --tier for a learned model), or --all")
        cells = [(a.target, a.tier, a.model, a.scheme)]
    for target, tier, model, scheme in cells:
        pred, row = run_cell(target, tier, model, scheme, df, a.seed)
        save(pred, row)


if __name__ == "__main__":
    main()
