"""Train / evaluate one (target, feature spec, model, scheme) cell, the original grid, or the ablation.

    uv run python -m src.train --target THMFP --tier 3 --model xgb --scheme site
    uv run python -m src.train --target THMFP --exp t3_basin --model xgb --scheme month
    uv run python -m src.train --all            # tiers 1-3 x 3 models x 4 schemes + baselines
    uv run python -m src.train --ablation       # experiments x 2 targets x {ridge, xgb} x {month, site, forward}

Every outer fold runs a nested RandomizedSearchCV whose inner folds share the outer scheme's grouping
(months or sites), so tuning never sees held-out months / sites. Schemes with a random element (`random`,
`site`) are repeated for `--seeds` seeds; metrics are averaged over seeds and `r2_seed_sd` is stored.
Out-of-fold predictions go to results/predictions.csv, aggregated metrics to results/experiments.csv
(rows with the same key are replaced, so cells can be rerun individually). The `tier` column holds the
feature spec: 1 / 2 / 3, an experiment name, or "base" for baselines.
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
from src.experiments import EXPERIMENT_ORDER
from src.features import TIERS, base_frame, build
from src.models import BASELINES, MODELS, N_SEARCH_ITER, make_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRED_PATH = RESULTS / "predictions.csv"
EXP_PATH = RESULTS / "experiments.csv"
KEY = ["target", "tier", "model", "scheme"]
SEED = 0
N_SEEDS = 3
SEEDED_SCHEMES = ("random", "site")
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


def _spec_key(spec, is_baseline: bool):
    if is_baseline:
        return "base"
    return int(spec) if str(spec).isdigit() else str(spec)


def _one_seed(target, spec, model, scheme, X, y, meta, seed, is_baseline):
    preds, best_params = [], []
    for fold, tr, te in splits(scheme, meta, seed):
        if is_baseline:
            p = _baseline_predict(model, y.iloc[tr], meta.iloc[tr], meta.iloc[te])
        else:
            pipe, params = make_model(model, spec, seed)
            if params:
                search = RandomizedSearchCV(pipe, params, n_iter=N_SEARCH_ITER, cv=inner_cv(scheme),
                                            scoring="neg_root_mean_squared_error", random_state=seed,
                                            n_jobs=N_JOBS, refit=True)
                search.fit(X.iloc[tr], y.iloc[tr], groups=inner_groups(scheme, meta.iloc[tr]))
                est = search.best_estimator_
                best_params.append({k.replace("model__", ""): v for k, v in search.best_params_.items()})
            else:
                est = pipe.fit(X.iloc[tr], y.iloc[tr])
            p = est.predict(X.iloc[te])
        preds.append(pd.DataFrame({"fold": fold, "seed": seed, "row": te,
                                   "Site": meta.Site.iloc[te].to_numpy(), "Date": meta.Date.iloc[te].to_numpy(),
                                   "group": meta.group.iloc[te].to_numpy(),
                                   "y": y.iloc[te].to_numpy(), "y_pred": p}))
    return pd.concat(preds, ignore_index=True), best_params


def run_cell(target: str, spec, model: str, scheme: str, df: pd.DataFrame | None = None,
             seed: int = SEED, n_seeds: int = N_SEEDS, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    is_baseline = model in BASELINES
    X, y, meta = build(target, 1 if is_baseline else spec, df)
    X, y, meta = X.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True)
    key = _spec_key(spec, is_baseline)
    seeds = list(range(seed, seed + n_seeds)) if scheme in SEEDED_SCHEMES else [seed]

    per_seed, all_pred, best_params = [], [], []
    for s in seeds:
        pred, bp = _one_seed(target, spec, model, scheme, X, y, meta, s, is_baseline)
        best_params += bp
        pred_key = pd.DataFrame({"target": target, "tier": key, "model": model, "scheme": scheme}, index=pred.index)
        all_pred.append(pd.concat([pred_key, pred], axis=1))
        m = metrics(pred.y, pred.y_pred)
        per_fold = pred.groupby("fold").apply(lambda d: pd.Series(metrics(d.y, d.y_pred)), include_groups=False)
        m["r2_fold_mean"], m["r2_fold_sd"] = float(per_fold.r2_log.mean()), float(per_fold.r2_log.std(ddof=0))
        m["rmse_fold_mean"], m["rmse_fold_sd"] = float(per_fold.rmse_log.mean()), float(per_fold.rmse_log.std(ddof=0))
        for grp, d in pred.groupby("group"):
            m[f"r2_{grp}"] = metrics(d.y, d.y_pred)["r2_log"]
            m[f"rmse_{grp}"] = metrics(d.y, d.y_pred)["rmse_log"]
        per_seed.append(m)
    pred = pd.concat(all_pred, ignore_index=True)
    agg = pd.DataFrame(per_seed)

    row = {"target": target, "tier": key, "model": model, "scheme": scheme, "n": int(len(pred) / len(seeds)),
           "n_seeds": len(seeds)}
    row.update(agg.mean(numeric_only=True).to_dict())
    row["r2_seed_sd"] = float(agg.r2_log.std(ddof=0)) if len(seeds) > 1 else 0.0
    row["best_params"] = str(pd.DataFrame(best_params).median(numeric_only=True).round(3).to_dict()) if best_params else ""
    row["seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"{target:6s} spec={key!s:9s} {model:12s} {scheme:8s} "
              f"R2={row['r2_log']:.3f} (folds {row['r2_fold_mean']:.3f}±{row['r2_fold_sd']:.3f}, seeds ±{row['r2_seed_sd']:.3f}) "
              f"RMSE={row['rmse_log']:.3f} medAPE={row['medape']:.2f}  [{row['seconds']}s]")
    return pred, row


def _upsert(path: Path, new: pd.DataFrame) -> None:
    RESULTS.mkdir(exist_ok=True)
    if "Date" in new.columns:                                   # one text format, so re-reads parse cleanly
        new = new.assign(Date=pd.to_datetime(new["Date"]).dt.strftime("%Y-%m-%d"))
    if path.exists():
        old = pd.read_csv(path, dtype={"tier": str})
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
    ap.add_argument("--exp", choices=EXPERIMENT_ORDER)
    ap.add_argument("--model", choices=MODELS + BASELINES)
    ap.add_argument("--scheme", choices=SCHEMES)
    ap.add_argument("--all", action="store_true", help="original tier grid, single seed")
    ap.add_argument("--ablation", action="store_true", help="experiments x targets x {ridge,xgb} x {month,site,forward}")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--seeds", type=int, default=None, help="repeats for random/site schemes (default 1 for --all, 3 otherwise)")
    a = ap.parse_args()

    df = base_frame()
    if a.all:
        n_seeds = a.seeds or 1
        cells = [(t, "base", b, s) for t in TARGETS for s in SCHEMES for b in BASELINES]
        cells += [(t, tier, m, s) for t in TARGETS for s in SCHEMES for tier in TIERS for m in MODELS]
    elif a.ablation:
        n_seeds = a.seeds or N_SEEDS
        cells = [(t, e, m, s) for e in EXPERIMENT_ORDER for t in TARGETS for m in ("ridge", "xgb")
                 for s in ("month", "site", "forward")]
    else:
        n_seeds = a.seeds or N_SEEDS
        spec = a.exp if a.exp else a.tier
        if not (a.target and a.model and a.scheme) or (a.model in MODELS and spec is None):
            ap.error("need --target, --model, --scheme and --tier/--exp for a learned model; or --all / --ablation")
        cells = [(a.target, spec, a.model, a.scheme)]
    for target, spec, model, scheme in cells:
        pred, row = run_cell(target, spec, model, scheme, df, a.seed, n_seeds)
        save(pred, row)


if __name__ == "__main__":
    main()
