"""Revision evaluation engine: information-masked scenarios, fully nested selection, grouped tuning, intervals.

Scenarios (what is permitted when predicting a recipient sample)
    S1a  water quality + location metadata (group, stage, plant, main-stem position, month); no assay history
    S1b  S1a + the site's calibration level = mean log10 target over the *training* rows of that site
         (falls back to the site-group mean, then the global mean, for sites without training rows)
    S3b  S1b + recipient's past assays (lag-1, history mean) + other sites' current-campaign assays
         (upstream, downstream, basin and group means, plant raw water) = network gap-filling ("basin" set)
    S3a  S3b + LC-OCD ratios, sparse extras and cross-target lags ("all" set)
    Sentinel scenarios (S2) live in src/rev_sentinel.py.

Candidates (prespecified; every choice made inside each outer training set)
    ridge   alpha in {0.3, 3, 30, 300} on standardised inputs (median imputation + missing indicators)
    xgb     4 configurations drawn once (seed 0) from the round-1 search space
    blend   equal-weight average of any ridge and any xgb candidate (16 pairs)
    Inner validation: GroupKFold(4) grouped like the outer scheme (campaigns for month/forward/rolling, sites
    for site/plant). Pooled inner R^2 picks the best ridge, best xgb, best blend and the overall best.

Intervals ("inner cross-validation conformal", not split conformal)
    For each winner, absolute residuals of its inner out-of-fold predictions are pooled by site group; the
    finite-sample quantile ceil((n+1)(1-a))/n gives the half-width q in log10 units, applied around the
    prediction of the model refitted on the full outer training set. Multiplicative half-width f = 10^q:
    interval [yhat / f, yhat * f]. Exchangeability does not hold exactly for clustered, temporally shifted
    data, so coverage is reported empirically rather than guaranteed.

    uv run python -m src.rev_eval            # -> results/revision/eval/*.csv
"""
from __future__ import annotations

import time
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.cv import splits as base_splits
from src.experiments import EXPERIMENTS
from src.features import SITE_COL, T1_COLS, T2_EXTRA, add_cross_target, add_target_context, base_frame, target_col
from src.models import XGB_SPACE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "eval"
TARGETS = ("THMFP", "HAAFP")
CONTEXT = [c for c in T2_EXTRA if c != SITE_COL]
S1A = T1_COLS + CONTEXT
LEVEL = "site_level"
SCENARIOS = {
    "S1a": dict(cols=S1A, level=False),
    "S1b": dict(cols=S1A, level=True),
    "S3b": dict(cols=[c for c in EXPERIMENTS["t3_basin"] if c != SITE_COL], level=True),
    "S3a": dict(cols=[c for c in EXPERIMENTS["t4_all"] if c != SITE_COL], level=True),
}
RIDGE_ALPHAS = [0.3, 3.0, 30.0, 300.0]


def _xgb_configs(n=4, seed=0):
    rng = np.random.default_rng(seed)
    cfgs = []
    for _ in range(n):
        cfgs.append({k: (int(v.rvs(random_state=rng)) if hasattr(v, "dist") and v.dist.name == "randint"
                         else float(v.rvs(random_state=rng))) for k, v in XGB_SPACE.items()})
    return cfgs


XGB_CONFIGS = _xgb_configs()
NOMINAL = (0.8, 0.9)


# ------------------------------------------------------------------------------------------ data + masks
def frame(target: str, bf: pd.DataFrame | None = None) -> pd.DataFrame:
    bf = base_frame() if bf is None else bf
    d = add_cross_target(add_target_context(bf, target), target)
    d = d[d[target_col(target)].notna()].reset_index(drop=True)
    d["y"] = d[target_col(target)]
    d["group"] = d["group"].astype(str)
    d["Site"] = d["Site"].astype(str)
    return d


def site_level(d: pd.DataFrame, train_idx: np.ndarray) -> pd.Series:
    """Calibration level from training rows only: site mean -> group mean -> global mean."""
    tr = d.iloc[train_idx]
    return d.Site.map(tr.groupby("Site").y.mean()).fillna(d.group.map(tr.groupby("group").y.mean())).fillna(tr.y.mean())


def design(d: pd.DataFrame, scen, train_idx: np.ndarray) -> pd.DataFrame:
    """`scen` is a scenario name from SCENARIOS or a dict(cols=..., level=..., name=...)."""
    spec = scen if isinstance(scen, dict) else SCENARIOS[scen]
    X = d[spec["cols"]].copy()
    if spec["level"]:
        X[LEVEL] = site_level(d, train_idx).to_numpy()
    return X


def inner_groups(scheme: str, d: pd.DataFrame) -> np.ndarray:
    return d.Site.to_numpy() if scheme in ("site", "plant") else d.Date.astype(str).to_numpy()


# ------------------------------------------------------------------------------------------ custom schemes
PLANT_SITES = {"Gumi": ["6", "7", "8", "9", "10", "11"], "Goryeong": ["M9", "12", "13", "14", "15", "16"],
               "Bansong": ["M14", "17", "18", "19", "20"]}


def splits(scheme: str, d: pd.DataFrame, seed: int = 0):
    idx = np.arange(len(d))
    if scheme == "rolling":
        months = np.sort(d.Date.unique())
        for m in (8, 10, 12, 14):
            tr = d.Date.isin(months[:m]).to_numpy(); te = d.Date.isin(months[m:m + 2]).to_numpy()
            yield f"rolling_{m}", idx[tr], idx[te]
    elif scheme == "plant":
        for p, sites in PLANT_SITES.items():
            te = d.Site.isin(sites).to_numpy()
            yield f"plant_{p}", idx[~te], idx[te]
    else:
        yield from base_splits(scheme, d[["Site", "Date", "group"]], seed)


# ------------------------------------------------------------------------------------------ candidates
def fit_ridge(X, y, alpha):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=alpha)).fit(X, y)


def fit_xgb(X, y, cfg):
    return XGBRegressor(**cfg, random_state=0, n_jobs=1, tree_method="hist", verbosity=0).fit(X, y)


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan


def conformal_q(abs_res: np.ndarray, groups: np.ndarray, level: float) -> dict:
    out = {}
    for g in np.unique(groups):
        r = np.sort(abs_res[groups == g]); n = len(r)
        out[g] = float(r[min(n - 1, ceil((n + 1) * level) - 1)]) if n else np.nan
    r = np.sort(abs_res); n = len(r)
    out["__all__"] = float(r[min(n - 1, ceil((n + 1) * level) - 1)])
    return out


def nested_fold(d: pd.DataFrame, scen: str, scheme: str, tr: np.ndarray, te: np.ndarray, fold: str, seed: int):
    """All selection inside `tr`. Returns row-level predictions for `te` plus chosen configurations."""
    dtr = d.iloc[tr].reset_index(drop=True)
    g = inner_groups(scheme, dtr)
    n_inner = min(4, len(np.unique(g)))
    inner = list(GroupKFold(n_inner).split(dtr, groups=g))
    ytr = dtr.y.to_numpy()
    oof = {f"ridge_{a}": np.full(len(dtr), np.nan) for a in RIDGE_ALPHAS}
    oof.update({f"xgb_{i}": np.full(len(dtr), np.nan) for i in range(len(XGB_CONFIGS))})
    for itr, ite in inner:
        Xi = design(dtr, scen, itr)                     # calibration level from inner-training rows only
        for a in RIDGE_ALPHAS:
            oof[f"ridge_{a}"][ite] = fit_ridge(Xi.iloc[itr], ytr[itr], a).predict(Xi.iloc[ite])
        for i, cfg in enumerate(XGB_CONFIGS):
            oof[f"xgb_{i}"][ite] = fit_xgb(Xi.iloc[itr], ytr[itr], cfg).predict(Xi.iloc[ite])
    for a in RIDGE_ALPHAS:
        for i in range(len(XGB_CONFIGS)):
            oof[f"blend_{a}_{i}"] = (oof[f"ridge_{a}"] + oof[f"xgb_{i}"]) / 2
    score = {k: r2(ytr, v) for k, v in oof.items()}
    best = {fam: max((k for k in score if k.startswith(fam + "_")), key=score.get) for fam in ("ridge", "xgb", "blend")}
    best["selected"] = max(score, key=score.get)

    Xo = design(d, scen, tr)                             # calibration level from outer-training rows only
    fitted = {}

    def predict(key):
        if key in fitted: return fitted[key]
        kind, *rest = key.split("_")
        if kind == "ridge":
            p = fit_ridge(Xo.iloc[tr], d.y.iloc[tr], float(rest[0])).predict(Xo.iloc[te])
        elif kind == "xgb":
            p = fit_xgb(Xo.iloc[tr], d.y.iloc[tr], XGB_CONFIGS[int(rest[0])]).predict(Xo.iloc[te])
        else:
            p = (predict(f"ridge_{rest[0]}") + predict(f"xgb_{rest[1]}")) / 2
        fitted[key] = p
        return p

    rows = pd.DataFrame({"row": te, "Site": d.Site.iloc[te].to_numpy(), "Date": d.Date.iloc[te].to_numpy(),
                         "group": d.group.iloc[te].to_numpy(), "y": d.y.iloc[te].to_numpy()})
    for fam, key in best.items():
        rows[f"yhat_{fam}"] = predict(key)
        res = np.abs(ytr - oof[key])
        for lev in NOMINAL:
            q = conformal_q(res, dtr.group.to_numpy(), lev)
            rows[f"q{int(lev * 100)}_{fam}"] = rows.group.map(q).fillna(q["__all__"]).to_numpy()
    label = scen["name"] if isinstance(scen, dict) else scen
    rows = rows.assign(scenario=label, scheme=scheme, fold=fold, seed=seed)
    chosen = {"scenario": label, "scheme": scheme, "fold": fold, "seed": seed, **{f"chosen_{k}": v for k, v in best.items()},
              **{f"inner_r2_{k}": score[v] for k, v in best.items()}}
    return rows, chosen


def run(target: str, scen: str, scheme: str, d: pd.DataFrame, seeds=(0,), n_jobs=-1):
    jobs = [(fold, tr, te, s) for s in seeds for fold, tr, te in splits(scheme, d, s)]
    out = Parallel(n_jobs=n_jobs)(delayed(nested_fold)(d, scen, scheme, tr, te, fold, s) for fold, tr, te, s in jobs)
    pred = pd.concat([o[0] for o in out], ignore_index=True).assign(target=target)
    chosen = pd.DataFrame([o[1] for o in out]).assign(target=target)
    return pred, chosen


PLAN = [  # (scenario, scheme, seeds)
    *[(s, "month", (0,)) for s in SCENARIOS], *[(s, "site", (0, 1, 2)) for s in SCENARIOS],
    *[(s, "forward", (0,)) for s in SCENARIOS], *[(s, "rolling", (0,)) for s in SCENARIOS],
    ("S1a", "plant", (0,)), ("S1b", "plant", (0,)),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bf = base_frame()
    P, C = [], []
    for t in TARGETS:
        d = frame(t, bf)
        for scen, scheme, seeds in PLAN:
            t0 = time.time()
            p, c = run(t, scen, scheme, d, seeds)
            P.append(p); C.append(c)
            print(f"{t} {scen} {scheme}: selected R2 {r2(p.y, p.yhat_selected):.3f} | ridge {r2(p.y, p.yhat_ridge):.3f} "
                  f"xgb {r2(p.y, p.yhat_xgb):.3f} blend {r2(p.y, p.yhat_blend):.3f}  [{time.time() - t0:.0f}s]", flush=True)
    pd.concat(P).assign(Date=lambda x: pd.to_datetime(x.Date).dt.strftime("%Y-%m-%d")).to_csv(OUT / "predictions.csv", index=False)
    pd.concat(C).to_csv(OUT / "chosen.csv", index=False)
    pd.DataFrame(XGB_CONFIGS).assign(config=range(len(XGB_CONFIGS))).to_csv(OUT / "xgb_configs.csv", index=False)


if __name__ == "__main__":
    main()
