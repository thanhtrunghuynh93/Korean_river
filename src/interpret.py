"""Pick the best (tier, model) per target, refit on all rows, save it, and explain it.

    uv run python -m src.interpret                 # both targets
    uv run python -m src.interpret --target THMFP --tier 3 --model xgb   # force a choice

Selection rule: highest mean of pooled R2 under the `month` and `site` schemes (the two honest schemes)
among learned models in results/experiments.csv. Outputs:
  models/<target>_tier<k>_<model>.joblib            final pipeline fit on all rows
  results/importance.csv                            permutation importance, evaluated on site-out test folds
  results/pdp.csv                                   partial dependence of the top 4 features (final model)
  results/final_models.csv                          which model was chosen, and its CV numbers
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import RandomizedSearchCV

from preprocessing.load import TARGETS
from src.cv import inner_cv, inner_groups, splits
from src.experiments import EXPERIMENT_ORDER
from src.features import SITE_COL, base_frame, build, spec_label
from src.models import MODELS, N_SEARCH_ITER, make_model
from src.train import EXP_PATH, RESULTS, SEED

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
TOP_K = 4


def _spec(tier: str) -> int | str:
    return int(tier) if str(tier).isdigit() else str(tier)


def choose(exp: pd.DataFrame, target: str) -> tuple[int | str, str, pd.Series]:
    """Best (feature spec, model) by mean pooled R² over the month and site schemes (both required)."""
    e = exp[(exp.target == target) & exp.model.isin(MODELS)].copy()
    e["tier"] = e["tier"].astype(str)
    honest = (e[e.scheme.isin(["month", "site"])].pivot_table(index=["tier", "model"], columns="scheme", values="r2_log")
                .dropna(subset=["month", "site"]).mean(axis=1))
    tier, model = honest.idxmax()
    summary = e[(e.tier == tier) & (e.model == model)].set_index("scheme")["r2_log"]
    return _spec(tier), str(model), summary


def _feature_names(pipe, X: pd.DataFrame) -> list[str]:
    cols = list(X.columns)
    if "site_te" in dict(pipe.steps):
        cols = [c for c in cols if c != SITE_COL] + ["site_te"]
    return cols


def fit_final(target: str, tier: int | str, model: str, df: pd.DataFrame, seed: int = SEED):
    X, y, meta = build(target, tier, df)
    pipe, params = make_model(model, tier, seed)
    if params:
        search = RandomizedSearchCV(pipe, params, n_iter=N_SEARCH_ITER, cv=inner_cv("site"),
                                    scoring="neg_root_mean_squared_error", random_state=seed, n_jobs=-1)
        search.fit(X, y, groups=inner_groups("site", meta))
        est, best = search.best_estimator_, search.best_params_
    else:
        est, best = pipe.fit(X, y), {}
    return est, best, X, y, meta


def site_out_importance(target: str, tier: int | str, model: str, best_params: dict, df: pd.DataFrame,
                        seed: int = SEED) -> pd.DataFrame:
    """Permutation importance on held-out sites: refit with the chosen params on each site fold."""
    X, y, meta = build(target, tier, df)
    X, y, meta = X.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True)
    rows = []
    for fold, tr, te in splits("site", meta, seed):
        pipe, _ = make_model(model, tier, seed)
        pipe.set_params(**best_params)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        # permute *input* columns (the Site column permutation measures the value of the site encoding)
        r = permutation_importance(pipe, X.iloc[te], y.iloc[te], n_repeats=10, random_state=seed,
                                   scoring="neg_root_mean_squared_error", n_jobs=-1)
        rows.append(pd.DataFrame({"target": target, "tier": tier, "model": model, "fold": fold,
                                  "feature": X.columns, "importance": r.importances_mean}))
    imp = pd.concat(rows, ignore_index=True)
    return (imp.groupby(["target", "tier", "model", "feature"], as_index=False)["importance"]
               .agg(mean="mean", sd="std").sort_values("mean", ascending=False))


def pdp_table(est, X: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    rows = []
    for f in features:
        if f == SITE_COL:
            continue
        # explicit grid from the non-missing values (sklearn's automatic grid breaks on NaN columns)
        grid = np.unique(X[f].dropna().quantile(np.linspace(0.05, 0.95, 30)).to_numpy())
        pd_res = partial_dependence(est, X, [f], kind="average", custom_values={f: grid})
        rows.append(pd.DataFrame({"target": target, "feature": f, "x": pd_res["grid_values"][0],
                                  "y_pred": pd_res["average"][0]}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS)
    ap.add_argument("--tier", type=int)
    ap.add_argument("--exp", choices=EXPERIMENT_ORDER)
    ap.add_argument("--model", choices=MODELS)
    a = ap.parse_args()

    exp = pd.read_csv(EXP_PATH, dtype={"tier": str})
    df = base_frame()
    MODEL_DIR.mkdir(exist_ok=True)
    finals, imps, pdps = [], [], []
    for target in ([a.target] if a.target else TARGETS):
        forced = a.exp if a.exp else a.tier
        if forced is not None and a.model:
            tier, model = forced, a.model
            cv_r2 = exp[(exp.target == target) & (exp.tier == str(tier)) & (exp.model == model)] \
                .set_index("scheme")["r2_log"]
        else:
            tier, model, cv_r2 = choose(exp, target)
        print(f"{target}: {spec_label(tier)} {model}  CV R2 by scheme: {cv_r2.round(3).to_dict()}")

        est, best, X, y, meta = fit_final(target, tier, model, df)
        path = MODEL_DIR / f"{target}_{spec_label(tier)}_{model}.joblib"
        joblib.dump({"pipeline": est, "features": list(X.columns), "target": target, "tier": tier,
                     "model": model, "best_params": best, "note": "predicts log10(target in mg/L)"}, path)
        finals.append({"target": target, "tier": tier, "model": model, "path": str(path.relative_to(ROOT)),
                       "best_params": str(best), **{f"r2_{k}": v for k, v in cv_r2.items()}})

        imp = site_out_importance(target, tier, model, best, df)
        imps.append(imp)
        top = [f for f in imp.feature.tolist() if f != SITE_COL][:TOP_K]
        pdps.append(pdp_table(est, X, top, target))
        print("  top features:", imp.head(6)[["feature", "mean"]].round(3).values.tolist())

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(finals).to_csv(RESULTS / "final_models.csv", index=False)
    pd.concat(imps).to_csv(RESULTS / "importance.csv", index=False)
    pd.concat(pdps).to_csv(RESULTS / "pdp.csv", index=False)
    print("saved", RESULTS / "final_models.csv", RESULTS / "importance.csv", RESULTS / "pdp.csv")


if __name__ == "__main__":
    main()
