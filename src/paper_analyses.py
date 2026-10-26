"""Analyses that feed the manuscript tables and figures (A1-A4, A6 of the paper plan).

A1  descriptive statistics by site group + Kruskal-Wallis tests (site group; season on river rows)
A2  treatment-train changes relative to raw water (median % change, Wilcoxon signed-rank, paired by month)
A3  consensus importance of the chemistry-only inputs: Spearman |rho|, XGBoost gain, permutation (site-out),
    mean |SHAP|, and Kendall's W agreement of the four rankings
A4  basin co-variation: share of within-site variance shared by all sites in a month, main-stem correlation
    versus separation, share left after chemistry
A6  scenario table (selection-adjusted R^2 + 90 % CI, conformal widths) and sentinel curve with CIs

    uv run python -m src.paper_analyses        # writes results/paper/*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from preprocessing.load import GROUP_ORDER, MAIN_STEM_ORDER, NUMERIC, STAGE_ORDER, UNITS, WTP_RAW_SITES, load_clean
from src.cv import splits
from src.features import T1_COLS, base_frame, target_col
from src.select import boot_r2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper"
TARGETS = ["THMFP", "HAAFP"]
XGB_FIXED = dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=3, reg_lambda=1.0, random_state=0, n_jobs=4, tree_method="hist", verbosity=0)


PAPER_EXCLUDED = ["Aromaticity", "MolWeight"]   # in the workbook, but not reported in the paper (2026-10-09)


def descriptive(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    river = df[df.group.astype(str) != "treatment"]
    for v in [c for c in NUMERIC if c not in PAPER_EXCLUDED]:
        s = df[v]
        r = {"variable": v, "unit": UNITS[v], "n": int(s.notna().sum()), "mean": s.mean(), "sd": s.std(),
             "median": s.median(), "min": s.min(), "max": s.max(), "skewness": s.skew(),
             "missing_pct": 100 * s.isna().mean()}
        for g in GROUP_ORDER:
            r[f"median_{g}"] = df.loc[df.group.astype(str) == g, v].median()
        groups = [df.loc[df.group.astype(str) == g, v].dropna() for g in GROUP_ORDER]
        groups = [g for g in groups if len(g) > 2]
        r["kw_group_H"], r["kw_group_p"] = stats.kruskal(*groups) if len(groups) > 1 else (np.nan, np.nan)
        seas = [river.loc[river.season.astype(str) == z, v].dropna() for z in ["spring", "summer", "autumn", "winter"]]
        seas = [z for z in seas if len(z) > 2]
        r["kw_season_H"], r["kw_season_p"] = stats.kruskal(*seas) if len(seas) > 1 else (np.nan, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def treatment_changes(df: pd.DataFrame) -> pd.DataFrame:
    raw_site = {v: k for k, v in WTP_RAW_SITES.items()}
    rows = []
    for wtp, raw in raw_site.items():
        base = df[df.Site.astype(str) == raw].set_index("Date")
        train = df[(df.wtp == wtp) & (df.Site.astype(str) != raw)]
        for stage, sub in train.groupby("stage", observed=True):
            sub = sub.set_index("Date")
            for v in ["TOC", "THMFP", "HAAFP"]:
                pair = pd.concat([base[v].rename("raw"), sub[v].rename("stage")], axis=1).dropna()
                if len(pair) < 5:
                    continue
                pct = 100 * (pair.stage - pair.raw) / pair.raw
                w = stats.wilcoxon(pair.stage, pair.raw)
                rows.append({"wtp": wtp, "stage": stage, "stage_idx": STAGE_ORDER.index(stage), "variable": v,
                             "n_months": len(pair), "median_raw": pair.raw.median(), "median_stage": pair.stage.median(),
                             "median_pct_change": pct.median(), "q1_pct": pct.quantile(0.25), "q3_pct": pct.quantile(0.75),
                             "wilcoxon_p": w.pvalue})
    return pd.DataFrame(rows).sort_values(["wtp", "variable", "stage_idx"])


def kendall_w(ranks: np.ndarray) -> float:
    """Kendall's coefficient of concordance for an (m raters x n items) rank matrix."""
    m, n = ranks.shape
    s = ((ranks.sum(axis=0) - m * (n + 1) / 2) ** 2).sum()
    return float(12 * s / (m ** 2 * (n ** 3 - n)))


def consensus_importance(bf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import shap
    rows, agree = [], []
    for t in TARGETS:
        d = bf[bf[target_col(t)].notna()].reset_index(drop=True)
        X, y = d[T1_COLS], d[target_col(t)]
        meta = d[["Site", "Date"]].assign(group=d.group.astype(str))
        rho = X.apply(lambda c: abs(stats.spearmanr(c, y, nan_policy="omit")[0]))
        model = XGBRegressor(**XGB_FIXED).fit(X, y)
        gain = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(T1_COLS).fillna(0)
        sv = shap.TreeExplainer(model).shap_values(X)
        shap_mean = pd.Series(np.abs(sv).mean(axis=0), index=T1_COLS)
        perm = []
        for _, tr, te in splits("site", meta, 0):
            m = XGBRegressor(**XGB_FIXED).fit(X.iloc[tr], y.iloc[tr])
            r = permutation_importance(m, X.iloc[te], y.iloc[te], n_repeats=10, random_state=0,
                                       scoring="neg_root_mean_squared_error")
            perm.append(r.importances_mean)
        perm = pd.Series(np.mean(perm, axis=0), index=T1_COLS)
        tab = pd.DataFrame({"spearman_abs_rho": rho, "xgb_gain": gain, "permutation_drmse": perm, "mean_abs_shap": shap_mean})
        ranks = tab.rank(ascending=False)
        tab = tab.join(ranks.add_suffix("_rank"))
        tab["mean_rank"] = ranks.mean(axis=1)
        tab["target"] = t
        rows.append(tab.reset_index(names="feature").sort_values("mean_rank"))
        agree.append({"target": t, "kendall_w": kendall_w(ranks.to_numpy().T), "n_methods": 4, "n_features": len(T1_COLS)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(agree)


def covariation(bf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summ, dist, series = [], [], []
    for t in TARGETS:
        y = target_col(t)
        d = bf.dropna(subset=[y]).copy()
        d["anom"] = d[y] - d.groupby("Site")[y].transform("mean")
        for name, sub in [("all", d), ("river", d[d.group.astype(str) != "treatment"]),
                          ("treatment", d[d.group.astype(str) == "treatment"])]:
            m = sub.groupby("Date").anom.transform("mean")
            summ.append({"target": t, "subset": name, "measure": "shared_monthly_share",
                         "value": 1 - (sub.anom - m).var() / sub.anom.var()})
        W = d.pivot_table(index="Date", columns="Site", values="anom")
        C = W.corr(min_periods=10).to_numpy()
        iu = np.triu_indices_from(C, 1)
        summ.append({"target": t, "subset": "all", "measure": "median_pairwise_corr", "value": float(np.nanmedian(C[iu]))})
        for k in range(1, len(MAIN_STEM_ORDER)):
            vals = [W[MAIN_STEM_ORDER[i]].corr(W[MAIN_STEM_ORDER[i + k]]) for i in range(len(MAIN_STEM_ORDER) - k)]
            dist.append({"target": t, "separation": k, "median_corr": float(np.nanmedian(vals)), "n_pairs": len(vals)})
        X = d[T1_COLS]
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV()).fit(X, d[y])
        res = d[y] - pipe.predict(X)
        res = res - res.groupby(d.Site).transform("mean")
        m = res.groupby(d.Date).transform("mean")
        summ.append({"target": t, "subset": "all", "measure": "shared_share_after_chemistry",
                     "value": 1 - (res - m).var() / res.var()})
        s = d.groupby("Date").anom.mean().rename("common_signal").reset_index().assign(target=t)
        series.append(s)
    return pd.DataFrame(summ), pd.DataFrame(dist), pd.concat(series)


def scenario_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    R = ROOT / "results"
    head = pd.read_csv(R / "headline.csv")
    conf = pd.read_csv(R / "conformal.csv")
    conf = conf[conf.group == "all"][["target", "scenario", "scheme", "coverage", "fold_change"]]
    tab = head.merge(conf, on=["target", "scenario", "scheme"], how="left")
    sp = pd.read_csv(R / "sentinel_predictions.csv")
    sp = sp[sp["qa"].astype(str).str.lower() == "true"].assign(seed=0)
    curve = []
    for t in TARGETS:
        for k, d in sp[sp.target == t].groupby("k"):
            pt, lo, hi = boot_r2(d, "month")
            curve.append({"target": t, "k": int(k), "r2": pt, "ci_lo": lo, "ci_hi": hi})
    return tab, pd.DataFrame(curve)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    df = df[df.n_measured > 2]
    bf = base_frame()
    descriptive(df).to_csv(OUT / "table1_descriptive.csv", index=False); print("A1 done", flush=True)
    treatment_changes(df).to_csv(OUT / "treatment_changes.csv", index=False); print("A2 done", flush=True)
    imp, agree = consensus_importance(bf)
    imp.to_csv(OUT / "importance_consensus.csv", index=False); agree.to_csv(OUT / "importance_agreement.csv", index=False)
    print("A3 done", agree.round(3).to_dict("records"), flush=True)
    summ, dist, series = covariation(bf)
    summ.to_csv(OUT / "covariation_summary.csv", index=False); dist.to_csv(OUT / "covariation_distance.csv", index=False)
    series.to_csv(OUT / "covariation_series.csv", index=False); print("A4 done", flush=True)
    tab, curve = scenario_tables()
    tab.to_csv(OUT / "table_scenarios.csv", index=False); curve.to_csv(OUT / "sentinel_curve.csv", index=False)
    print("A6 done", flush=True)


if __name__ == "__main__":
    main()
