"""Revision analyses (P1-A..D, P2-A, P2-B) on top of the nested engine outputs.

    uv run python -m src.rev_analyses      # -> results/revision/analyses/*.csv

Stage sequences used for adjacent-stage changes are inferred from stage names and site numbering in the
workbook (to be verified with the plants):
    Gumi      6 raw -> 7 settled -> 8 filtered -> 9 ozonated -> 10 GAC -> 11 finished
    Goryeong  M9 raw -> 12 settled -> 13 filtered -> 14 ozonated -> 15 GAC -> 16 finished
    Bansong   M14 raw -> 17 ozonated -> 18 settled -> 19 F/A -> 20 finished   (site numbering implies
              pre-ozonation before coagulation; an earlier ordering 18 -> 17 produced a chemically implausible
              22 % TOC increase across "ozonation" and was abandoned)
TOC-normalised formation potential = FP / TOC in the workbook units (mg/L / mg C/L = mg DBP per mg C, reported
x1000 as ug/mg C); units are unverified, so ratios are labelled "per reported unit". TOC is not DOC.
"""
from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from preprocessing.load import GROUP_ORDER, MAIN_STEM_ORDER, NUMERIC, load_clean
from src import rev_eval as E
from src.features import T1_COLS, base_frame, target_col
from src.select import boot_r2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "analyses"
TARGETS = ("THMFP", "HAAFP")
TRAINS = {"Gumi": ["6", "7", "8", "9", "10", "11"], "Goryeong": ["M9", "12", "13", "14", "15", "16"],
          "Bansong": ["M14", "17", "18", "19", "20"]}
CONTEXT = E.CONTEXT
TIERS = {
    "Q1 field": ["Temp", "pH", "log_Turbidity", "log_EC"],
    "Q2 +TOC": ["Temp", "pH", "log_Turbidity", "log_EC", "log_TOC"],
    "Q3 +bromide": ["Temp", "pH", "log_Turbidity", "log_EC", "log_TOC", "log_Br"],
    "Q4 +HS": ["Temp", "pH", "log_Turbidity", "log_EC", "log_TOC", "log_Br", "log_HS"],
    "Q5 +all LC-OCD": ["Temp", "pH", "log_Turbidity", "log_EC", "log_TOC", "log_Br", "log_HS", "log_BB", "log_Biopolymer", "log_LMWN"],
}
Q6_EXTRA = ["SUVA"]                    # aromaticity and molecular weight dropped from the paper (2026-10-09)
RNG = np.random.default_rng(20260926)


def boot_ci(fn, units, n=1000):
    vals = []
    for _ in range(n):
        vals.append(fn(RNG.choice(units, len(units), replace=True)))
    return np.nanquantile(vals, [0.05, 0.95])


# ------------------------------------------------------------------------------------------ P1-B tiers
def tiers(bf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, diffs = [], []
    for t in TARGETS:
        d = E.frame(t, bf)
        need = TIERS["Q5 +all LC-OCD"]
        d = d[d[need].notna().all(axis=1)].reset_index(drop=True)          # common cohort for Q1-Q5
        specs = {name: dict(cols=cols + CONTEXT, level=False, name=name) for name, cols in TIERS.items()}
        preds = {}
        for name in TIERS:
            for scheme, seeds in (("month", (0,)), ("site", (0, 1, 2)), ("forward", (0,))):
                p, _ = E.run(t, specs[name], scheme, d, seeds)
                preds[(name, scheme)] = p
                for fam in ("ridge", "xgb", "selected"):
                    q = p.rename(columns={f"yhat_{fam}": "y_pred"})
                    pt, lo, hi = boot_r2(q, "site" if scheme == "site" else "month", n_boot=1000)
                    rows.append({"target": t, "tier": name, "scheme": scheme, "family": fam, "n": int(len(p) / len(seeds)),
                                 "r2": pt, "ci_lo": lo, "ci_hi": hi})
            print(t, name, "done", flush=True)
        from src.select import boot_diff
        for a, b in [("Q2 +TOC", "Q1 field"), ("Q3 +bromide", "Q2 +TOC"), ("Q4 +HS", "Q3 +bromide"),
                     ("Q5 +all LC-OCD", "Q4 +HS"), ("Q5 +all LC-OCD", "Q3 +bromide")]:
            for scheme in ("month", "site", "forward"):
                for fam in ("ridge", "xgb"):
                    A = preds[(a, scheme)].rename(columns={f"yhat_{fam}": "y_pred"})
                    B = preds[(b, scheme)].rename(columns={f"yhat_{fam}": "y_pred"})
                    dd, lo, hi, n = boot_diff(A, B, "site" if scheme == "site" else "month", n_boot=1000)
                    diffs.append({"target": t, "a": a, "b": b, "scheme": scheme, "family": fam, "delta_r2": dd,
                                  "ci_lo": lo, "ci_hi": hi, "n": n})
        r6, d6 = tiers_q6(t, d)
        rows += r6; diffs += d6
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def tiers_q6(t: str, d: pd.DataFrame) -> tuple[list, list]:
    """Q6 on the matched river subset: Q5 vs Q5 + SUVA on the same rows (`d` = Q1-Q5 common cohort)."""
    from src.select import boot_diff
    rows, diffs, preds = [], [], {}
    # Q6: matched river subset with SUVA; reduced (Q5) vs extended on the same rows
    d6 = d[d[Q6_EXTRA].notna().all(axis=1)].reset_index(drop=True)
    specs6 = {"Q5 subset": dict(cols=TIERS["Q5 +all LC-OCD"] + CONTEXT, level=False, name="Q5 subset"),
              "Q6 +SUVA": dict(cols=TIERS["Q5 +all LC-OCD"] + Q6_EXTRA + CONTEXT, level=False, name="Q6 +SUVA")}
    for name in ("Q5 subset", "Q6 +SUVA"):
        p, _ = E.run(t, specs6[name], "month", d6, (0,))
        preds[(name, "month")] = p
        for fam in ("ridge", "xgb"):
            q = p.rename(columns={f"yhat_{fam}": "y_pred"})
            pt, lo, hi = boot_r2(q, "month", n_boot=1000)
            rows.append({"target": t, "tier": name, "scheme": "month", "family": fam, "n": len(p), "r2": pt, "ci_lo": lo, "ci_hi": hi})
    for fam in ("ridge", "xgb"):
        A = preds[("Q6 +SUVA", "month")].rename(columns={f"yhat_{fam}": "y_pred"})
        B = preds[("Q5 subset", "month")].rename(columns={f"yhat_{fam}": "y_pred"})
        dd, lo, hi, n = boot_diff(A, B, "month", n_boot=1000)
        diffs.append({"target": t, "a": "Q6 +SUVA", "b": "Q5 subset", "scheme": "month", "family": fam,
                      "delta_r2": dd, "ci_lo": lo, "ci_hi": hi, "n": n})
    return rows, diffs


# ------------------------------------------------------------------------------------------ P1-A associations
def associations(bf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, coefs = [], []
    import statsmodels.api as sm
    for t in TARGETS:
        y = target_col(t)
        d = bf[bf[y].notna()].copy(); d["Site"] = d.Site.astype(str)
        for x in ("log_TOC", "log_HS", "log_Br", "Temp"):
            e = d[[y, x, "Site", "Date"]].dropna()
            sites = e.Site.unique()
            by = {s: e[e.Site == s] for s in sites}
            def pooled(ss): z = pd.concat([by[s] for s in ss]); return stats.spearmanr(z[x], z[y])[0]
            def means(ss): z = pd.concat([by[s] for s in ss]).groupby("Site")[[x, y]].mean(); return stats.spearmanr(z[x], z[y])[0] if len(z) > 3 else np.nan
            def within(ss):
                z = pd.concat([by[s] for s in ss]); zc = z[[x, y]] - z.groupby("Site")[[x, y]].transform("mean")
                return stats.spearmanr(zc[x], zc[y])[0]
            for level, fn in (("pooled", pooled), ("site means", means), ("within site", within)):
                lo, hi = boot_ci(fn, sites, 500)
                rows.append({"target": t, "predictor": x, "level": level, "rho": fn(sites), "ci_lo": lo, "ci_hi": hi, "n_sites": len(sites)})
        # site + campaign fixed effects: log FP ~ log TOC + log HS  (elasticities), cluster bootstrap over sites
        e = d[[y, "log_TOC", "log_HS", "Site", "Date"]].dropna().reset_index(drop=True)
        def fe_fit(z):
            X = pd.get_dummies(z[["Site", "Date"]].astype(str), drop_first=True, dtype=float)
            X = pd.concat([z[["log_TOC", "log_HS"]], X], axis=1); X = sm.add_constant(X)
            return sm.OLS(z[y], X).fit().params[["log_TOC", "log_HS"]]
        est = fe_fit(e)
        sites = e.Site.unique()
        bs = []
        for _ in range(300):
            ss = RNG.choice(sites, len(sites), replace=True)
            z = pd.concat([e[e.Site == s].assign(Site=f"{s}_{i}") for i, s in enumerate(ss)], ignore_index=True)
            bs.append(fe_fit(z))
        bs = pd.DataFrame(bs)
        for c in ("log_TOC", "log_HS"):
            coefs.append({"target": t, "term": c, "estimate": est[c], "ci_lo": bs[c].quantile(0.05), "ci_hi": bs[c].quantile(0.95),
                          "model": "log10 FP ~ log10 TOC + log10 HS + site FE + campaign FE"})
    return pd.DataFrame(rows), pd.DataFrame(coefs)


def year_over_year(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df.copy(); d["Site"] = d.Site.astype(str); d["m"] = d.Date.dt.month; d["yr"] = d.Date.dt.year
    for v in ("THMFP", "HAAFP", "TOC", "HS", "Temp", "Turbidity"):
        w = d[d.m.isin([5, 6, 7, 8])].pivot_table(index=["Site", "m"], columns="yr", values=v)
        diff = (np.log10(w[2018]) - np.log10(w[2017])) if v != "Temp" else (w[2018] - w[2017])
        diff = diff.replace([np.inf, -np.inf], np.nan).dropna()
        per_site = diff.groupby(level=0).mean()
        lo, hi = boot_ci(lambda ss: per_site.loc[ss].median(), per_site.index.to_numpy(), 1000)
        rows.append({"variable": v, "scale": "log10 ratio 2018/2017" if v != "Temp" else "difference (reported unit)",
                     "median_site_change": per_site.median(), "ci_lo": lo, "ci_hi": hi, "n_sites": len(per_site),
                     "share_sites_increase": float((per_site > 0).mean())})
    return pd.DataFrame(rows)


def descriptive_extended(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in NUMERIC:
        for g in GROUP_ORDER + ["all"]:
            s = df[v] if g == "all" else df.loc[df.group.astype(str) == g, v]
            s = s.dropna()
            if s.empty: continue
            rows.append({"variable": v, "group": g, "n": len(s), "median": s.median(), "q25": s.quantile(.25), "q75": s.quantile(.75),
                         "p05": s.quantile(.05), "p95": s.quantile(.95)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ P1-C targets + treatment
def discordance(bf: pd.DataFrame) -> pd.DataFrame:
    d = bf[bf.y_THMFP.notna() & bf.y_HAAFP.notna()].copy()
    d["log_ratio"] = d.y_HAAFP - d.y_THMFP
    rows = []
    for g, s in d.groupby(d.group.astype(str)):
        rows.append({"level": "group", "key": g, "n": len(s), "median_log_ratio": s.log_ratio.median(), "q25": s.log_ratio.quantile(.25), "q75": s.log_ratio.quantile(.75)})
    for dt, s in d.groupby("Date"):
        rows.append({"level": "campaign", "key": str(dt.date()), "n": len(s), "median_log_ratio": s.log_ratio.median(), "q25": s.log_ratio.quantile(.25), "q75": s.log_ratio.quantile(.75)})
    for x in ("log_TOC", "log_HS", "ratio_HS_TOC", "log_Br", "Temp"):
        e = d[[x, "log_ratio"]].dropna()
        rows.append({"level": "association", "key": x, "n": len(e), "median_log_ratio": np.nan,
                     "spearman": stats.spearmanr(e[x], e.log_ratio)[0]})
    return pd.DataFrame(rows)


def treatment(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy(); d["Site"] = d.Site.astype(str)
    for t in TARGETS:
        d[f"sp_{t}"] = 1000 * d[t] / d.TOC.where(d.TOC > 0)
    vars_ = ["TOC", "HS", "BB", "Biopolymer", "LMWN", "THMFP", "HAAFP", "sp_THMFP", "sp_HAAFP"]
    rows = []
    for plant, seq in TRAINS.items():
        w = {s: d[d.Site == s].set_index("Date") for s in seq}
        for k in range(1, len(seq)):
            s_prev, s_cur, s_raw = seq[k - 1], seq[k], seq[0]
            stage = d.loc[d.Site == s_cur, "stage"].astype(str).iloc[0]
            for v in vars_:
                pair = pd.concat([w[s_raw][v].rename("raw"), w[s_prev][v].rename("prev"), w[s_cur][v].rename("cur")], axis=1).dropna()
                if len(pair) < 5: continue
                raw_red = 100 * (1 - pair.cur / pair.raw); adj_red = 100 * (1 - pair.cur / pair.prev)
                camp = pair.index.to_numpy()
                ci_raw = boot_ci(lambda cc: np.median(raw_red.loc[cc]), camp, 1000)
                ci_adj = boot_ci(lambda cc: np.median(adj_red.loc[cc]), camp, 1000)
                rows.append({"plant": plant, "step": k, "stage": stage, "from_site": s_prev, "site": s_cur, "variable": v,
                             "n_campaigns": len(pair), "raw_relative_reduction_pct": raw_red.median(),
                             "raw_ci_lo": ci_raw[0], "raw_ci_hi": ci_raw[1], "adjacent_reduction_pct": adj_red.median(),
                             "adj_ci_lo": ci_adj[0], "adj_ci_hi": ci_adj[1], "share_negative_adjacent": float((adj_red < 0).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ P1-D coherence
def coherence(bf: pd.DataFrame, eval_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, loo = [], []
    for t in TARGETS:
        y = target_col(t)
        d = bf[bf[y].notna()].copy(); d["Site"] = d.Site.astype(str); d["group"] = d.group.astype(str)
        d["anom"] = d[y] - d.groupby("Site")[y].transform("mean")
        d["unit"] = d.Site
        for p, seq in TRAINS.items():                        # a plant's stages + intake count as one unit
            d.loc[d.Site.isin(seq), "unit"] = f"plant_{p}"
        W = d.pivot_table(index="Date", columns="Site", values="anom")
        unit_of = d.drop_duplicates("Site").set_index("Site").unit
        river = set(d.Site[d.group != "treatment"]) | {"6"}

        def site_coh(Wm, ref_rule):
            cors, r2s = [], []
            for s in Wm.columns:
                if ref_rule == "all_others":
                    others = [c for c in Wm.columns if c != s]
                    ref = Wm[others].mean(axis=1)
                elif ref_rule == "river_only":
                    others = [c for c in Wm.columns if c != s and c in river]
                    ref = Wm[others].mean(axis=1)
                else:                                        # unit-weighted: average within unit, then across units
                    others = [c for c in Wm.columns if unit_of[c] != unit_of[s]]
                    ref = Wm[others].T.groupby(unit_of[others]).mean().T.mean(axis=1)
                ok = Wm[s].notna() & ref.notna()
                if ok.sum() < 8: continue
                c = np.corrcoef(Wm[s][ok], ref[ok])[0, 1]; cors.append(c); r2s.append(c ** 2)
            return np.median(cors), np.median(r2s), len(cors)

        for rule in ("all_others", "river_only", "unit_weighted"):
            mc, mr2, n = site_coh(W, rule)
            rows.append({"target": t, "analysis": "leave-self-out reference", "rule": rule, "median_corr": mc, "median_r2": mr2, "n_sites": n})
        # two-way additive variance decomposition of within-site variation (site + campaign)
        import statsmodels.formula.api as smf
        m = smf.ols("anom ~ C(Date)", data=d).fit()
        rows.append({"target": t, "analysis": "campaign effect share of within-site variance", "rule": "site-centred, OLS on campaign",
                     "median_corr": np.nan, "median_r2": m.rsquared, "n_sites": d.Site.nunique()})
        # leave one campaign out
        for dt in W.index:
            mc, mr2, _ = site_coh(W.drop(index=dt), "unit_weighted")
            loo.append({"target": t, "dropped": str(pd.Timestamp(dt).date()), "median_corr": mc, "median_r2": mr2})
        mc, mr2, n = site_coh(W[W.index < pd.Timestamp("2018-04-01")], "unit_weighted")
        rows.append({"target": t, "analysis": "excluding Apr-Aug 2018 (sensitivity)", "rule": "unit_weighted", "median_corr": mc, "median_r2": mr2, "n_sites": n})
        # out-of-fold chemistry residuals (S1a, leave-months-out, selected model)
        p = eval_pred[(eval_pred.target == t) & (eval_pred.scenario == "S1a") & (eval_pred.scheme == "month")].copy()
        p["res"] = p.y - p.yhat_selected
        p["res_anom"] = p.res - p.groupby("Site").res.transform("mean")
        Wr = p.pivot_table(index="Date", columns="Site", values="res_anom")
        Wr.index = pd.to_datetime(Wr.index)
        mc, mr2, n = site_coh(Wr, "unit_weighted")
        share = smf.ols("res_anom ~ C(Date)", data=p).fit().rsquared
        rows.append({"target": t, "analysis": "out-of-fold chemistry residuals", "rule": "unit_weighted", "median_corr": mc, "median_r2": mr2, "n_sites": n})
        rows.append({"target": t, "analysis": "campaign share of out-of-fold residual variance", "rule": "OLS on campaign",
                     "median_corr": np.nan, "median_r2": share, "n_sites": p.Site.nunique()})
    return pd.DataFrame(rows), pd.DataFrame(loo)


# ------------------------------------------------------------------------------------------ P2-A / P2-B
def shift(bf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tr = bf.Date <= pd.Timestamp("2018-04-01"); te = ~tr
    for v in ["y_THMFP", "y_HAAFP"] + T1_COLS + ["basin_log_TOC"]:
        a, b = bf.loc[tr, v].dropna(), bf.loc[te, v].dropna()
        rows.append({"variable": v, "train_median": a.median(), "holdout_median": b.median(), "train_p95": a.quantile(.95),
                     "holdout_p95": b.quantile(.95), "share_holdout_above_train_max": float((b > a.max()).mean()),
                     "share_holdout_below_train_min": float((b < a.min()).mean()), "ks_stat": stats.ks_2samp(a, b).statistic})
    core = bf.loc[te, T1_COLS]; lo = bf.loc[tr, T1_COLS].min(); hi = bf.loc[tr, T1_COLS].max()
    rows.append({"variable": "any core input outside training range", "share_holdout_above_train_max": float(((core > hi) | (core < lo)).any(axis=1).mean())})
    return pd.DataFrame(rows)


def performance(pred: pd.DataFrame, bf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, strata = [], []
    thr = {t: bf.loc[bf.Date <= pd.Timestamp("2018-04-01"), target_col(t)].quantile(0.9) for t in TARGETS}
    for (t, scen, scheme), p in pred.groupby(["target", "scenario", "scheme"]):
        for fam in ("ridge", "xgb", "blend", "selected"):
            q = p.rename(columns={f"yhat_{fam}": "y_pred"})
            pt, lo, hi = boot_r2(q, "site" if scheme in ("site", "plant") else "month", n_boot=1000)
            res = q.y_pred - q.y
            rows.append({"target": t, "scenario": scen, "scheme": scheme, "family": fam, "n": int(len(q) / q.seed.nunique()),
                         "r2": pt, "ci_lo": lo, "ci_hi": hi, "rmse_log": float(np.sqrt(np.mean(res ** 2))), "bias_log": float(res.mean()),
                         "mae_mgL": float(np.mean(np.abs(10 ** q.y_pred - 10 ** q.y))),
                         **{f"cov{lev}": float(np.mean(np.abs(res) <= q[f"q{lev}_{fam}"])) for lev in (80, 90)},
                         **{f"f{lev}_median": float(np.median(10 ** q[f"q{lev}_{fam}"])) for lev in (80, 90)},
                         "len90_median_mgL": float(np.median(10 ** q.y_pred * (10 ** q[f"q90_{fam}"] - 10 ** -q[f"q90_{fam}"])))})
            if scheme in ("forward", "month") and fam in ("ridge", "xgb", "blend"):
                hi_fp = q.y > thr[t]
                for name, m in (("high FP (> training P90)", hi_fp), ("other", ~hi_fp)):
                    z = q[m]
                    if len(z) < 5: continue
                    strata.append({"target": t, "scenario": scen, "scheme": scheme, "family": fam, "stratum": name, "n": len(z),
                                   "bias_log": float((z.y_pred - z.y).mean()), "mae_mgL": float(np.mean(np.abs(10 ** z.y_pred - 10 ** z.y))),
                                   "cov90": float(np.mean(np.abs(z.y_pred - z.y) <= z[f"q90_{fam}"]))})
    return pd.DataFrame(rows), pd.DataFrame(strata)


def per_campaign(pred: pd.DataFrame) -> pd.DataFrame:
    p = pred[pred.scheme.isin(["forward", "month"])]
    rows = []
    for (t, scen, scheme, dt), z in p.groupby(["target", "scenario", "scheme", "Date"]):
        for fam in ("ridge", "xgb", "blend"):
            res = z[f"yhat_{fam}"] - z.y
            rows.append({"target": t, "scenario": scen, "scheme": scheme, "Date": dt, "family": fam, "n": len(z),
                         "bias_log": float(res.mean()), "rmse_log": float(np.sqrt((res ** 2).mean())),
                         "cov90": float(np.mean(np.abs(res) <= z[f"q90_{fam}"]))})
    return pd.DataFrame(rows)


def decisions(pred: pd.DataFrame, bf: pd.DataFrame) -> pd.DataFrame:
    """Operational screening at a prespecified threshold (training-period 75th percentile of the target) and
    within-campaign ranking. Threshold is an operational formation-potential level, not a compliance limit."""
    rows = []
    for t in TARGETS:
        thr = bf.loc[bf.Date <= pd.Timestamp("2018-04-01"), target_col(t)].quantile(0.75)
        for (scen, scheme), p in pred[(pred.target == t) & pred.scheme.isin(["forward", "month"])].groupby(["scenario", "scheme"]):
            for fam in ("ridge", "blend"):
                obs = p.y > thr
                for rule, flag in (("point > threshold", p[f"yhat_{fam}"] > thr),
                                   ("upper 80% bound > threshold", p[f"yhat_{fam}"] + p[f"q80_{fam}"] > thr)):
                    tp = int((flag & obs).sum()); fn = int((~flag & obs).sum()); fp = int((flag & ~obs).sum())
                    rows.append({"target": t, "scenario": scen, "scheme": scheme, "family": fam, "decision": "screening", "rule": rule,
                                 "threshold_mgL": 10 ** thr, "n": len(p), "n_exceed": int(obs.sum()),
                                 "sensitivity": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1), "false_negatives": fn,
                                 "review_rate": float(flag.mean())})
                rs, top = [], []
                for dt, z in p.groupby("Date"):
                    if len(z) < 6: continue
                    rs.append(stats.spearmanr(z[f"yhat_{fam}"], z.y)[0])
                    k = max(1, round(0.2 * len(z)))
                    top.append(len(set(z.nlargest(k, f"yhat_{fam}").row) & set(z.nlargest(k, "y").row)) / k)
                rows.append({"target": t, "scenario": scen, "scheme": scheme, "family": fam, "decision": "ranking",
                             "rule": "within-campaign", "n": len(p), "median_spearman": float(np.median(rs)),
                             "top20pct_recall": float(np.mean(top)), "n_campaigns": len(rs)})
    return pd.DataFrame(rows)


def rerun_q6(bf: pd.DataFrame) -> None:
    """Recompute only the Q6 rows of tiers.csv / tier_diffs.csv (Q1-Q5 are unchanged by the Q6 definition)."""
    rows, diffs = [], []
    for t in TARGETS:
        d = E.frame(t, bf)
        d = d[d[TIERS["Q5 +all LC-OCD"]].notna().all(axis=1)].reset_index(drop=True)
        r6, d6 = tiers_q6(t, d); rows += r6; diffs += d6
    tr = pd.read_csv(OUT / "tiers.csv"); dif = pd.read_csv(OUT / "tier_diffs.csv")
    tr = pd.concat([tr[~tr.tier.str.startswith(("Q5 subset", "Q6"))], pd.DataFrame(rows)], ignore_index=True)
    dif = pd.concat([dif[~dif.a.str.startswith("Q6")], pd.DataFrame(diffs)], ignore_index=True)
    tr.to_csv(OUT / "tiers.csv", index=False); dif.to_csv(OUT / "tier_diffs.csv", index=False)
    print(pd.DataFrame(diffs).round(3).to_string(), flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bf = base_frame()
    if "--q6" in sys.argv:
        rerun_q6(bf); return
    df = load_clean(); df = df[df.n_measured > 2]
    pred = pd.read_csv(ROOT / "results/revision/eval/predictions.csv", parse_dates=["Date"])
    perf, strata = performance(pred, bf)
    perf.to_csv(OUT / "performance.csv", index=False); strata.to_csv(OUT / "strata.csv", index=False)
    per_campaign(pred).to_csv(OUT / "per_campaign.csv", index=False)
    decisions(pred, bf).to_csv(OUT / "decisions.csv", index=False)
    shift(bf).to_csv(OUT / "shift.csv", index=False)
    print("performance / decisions / shift done", flush=True)
    a, c = associations(bf); a.to_csv(OUT / "associations.csv", index=False); c.to_csv(OUT / "fe_coefficients.csv", index=False)
    year_over_year(df).to_csv(OUT / "year_over_year.csv", index=False)
    descriptive_extended(df).to_csv(OUT / "descriptive_extended.csv", index=False)
    discordance(bf).to_csv(OUT / "discordance.csv", index=False)
    treatment(df).to_csv(OUT / "treatment_paired.csv", index=False)
    print("associations / treatment done", flush=True)
    coh, loo = coherence(bf, pred); coh.to_csv(OUT / "coherence.csv", index=False); loo.to_csv(OUT / "coherence_leave_one_campaign.csv", index=False)
    print("coherence done", flush=True)
    tr, dif = tiers(bf); tr.to_csv(OUT / "tiers.csv", index=False); dif.to_csv(OUT / "tier_diffs.csv", index=False)
    print("tiers done", flush=True)


if __name__ == "__main__":
    main()
