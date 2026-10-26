"""P1-E frozen sentinel protocol, comparator ladder, selection strategies and robustness.

Primary experiment (retrospective temporal stress test; the period was inspected earlier in the project):
    calibration = May 2017 - Apr 2018; evaluation = May - Aug 2018.
    Chosen on calibration months only and then frozen: sentinel sets, ridge penalty, screening rule, site levels.
    In each evaluation campaign only the sentinels' own formation-potential results are revealed.

Recipient mask (identical rows across k, strategies and comparators):
    sentinel pool  = odd main-stem sites M1..M17, tributaries T1/T3/T5, reservoirs 1/3/5          (15 sites)
    assessment panel = every other site (even main stem, T2/T4/T6, reservoirs 2/4, all WTP samples)  (29 sites)
    Secondary mask: pool = all 30 river + raw-water sites, recipients = non-sentinel rows (as in round 1).

Comparator ladder (same recipients, same calibration history):
    L1 chemistry only                           (= S1a inputs)
    L2 calibration level only
    L3 calibration level + sentinel anomaly     (no chemistry)
    L4 chemistry + calibration level            (= S1b, k = 0)
    L5 chemistry + calibration + sentinel       (= S2)
    L4c / L5c  as L4 / L5 with conventional chemistry only (no LC-OCD fractions)
    B1 last calibration-period value at the site (historical persistence)

Sentinel anomaly for campaign t: aggregate over accepted sentinel readings of (log10 FP - site calibration
level). Screens: `none`; `mad4` (reject |anomaly| > 4 x max(1.4826 MAD_site, 0.05), MAD from calibration months);
aggregation `mean` or `median`. If no reading is accepted the anomaly is missing and the model falls back to
its calibration/chemistry inputs (imputed anomaly = training median, i.e. about 0); fallbacks are counted.

    uv run python -m src.rev_sentinel        # -> results/revision/sentinel/*.csv
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import T1_COLS, T2_EXTRA, SITE_COL, base_frame, target_col
from src.rev_eval import r2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "sentinel"
TARGETS = ("THMFP", "HAAFP")
CUTOFF = pd.Timestamp("2018-04-01")
CONTEXT = [c for c in T2_EXTRA if c != SITE_COL]
CHEM_FULL = T1_COLS + CONTEXT
CHEM_CONV = ["Temp", "pH", "log_Turbidity", "log_EC", "log_Br", "log_TOC"] + CONTEXT
POOL_B = [f"M{i}" for i in range(1, 18, 2)] + ["T1", "T3", "T5", "1", "3", "5"]
FIXED_B = {1: ["M9"], 2: ["M5", "M13"], 4: ["M3", "M7", "M11", "M15"], 8: [f"M{i}" for i in range(1, 16, 2)]}
KS = (0, 1, 2, 4, 8)
ALPHAS = [0.3, 3.0, 30.0, 300.0]
ALPHA_GREEDY = 3.0
QA_K, QA_FLOOR = 4.0, 0.05
LADDER = {
    "L1 chemistry": dict(chem=CHEM_FULL, level=False, sent=False),
    "L2 calibration": dict(chem=[], level=True, sent=False),
    "L3 calibration+sentinel": dict(chem=[], level=True, sent=True),
    "L4 chemistry+calibration": dict(chem=CHEM_FULL, level=True, sent=False),
    "L5 chemistry+calibration+sentinel": dict(chem=CHEM_FULL, level=True, sent=True),
    "L4c conventional+calibration": dict(chem=CHEM_CONV, level=True, sent=False),
    "L5c conventional+calibration+sentinel": dict(chem=CHEM_CONV, level=True, sent=True),
}


def panel() -> pd.DataFrame:
    bf = base_frame()
    d = bf[bf.y_THMFP.notna() & bf.y_HAAFP.notna()].reset_index(drop=True).copy()
    d["Site"] = d.Site.astype(str); d["group"] = d.group.astype(str)
    return d


def levels(d, tr, ycol):
    t = d.iloc[tr]
    return d.Site.map(t.groupby("Site")[ycol].mean()).fillna(d.group.map(t.groupby("group")[ycol].mean())).fillna(t[ycol].mean())


def sentinel_anomaly(d, tr, ycol, sentinels, screen="mad4", agg="mean", perturb=None):
    """Per-row sentinel anomaly (same campaign) + accepted count. `perturb`: dict (Site, Date) -> added log10 bias,
    or (Site, Date) -> None to simulate a missing sentinel."""
    lev = levels(d, tr, ycol)
    anom = d[ycol] - lev
    if perturb:
        for (s, dt), delta in perturb.items():
            m = (d.Site == s) & (d.Date == dt)
            anom[m] = np.nan if delta is None else anom[m] + delta
    is_s = d.Site.isin(sentinels) & anom.notna()
    if screen == "mad4":
        ta = anom.iloc[tr]
        mad = (ta - ta.groupby(d.Site.iloc[tr]).transform("median")).abs().groupby(d.Site.iloc[tr]).median()
        thr = d.Site.map(QA_K * np.maximum(1.4826 * mad, QA_FLOOR)).fillna(QA_K * QA_FLOOR)
        accepted = is_s & (anom.abs() <= thr)
    else:
        accepted = is_s
    grp = anom[accepted].groupby(d.Date[accepted])
    per_month = grp.mean() if agg == "mean" else grp.median()
    return d.Date.map(per_month), d.Date.map(accepted.groupby(d.Date).sum()).fillna(0), int(is_s.sum()), int((is_s & ~accepted).sum())


def features(d, tr, spec, sentinels, screen, agg, ycol, perturb=None):
    X = d[spec["chem"]].copy() if spec["chem"] else pd.DataFrame(index=d.index)
    if spec["level"]:
        X["level"] = levels(d, tr, ycol).to_numpy()
    info = {"n_sent_readings": 0, "n_rejected": 0}
    if spec["sent"] and sentinels:
        other = "y_HAAFP" if ycol == "y_THMFP" else "y_THMFP"
        a, n_acc, n_read, n_rej = sentinel_anomaly(d, tr, ycol, sentinels, screen, agg, perturb)
        a2, *_ = sentinel_anomaly(d, tr, other, sentinels, screen, agg, perturb)
        X["sent_anom"] = a.to_numpy(); X["sent_anom_other"] = a2.to_numpy()
        info = {"n_sent_readings": n_read, "n_rejected": n_rej, "n_accepted": n_acc}
    if X.shape[1] == 0:
        X["const"] = 0.0
    return X, info


def fit(X, y, alpha):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=alpha)).fit(X, y)


def tune_alpha(d, tr, spec, sentinels, screen, agg, ycol, recip_mask):
    """Grouped (campaign) inner CV on training rows; scored on recipient rows only."""
    dtr = d.iloc[tr].reset_index(drop=True)
    folds = list(GroupKFold(4).split(dtr, groups=dtr.Date.astype(str)))
    best, best_s = None, -np.inf
    for a in ALPHAS:
        yp, yt = [], []
        for itr, ite in folds:
            X, _ = features(dtr, itr, spec, sentinels, screen, agg, ycol)
            ite = ite[recip_mask[tr][ite]]
            if len(ite) == 0: continue
            yp.append(fit(X.iloc[itr], dtr[ycol].iloc[itr], a).predict(X.iloc[ite])); yt.append(dtr[ycol].iloc[ite])
        s = r2(np.concatenate(yt), np.concatenate(yp))
        if s > best_s: best, best_s = a, s
    return best


def greedy(d, tr, pool, k_max, targets, screen="mad4", agg="mean", recip_all=None):
    """Forward selection on training campaigns (inner leave-campaigns-out), objective = mean R^2 over `targets`
    on recipient rows (non-sentinel rows within the recipient mask)."""
    dtr = d.iloc[tr].reset_index(drop=True)
    folds = list(GroupKFold(4).split(dtr, groups=dtr.Date.astype(str)))
    spec = LADDER["L5 chemistry+calibration+sentinel"]
    chosen = []
    for _ in range(k_max):
        best, best_s = None, -np.inf
        for c in pool:
            if c in chosen: continue
            s_set = chosen + [c]
            scores = []
            for t in targets:
                ycol = target_col(t); yp, yt = [], []
                for itr, ite in folds:
                    X, _ = features(dtr, itr, spec, s_set, screen, agg, ycol)
                    keep = ~dtr.Site.iloc[ite].isin(s_set).to_numpy()
                    if recip_all is not None:
                        keep &= recip_all[tr][ite]
                    ite2 = ite[keep]
                    yp.append(fit(X.iloc[itr], dtr[ycol].iloc[itr], ALPHA_GREEDY).predict(X.iloc[ite2])); yt.append(dtr[ycol].iloc[ite2])
                scores.append(r2(np.concatenate(yt), np.concatenate(yp)))
            s = float(np.mean(scores))
            if s > best_s: best, best_s = c, s
        chosen.append(best)
    return chosen


def evaluate(d, tr, te_mask, spec, sentinels, screen, agg, t, perturb=None, alpha=None):
    ycol = target_col(t)
    recip = te_mask & ~d.Site.isin(sentinels).to_numpy()
    if alpha is None:
        alpha = tune_alpha(d, tr, spec, sentinels, screen, agg, ycol, ~d.Site.isin(sentinels).to_numpy())
    X, info = features(d, tr, spec, sentinels, screen, agg, ycol, perturb)
    idx = np.flatnonzero(recip)
    p = fit(X.iloc[tr], d[ycol].iloc[tr], alpha).predict(X.iloc[idx])
    fallback = float(np.mean(X["sent_anom"].iloc[idx].isna())) if "sent_anom" in X else np.nan
    return pd.DataFrame({"row": idx, "Site": d.Site.iloc[idx].to_numpy(), "Date": d.Date.iloc[idx].to_numpy(),
                         "group": d.group.iloc[idx].to_numpy(), "y": d[ycol].iloc[idx].to_numpy(), "yhat": p}), \
        {**info, "alpha": alpha, "fallback_rate": fallback}


def summarize(p: pd.DataFrame) -> dict:
    y, yh = p.y.to_numpy(), p.yhat.to_numpy()
    return {"n": len(p), "r2": r2(y, yh), "rmse_log": float(np.sqrt(np.mean((yh - y) ** 2))), "bias_log": float(np.mean(yh - y)),
            "mae_mgL": float(np.mean(np.abs(10 ** yh - 10 ** y)))}


def boot_site(p: pd.DataFrame, n=1000, seed=0):
    rng = np.random.default_rng(seed); sites = p.Site.unique(); pos = {s: np.flatnonzero(p.Site.to_numpy() == s) for s in sites}
    vals = []
    for _ in range(n):
        idx = np.concatenate([pos[s] for s in rng.choice(sites, len(sites))])
        vals.append(r2(p.y.to_numpy()[idx], p.yhat.to_numpy()[idx]))
    return np.nanquantile(vals, [0.05, 0.95])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = panel()
    tr = np.flatnonzero(d.Date <= CUTOFF)
    te = (d.Date > CUTOFF).to_numpy()
    panel_mask = ~d.Site.isin(POOL_B).to_numpy()
    rows, preds = [], []

    # --- sentinel sets chosen on calibration months, frozen
    sets = {}
    for t in TARGETS:
        sets[("greedy", t)] = greedy(d, tr, POOL_B, max(KS), [t], recip_all=panel_mask)
    sets[("greedy_joint", "both")] = greedy(d, tr, POOL_B, max(KS), list(TARGETS), recip_all=panel_mask)
    pd.DataFrame([{"strategy": s, "target": t, "order": ",".join(v)} for (s, t), v in sets.items()]).to_csv(OUT / "frozen_sets.csv", index=False)
    print("frozen sets:", {f"{s}/{t}": v for (s, t), v in sets.items()}, flush=True)

    def run(tag, t, sentinels, spec_name, screen="mad4", agg="mean", perturb=None, keep_pred=False, alpha=None):
        p, info = evaluate(d, tr, te & panel_mask, LADDER[spec_name], sentinels, screen, agg, t, perturb, alpha)
        lo, hi = boot_site(p)
        rows.append({"experiment": tag, "target": t, "k": len(sentinels), "sentinels": ",".join(sentinels),
                     "comparator": spec_name, "screen": screen, "agg": agg, **summarize(p), "ci_lo": lo, "ci_hi": hi, **info})
        if keep_pred:
            preds.append(p.assign(experiment=tag, target=t, k=len(sentinels), comparator=spec_name))
        return p, info

    # --- ladder x k for target-specific greedy, joint greedy, fixed geographic sets
    for t in TARGETS:
        for k in KS:
            for strat in ("greedy", "greedy_joint", "fixed"):
                if k == 0 and strat != "greedy": continue
                s_set = [] if k == 0 else (FIXED_B[k] if strat == "fixed" else sets[(strat, t if strat == "greedy" else "both")][:k])
                for comp in LADDER:
                    if not LADDER[comp]["sent"] and k > 0 and strat != "greedy": continue
                    run(f"forward_{strat}", t, s_set, comp, keep_pred=(comp.startswith("L5 ") or comp.startswith("L4 ")))
        # historical persistence baseline: last calibration value at the site
        last = d.iloc[tr].sort_values("Date").groupby("Site")[target_col(t)].last()
        idx = np.flatnonzero(te & panel_mask)
        p = pd.DataFrame({"row": idx, "Site": d.Site.iloc[idx].to_numpy(), "Date": d.Date.iloc[idx].to_numpy(),
                          "group": d.group.iloc[idx].to_numpy(), "y": d[target_col(t)].iloc[idx].to_numpy()})
        p["yhat"] = p.Site.map(last).fillna(d.iloc[tr][target_col(t)].median()).to_numpy()
        lo, hi = boot_site(p)
        rows.append({"experiment": "forward_baseline", "target": t, "k": 0, "sentinels": "", "comparator": "B1 last calibration value",
                     "screen": "", "agg": "", **summarize(p), "ci_lo": lo, "ci_hi": hi})
        print(t, "ladder done", flush=True)

    # --- random k-sets from the pool (distribution of what an arbitrary network achieves)
    rng = np.random.default_rng(1)
    def rand_eval(t, k, s_set):
        p, _ = evaluate(d, tr, te & panel_mask, LADDER["L5 chemistry+calibration+sentinel"], s_set, "mad4", "mean", t)
        return {"target": t, "k": k, "sentinels": ",".join(s_set), **summarize(p)}
    jobs = [(t, k, list(rng.choice(POOL_B, k, replace=False))) for t in TARGETS for k in (1, 2, 4, 8) for _ in range(200)]
    rand = Parallel(n_jobs=-1)(delayed(rand_eval)(*j) for j in jobs)
    pd.DataFrame(rand).to_csv(OUT / "random_sets.csv", index=False)
    print("random sets done", flush=True)

    # --- robustness on the target-specific greedy k = 4 network
    for t in TARGETS:
        s4 = sets[("greedy", t)][:4]
        base_alpha = None
        for screen, agg in (("none", "mean"), ("mad4", "mean"), ("none", "median")):
            run("robust_screen", t, s4, "L5 chemistry+calibration+sentinel", screen, agg)
        ev_months = sorted(d.Date[te].unique())
        for trial in range(50):
            r = np.random.default_rng(100 + trial)
            miss = {(r.choice(s4), m): None for m in ev_months}               # one sentinel missing each campaign
            run("outage_one_missing", t, s4, "L5 chemistry+calibration+sentinel", "mad4", "mean", miss)
            bias = {(r.choice(s4), m): 0.3 for m in ev_months}                 # one reading biased x2 each campaign
            for screen, agg in (("none", "mean"), ("mad4", "mean"), ("none", "median")):
                run("bias_one_x2", t, s4, "L5 chemistry+calibration+sentinel", screen, agg, bias)
        # lone sentinel missing in every campaign -> full fallback
        s1 = sets[("greedy", t)][:1]
        run("outage_lone_sentinel", t, s1, "L5 chemistry+calibration+sentinel", "mad4", "mean",
            {(s1[0], m): None for m in ev_months})
        print(t, "robustness done", flush=True)

    # --- secondary mask (round-1 style): pool = all river + raw sites, recipients = non-sentinel rows
    pool_a = sorted(set(d.Site[d.group != "treatment"]) | {"6"})
    for t in TARGETS:
        sa = greedy(d, tr, pool_a, 4, [t])
        for k in KS:
            p, info = evaluate(d, tr, te, LADDER["L5 chemistry+calibration+sentinel" if k else "L4 chemistry+calibration"], sa[:k], "mad4", "mean", t)
            lo, hi = boot_site(p)
            rows.append({"experiment": "forward_maskA", "target": t, "k": k, "sentinels": ",".join(sa[:k]),
                         "comparator": "L5" if k else "L4", "screen": "mad4", "agg": "mean", **summarize(p), "ci_lo": lo, "ci_hi": hi, **info})
            if k == 4: break
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    pd.concat(preds).assign(Date=lambda x: pd.to_datetime(x.Date).dt.strftime("%Y-%m-%d")).to_csv(OUT / "predictions.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
