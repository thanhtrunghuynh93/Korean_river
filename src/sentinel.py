"""Sentinel sampling design: lab-test DBP at k sentinel sites each month, predict the rest.

Deployment being simulated
  * A calibration campaign (this 16-month dataset) gives every site's normal log DBP level.
  * Afterwards, each month: water-quality chemistry is measured at every site, but THMFP / HAAFP are
    lab-tested only at the k sentinel sites.
  * Non-sentinel sites are predicted from their own chemistry + calibration site mean + how far the
    sentinels are from *their* normal level this month (the shared monthly signal).
  Non-sentinel sites therefore have NO DBP history after calibration: own lag features are not used.

Evaluation (leave-months-out, 8 folds of 2 months)
  * Site means and model weights come from training months only.
  * Sentinels are chosen by greedy forward selection inside each outer fold, scored by an inner
    leave-months-out on the training months only.
  * Scores are pooled R² (log10 target) on non-sentinel rows of the held-out months.

    uv run python -m src.sentinel            # both targets, k = 0..8, with and without the QA screen
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing.load import TARGETS
from src.features import SITE_COL, base_frame, other_target, target_col, tier_columns

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASE_COLS = [c for c in tier_columns(2) if c != SITE_COL]     # chemistry + group/stage/position/month
SENT_COLS = ["site_mean", "sent_anom", "sent_anom_other", "sent_near_anom", "sent_near_dist"]
K_LIST = [0, 1, 2, 3, 4, 6, 8]
N_OUTER, N_INNER = 8, 4
ALPHAS = np.logspace(-2, 3, 11)
QA_K = 4.0            # QA screen: drop a sentinel reading > QA_K robust SDs from that site's normal level
QA_FLOOR = 0.05       # minimum robust SD (log10) so near-constant sites are not over-screened


def candidate_sites(d: pd.DataFrame) -> list[str]:
    """Sentinel candidates: every river site (main stem, tributary, reservoir) + Gumi raw water (6)."""
    river = d.loc[d.group != "treatment", "Site"].unique().tolist()
    return sorted(set(river) | {"6"})


def prepare(target: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = base_frame() if df is None else df
    d = df.copy()
    d["y"] = d[target_col(target)]
    d["y_other"] = d[target_col(other_target(target))]
    d["group"] = d["group"].astype(str)
    d = d[d.y.notna()].reset_index(drop=True)
    return d


def sentinel_features(d: pd.DataFrame, train_idx: np.ndarray, sentinels: list[str], qa: bool = True) -> pd.DataFrame:
    """Features for every row given the training rows (for site means) and the sentinel set.

    site_mean       calibration mean log DBP of the row's site (training rows; fallback group, then global)
    sent_anom       mean over sentinels measured that month of (log DBP - sentinel's calibration mean)
    sent_anom_other same for the other DBP target (both are measured at a sentinel)
    sent_near_anom  anomaly of the nearest main-stem sentinel (main-stem rows only)
    sent_near_dist  distance in main-stem steps to that sentinel
    qa=True screens out a sentinel reading whose anomaly exceeds QA_K robust SDs (1.4826 x MAD of that
    site's training-month anomalies, floored at QA_FLOOR) - a lab-error guard; thresholds use training rows only.
    A row's own target never enters its own features unless the row is itself a sentinel."""
    tr = d.iloc[train_idx]
    out = pd.DataFrame(index=d.index)
    feats = {}
    for col, name in [("y", ""), ("y_other", "_other")]:
        site_m = tr.groupby("Site")[col].mean()
        grp_m = tr.groupby("group")[col].mean()
        glob = tr[col].mean()
        sm = d.Site.map(site_m).fillna(d.group.map(grp_m)).fillna(glob)
        feats[f"site_mean{name}"] = sm
        anom = d[col] - sm
        is_s = d.Site.isin(sentinels) & d[col].notna()
        if qa:
            tr_anom = anom.iloc[train_idx]
            mad = (tr_anom - tr_anom.groupby(d.Site.iloc[train_idx]).transform("median")).abs() \
                .groupby(d.Site.iloc[train_idx]).median()
            thr = (QA_K * np.maximum(1.4826 * mad, QA_FLOOR)).reindex(d.Site.unique()).fillna(QA_K * QA_FLOOR)
            is_s &= anom.abs() <= d.Site.map(thr)
        feats[f"sent_anom{name}"] = d.Date.map(anom[is_s].groupby(d.Date[is_s]).mean())
        if name == "":
            feats["_anom"] = anom.where(is_s)
    out["site_mean"] = feats["site_mean"]
    out["sent_anom"] = feats["sent_anom"]
    out["sent_anom_other"] = feats["sent_anom_other"]

    # nearest main-stem sentinel, same month
    ms_sent = [s for s in sentinels if s.startswith("M")]
    near_anom = pd.Series(np.nan, index=d.index)
    near_dist = pd.Series(np.nan, index=d.index)
    if ms_sent:
        pos = {s: int(s[1:]) for s in ms_sent}
        a = feats["_anom"]
        lookup = pd.Series(a.to_numpy(), index=pd.MultiIndex.from_arrays([d.Site, d.Date])).dropna()
        is_ms = d.ms_position.notna()
        for i in d.index[is_ms]:
            p = d.at[i, "ms_position"]
            order = sorted(ms_sent, key=lambda s: (abs(pos[s] - p), pos[s]))
            for s in order:
                if s == d.at[i, "Site"]:
                    continue                                     # a sentinel never uses itself
                v = lookup.get((s, d.at[i, "Date"]))
                if v is not None and not np.isnan(v):
                    near_anom[i], near_dist[i] = v, abs(pos[s] - p)
                    break
    out["sent_near_anom"] = near_anom
    out["sent_near_dist"] = near_dist
    return out


def _model():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), RidgeCV(alphas=ALPHAS))


def fit_predict(d: pd.DataFrame, tr: np.ndarray, te: np.ndarray, sentinels: list[str], qa: bool = True) -> np.ndarray:
    X = pd.concat([d[BASE_COLS], sentinel_features(d, tr, sentinels, qa)], axis=1)
    return _model().fit(X.iloc[tr], d.y.iloc[tr]).predict(X.iloc[te])


def _score(d, tr, te, sentinels, qa: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Predictions and true values on the non-sentinel rows of `te`."""
    te_ns = te[~d.Site.iloc[te].isin(sentinels).to_numpy()]
    return fit_predict(d, tr, te_ns, sentinels, qa), d.y.iloc[te_ns].to_numpy()


def greedy_select(d: pd.DataFrame, train_idx: np.ndarray, k_max: int, candidates: list[str], qa: bool = True) -> list[str]:
    """Forward selection of sentinels using only `train_idx` rows (inner leave-months-out)."""
    sub = d.iloc[train_idx].reset_index(drop=True)
    inner = list(GroupKFold(N_INNER).split(sub, groups=sub.Date.astype(str)))
    chosen: list[str] = []
    for _ in range(k_max):
        best, best_r2 = None, -np.inf
        for c in candidates:
            if c in chosen:
                continue
            s = chosen + [c]
            yp, yt = [], []
            for itr, ite in inner:
                p, t = _score(sub, itr, ite, s, qa)
                yp.append(p); yt.append(t)
            r2 = r2_score(np.concatenate(yt), np.concatenate(yp))
            if r2 > best_r2:
                best, best_r2 = c, r2
        chosen.append(best)
    return chosen


def evaluate(target: str, df: pd.DataFrame | None = None, k_list=K_LIST, qa: bool = True, verbose=True):
    d = prepare(target, df)
    cands = candidate_sites(d)
    outer = list(GroupKFold(N_OUTER).split(d, groups=d.Date.astype(str)))
    preds, choices = [], []
    for f, (tr, te) in enumerate(outer):
        t0 = time.time()
        order = greedy_select(d, tr, max(k_list), cands, qa)
        choices.append({"target": target, "qa": qa, "fold": f, "order": ",".join(order)})
        for k in k_list:
            s = order[:k]
            p, y = _score(d, tr, te, s, qa)
            te_ns = te[~d.Site.iloc[te].isin(s).to_numpy()]
            preds.append(pd.DataFrame({"target": target, "qa": qa, "k": k, "fold": f, "Site": d.Site.iloc[te_ns].to_numpy(),
                                       "Date": d.Date.iloc[te_ns].dt.strftime("%Y-%m-%d").to_numpy(),
                                       "group": d.group.iloc[te_ns].to_numpy(), "y": y, "y_pred": p}))
        if verbose:
            print(f"{target} qa={qa} fold {f}: sentinels {order}  [{time.time() - t0:.0f}s]", flush=True)
    pred = pd.concat(preds, ignore_index=True)
    rows = []
    for k, g in pred.groupby("k"):
        row = {"target": target, "qa": qa, "k": k, "lab_share": k / d.Site.nunique(), "n_eval": len(g),
               "r2": r2_score(g.y, g.y_pred),
               "r2_fold_sd": float(g.groupby("fold").apply(lambda x: r2_score(x.y, x.y_pred), include_groups=False).std(ddof=0)),
               "medape": float(np.median(np.abs(10 ** g.y_pred - 10 ** g.y) / 10 ** g.y))}
        for grp, gg in g.groupby("group"):
            row[f"r2_{grp}"] = r2_score(gg.y, gg.y_pred)
        rows.append(row)
    return pred, pd.DataFrame(rows), pd.DataFrame(choices)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS)
    a = ap.parse_args()
    df = base_frame()
    P, S, C = [], [], []
    for t in ([a.target] if a.target else TARGETS):
        for qa in (False, True):                 # False = unscreened run, kept for the record
            p, s, c = evaluate(t, df, qa=qa)
            P.append(p); S.append(s); C.append(c)
            print(s.round(3).to_string(index=False))
    RESULTS.mkdir(exist_ok=True)
    pd.concat(P).to_csv(RESULTS / "sentinel_predictions.csv", index=False)
    pd.concat(S).to_csv(RESULTS / "sentinel_summary.csv", index=False)
    pd.concat(C).to_csv(RESULTS / "sentinel_choices.csv", index=False)


if __name__ == "__main__":
    main()
