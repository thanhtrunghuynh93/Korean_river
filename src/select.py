"""Honest headline numbers: cross-validated model selection, block-bootstrap CIs, conformal intervals.

Works on stored out-of-fold (OOF) predictions (results/predictions.csv, results/sentinel_predictions.csv).

Selection-adjusted score
    Within a deployment scenario there are several candidate (feature spec, model) pairs. Picking the best
    one on the same folds that report it is optimistic. Here, for every outer fold f the candidate is chosen
    by its pooled R² on the *other* folds, and its predictions on fold f are kept. The kept predictions are
    pooled into one selection-adjusted R². (Approximation: the other folds' models were trained on data that
    includes fold f, so a little optimism remains; far less than choosing on all folds.)
    The forward holdout has one fold, so its candidate is chosen on the month-scheme OOF predictions for the
    months before the holdout.

Block bootstrap
    Rows are resampled in blocks: whole months for the `month` and `forward` schemes, whole sites for `site`.
    2000 resamples, 90 % percentile intervals. Differences between two candidates use the same resampled
    blocks (paired), so "A beats B" is claimed only when the interval of the difference excludes 0.

Conformal intervals
    Split-conformal, group-conditional: for each outer fold, the 90 % half-width is the 90 % quantile of
    |OOF residual| on the other folds within the same site group. Coverage is checked on fold f.

    uv run python -m src.select          # writes results/headline.csv, headline_diffs.csv, conformal.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.load import TARGETS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N_BOOT = 2000
ALPHA = 0.10
SEED = 20260925
GAPFILL = {"THMFP": ["t3_basin", "t4_all", "t3", "t3_chem", "t3_xlag", "3"],
           "HAAFP": ["t3_basin", "t4_all", "t3", "t3_chem", "t3_xlag", "3"]}
SCENARIOS = {
    "S1 single sample": lambda t: ["1", "2"],
    "S3 network gap-fill": lambda t: GAPFILL[t],
}
EXCLUDE_MODELS = {"median", "site_median", "persistence"}
SCHEMES = ("month", "site", "forward")


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan


def _block_col(scheme: str) -> str:
    return "Site" if scheme == "site" else "Date"


def load_predictions() -> pd.DataFrame:
    p = pd.read_csv(RESULTS / "predictions.csv", dtype={"tier": str})
    if "seed" not in p.columns:
        p["seed"] = 0
    p["seed"] = p["seed"].fillna(0).astype(int)
    p["cand"] = p["tier"] + "|" + p["model"]
    return p


def candidates(p: pd.DataFrame, target: str, scheme: str, specs: list[str]) -> pd.DataFrame:
    sub = p[(p.target == target) & (p.scheme == scheme) & p.tier.isin(specs) & ~p.model.isin(EXCLUDE_MODELS)]
    # keep candidates evaluated on every fold/seed of this scheme with the full row set
    n = sub.groupby("cand").size()
    return sub[sub.cand.isin(n[n == n.max()].index)]


def cv_select(sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Per seed and outer fold, choose the candidate with the best pooled R² on the other folds."""
    kept, chosen = [], []
    for seed, s in sub.groupby("seed"):
        for fold in s.fold.unique():
            other = s[s.fold != fold]
            score = other.groupby("cand").apply(lambda d: r2(d.y, d.y_pred), include_groups=False)
            best = score.idxmax()
            kept.append(s[(s.fold == fold) & (s.cand == best)])
            chosen.append(best)
    return pd.concat(kept, ignore_index=True), pd.Series(chosen).value_counts()


def _boot_indices(blocks: np.ndarray, rng, n_boot: int):
    uniq = np.unique(blocks)
    pos = {b: np.flatnonzero(blocks == b) for b in uniq}
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        yield np.concatenate([pos[b] for b in draw])


def boot_r2(df: pd.DataFrame, scheme: str, n_boot=N_BOOT, seed=SEED) -> tuple[float, float, float]:
    """Point R² (mean over seeds) and 90 % block-bootstrap interval (seeds averaged within each draw)."""
    rng = np.random.default_rng(seed)
    by_seed = [s.reset_index(drop=True) for _, s in df.groupby("seed")]
    point = float(np.mean([r2(s.y, s.y_pred) for s in by_seed]))
    blocks = _block_col(scheme)
    uniq = np.unique(df[blocks].astype(str))
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = []
        for s in by_seed:
            b = s[blocks].astype(str).to_numpy()
            idx = np.concatenate([np.flatnonzero(b == u) for u in pick])
            vals.append(r2(s.y.to_numpy()[idx], s.y_pred.to_numpy()[idx]))
        draws.append(np.nanmean(vals))
    lo, hi = np.nanquantile(draws, [ALPHA / 2, 1 - ALPHA / 2])
    return point, float(lo), float(hi)


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, scheme: str, n_boot=N_BOOT, seed=SEED):
    """Paired block bootstrap of R²(a) - R²(b) on the rows both candidates predicted."""
    key = ["seed", "fold", "row"]
    m = a.merge(b[key + ["y_pred"]], on=key, suffixes=("_a", "_b"))
    rng = np.random.default_rng(seed)
    by_seed = [s.reset_index(drop=True) for _, s in m.groupby("seed")]
    point = float(np.mean([r2(s.y, s.y_pred_a) - r2(s.y, s.y_pred_b) for s in by_seed]))
    blocks = _block_col(scheme)
    uniq = np.unique(m[blocks].astype(str))
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = []
        for s in by_seed:
            bl = s[blocks].astype(str).to_numpy()
            idx = np.concatenate([np.flatnonzero(bl == u) for u in pick])
            yy = s.y.to_numpy()[idx]
            vals.append(r2(yy, s.y_pred_a.to_numpy()[idx]) - r2(yy, s.y_pred_b.to_numpy()[idx]))
        draws.append(np.nanmean(vals))
    lo, hi = np.nanquantile(draws, [ALPHA / 2, 1 - ALPHA / 2])
    return point, float(lo), float(hi), len(m)


def conformal(df: pd.DataFrame, level: float = 0.90) -> pd.DataFrame:
    """Group-conditional split-conformal coverage and median interval width (log10 and fold-change)."""
    out = []
    for seed, s in df.groupby("seed"):
        s = s.assign(abs_res=(s.y - s.y_pred).abs())
        for fold in s.fold.unique():
            te, cal = s[s.fold == fold], s[s.fold != fold]
            q = cal.groupby("group").abs_res.quantile(level)
            half = te.group.map(q).fillna(cal.abs_res.quantile(level))
            out.append(te.assign(half=half.to_numpy(), covered=(te.abs_res <= half).to_numpy()))
    o = pd.concat(out)
    g = o.groupby("group").agg(coverage=("covered", "mean"), half_width_log10=("half", "median"))
    g.loc["all"] = [o.covered.mean(), o.half.median()]
    g["fold_change"] = 10 ** g.half_width_log10            # predicted value x/÷ this factor
    return g.reset_index()


def main() -> None:
    p = load_predictions()
    heads, diffs, confs = [], [], []
    for t in TARGETS:
        for scen, specs_fn in SCENARIOS.items():
            specs = specs_fn(t)
            for scheme in SCHEMES:
                sub = candidates(p, t, scheme, specs)
                if sub.empty:
                    continue
                best_single = sub.groupby("cand").apply(lambda d: r2(d.y, d.y_pred), include_groups=False).idxmax()
                if scheme == "forward":
                    # single fold: choose on the month-scheme OOF predictions for months before the holdout
                    # (their models saw the holdout months in training, a small remaining leak; documented)
                    ms = candidates(p, t, "month", specs)
                    ms = ms[pd.to_datetime(ms.Date) < pd.Timestamp("2018-05-01")]
                    pick = ms.groupby("cand").apply(lambda d: r2(d.y, d.y_pred), include_groups=False).idxmax()
                    kept, chosen = sub[sub.cand == pick], pd.Series({pick: 1})
                else:
                    kept, chosen = cv_select(sub)
                pt, lo, hi = boot_r2(kept, scheme)
                bpt, blo, bhi = boot_r2(sub[sub.cand == best_single], scheme)
                heads.append({"target": t, "scenario": scen, "scheme": scheme, "n_candidates": sub.cand.nunique(),
                              "r2_selected": pt, "ci_lo": lo, "ci_hi": hi, "chosen": "; ".join(f"{k} ({v})" for k, v in chosen.items()),
                              "best_single": best_single, "r2_best_single": bpt, "best_ci_lo": blo, "best_ci_hi": bhi})
                print(f"{t} {scen:20s} {scheme:8s} selected R2 {pt:.3f} [{lo:.3f}, {hi:.3f}]  best single {best_single} {bpt:.3f}", flush=True)
                if scheme in ("month", "site"):
                    c = conformal(sub[sub.cand == best_single]).assign(target=t, scenario=scen, scheme=scheme, cand=best_single)
                    confs.append(c)
        # paired comparisons that the paper makes
        for scheme in SCHEMES:
            pairs = [("2|mixed", "2|ridge"), ("2|mixed", "1|xgb"), ("2|spline_ridge", "2|ridge"), ("2|xgb_mono", "2|xgb"),
                     ("2|xgb_spec", "2|xgb"), ("2|blend", "2|xgb"), ("1|chained", "1|xgb"),
                     (f"{GAPFILL[t][0]}|ridge", f"{GAPFILL[t][0]}|xgb"), (f"{GAPFILL[t][1]}|ridge", f"{GAPFILL[t][1]}|xgb"),
                     (f"{GAPFILL[t][0]}|mixed", f"{GAPFILL[t][0]}|ridge"), (f"{GAPFILL[t][1]}|mixed", f"{GAPFILL[t][1]}|ridge"),
                     ("2|ridge", "1|xgb"), ("2|xgb", "1|xgb")]
            s = p[(p.target == t) & (p.scheme == scheme)]
            for a, b in pairs:
                A, B = s[s.cand == a], s[s.cand == b]
                if A.empty or B.empty:
                    continue
                d, lo, hi, n = boot_diff(A, B, scheme)
                diffs.append({"target": t, "scheme": scheme, "a": a, "b": b, "delta_r2": d, "ci_lo": lo, "ci_hi": hi,
                              "n": n, "verdict": "a better" if lo > 0 else ("b better" if hi < 0 else "no clear difference")})
    # sentinel scenario (QA-screened), month scheme only
    sp = pd.read_csv(RESULTS / "sentinel_predictions.csv")
    sp = sp[sp["qa"].astype(str).str.lower() == "true"].assign(seed=0)
    for t in TARGETS:
        for k in (1, 2, 4, 8):
            d = sp[(sp.target == t) & (sp.k == k)]
            pt, lo, hi = boot_r2(d, "month")
            heads.append({"target": t, "scenario": f"S2 sentinel k={k}", "scheme": "month", "n_candidates": 1,
                          "r2_selected": pt, "ci_lo": lo, "ci_hi": hi, "chosen": "ridge (sentinels chosen in-fold)",
                          "best_single": "sentinel ridge", "r2_best_single": pt, "best_ci_lo": lo, "best_ci_hi": hi})
            print(f"{t} S2 sentinel k={k}    month    R2 {pt:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    pd.DataFrame(heads).to_csv(RESULTS / "headline.csv", index=False)
    pd.DataFrame(diffs).to_csv(RESULTS / "headline_diffs.csv", index=False)
    pd.concat(confs).to_csv(RESULTS / "conformal.csv", index=False)


if __name__ == "__main__":
    main()
