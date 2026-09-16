"""Feature tiers for the THMFP / HAAFP predictors.

Tier 1  chemistry   core-10 water-quality variables (log10 where skewed)
Tier 2  context     + site group, treatment stage, WTP, main-stem position, month sin/cos,
                    + site target-encoding (fit inside each training fold by `SiteTargetEncoder`)
Tier 3  network     + previous-month same-site values, same-site history mean, upstream same-month values

Tier 3 features use only *earlier dates* at the same site or *other sites* at the same date, never the
row's own target. Rows are identified by the index of the frame returned by `base_frame()`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing.load import (MAIN_STEM_ORDER, STAGE_ORDER, TARGETS, WTP_RAW_SITES, load_clean)

CORE10 = ["Temp", "pH", "Turbidity", "EC", "Br", "TOC", "Biopolymer", "HS", "BB", "LMWN"]
SKEWED = ["Turbidity", "EC", "SS", "Br", "BOD", "COD", "NH3N", "TOC", "Biopolymer", "HS", "BB", "LMWN"]
GROUPS = ["main_stem", "tributary", "reservoir", "treatment"]
WTPS = ["Gumi", "Goryeong", "Bansong"]
SITE_COL = "Site"                 # carried in tier >= 2 matrices for the fold-aware encoder
TIERS = (1, 2, 3)


def _feat_name(col: str) -> str:
    return f"log_{col}" if col in SKEWED else col


T1_COLS = [_feat_name(c) for c in CORE10]
T2_EXTRA = ([f"group_{g}" for g in GROUPS] + [f"wtp_{w}" for w in WTPS]
            + ["stage_idx", "ms_position", "month_sin", "month_cos", SITE_COL])
T3_EXTRA = ["lag1_y", "lag1_log_TOC", "lag1_log_HS", "hist_mean_y", "up_y", "up_log_TOC"]


def tier_columns(tier: int) -> list[str]:
    if tier == 1:
        return list(T1_COLS)
    if tier == 2:
        return T1_COLS + T2_EXTRA
    if tier == 3:
        return T1_COLS + T2_EXTRA + T3_EXTRA
    raise ValueError(f"tier must be one of {TIERS}")


def target_col(target: str) -> str:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    return f"y_{target}"


# --------------------------------------------------------------------------- base frame

def _log10_pos(s: pd.Series) -> pd.Series:
    return np.log10(s.where(s > 0))


def upstream_map(df: pd.DataFrame) -> dict[str, str]:
    """Site -> upstream site (same month). Main stem: M(k-1). WTP trains: previous stage at the same
    plant, head of the train = the plant's raw-water site. Tributaries / reservoirs: absent (NaN)."""
    up: dict[str, str] = {b: a for a, b in zip(MAIN_STEM_ORDER[:-1], MAIN_STEM_ORDER[1:])}
    raw_of = {v: k for k, v in WTP_RAW_SITES.items()}          # wtp -> raw site code
    train = df[(df.group == "treatment") & df.wtp.notna()][["Site", "wtp", "stage"]].drop_duplicates()
    train["stage_idx"] = train["stage"].map(STAGE_ORDER.index)
    for wtp, sub in train.groupby("wtp"):
        chain = [raw_of[wtp]] + sub.sort_values("stage_idx")["Site"].astype(str).tolist()
        chain = [c for i, c in enumerate(chain) if c not in chain[:i]]   # Gumi raw (6) appears in both
        for a, b in zip(chain[:-1], chain[1:]):
            up[b] = a
    return up


def base_frame() -> pd.DataFrame:
    """Cleaned rows + all tier features + log10 targets. Drops the 3 fully blank rows."""
    df = load_clean()
    df = df[df.n_measured > 2].copy()
    df["Site"] = df["Site"].astype(str)
    df = df.sort_values(["Site", "Date"]).reset_index(drop=True)

    # targets
    for t in TARGETS:
        df[target_col(t)] = _log10_pos(df[t])

    # tier 1
    for c in CORE10:
        df[_feat_name(c)] = _log10_pos(df[c]) if c in SKEWED else df[c]

    # tier 2 (except the site target-encoding, which is fold-aware)
    for g in GROUPS:
        df[f"group_{g}"] = (df.group.astype(str) == g).astype(int)
    for w in WTPS:
        df[f"wtp_{w}"] = (df.wtp == w).astype(int)
    df["stage_idx"] = df["stage"].map(lambda s: STAGE_ORDER.index(s) if isinstance(s, str) else 0).astype(int)
    df["ms_position"] = df["Site"].map({s: i + 1 for i, s in enumerate(MAIN_STEM_ORDER)})
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)

    # tier 3: same-site previous month (dates are monthly, so shift(1) within site == previous month)
    g = df.groupby("Site", sort=False)
    df["lag1_log_TOC"] = g["log_TOC"].shift(1)
    df["lag1_log_HS"] = g["log_HS"].shift(1)
    up = upstream_map(df)
    df["up_site"] = df["Site"].map(up)
    key = df.set_index(["Site", "Date"])
    df["up_log_TOC"] = key["log_TOC"].reindex(pd.MultiIndex.from_arrays([df.up_site, df.Date])).to_numpy()
    return df


def add_target_context(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Tier-3 columns that depend on the target: lag-1 target, history mean, upstream same-month target."""
    out = df.copy()
    y = target_col(target)
    g = out.groupby("Site", sort=False)[y]
    out["lag1_y"] = g.shift(1)
    out["hist_mean_y"] = g.transform(lambda s: s.shift(1).expanding().mean())
    key = out.set_index(["Site", "Date"])[y]
    out["up_y"] = key.reindex(pd.MultiIndex.from_arrays([out.up_site, out.Date])).to_numpy()
    return out


def build(target: str, tier: int, df: pd.DataFrame | None = None):
    """Return X (DataFrame), y (Series, log10 target), meta (Site, Date, group) with target present."""
    df = base_frame() if df is None else df
    df = add_target_context(df, target)
    y = df[target_col(target)]
    keep = y.notna()
    X = df.loc[keep, tier_columns(tier)].copy()
    meta = df.loc[keep, ["Site", "Date", "group", "wtp", "stage", "lag1_y"]].copy()
    meta["group"] = meta["group"].astype(str)
    return X, y[keep], meta


# --------------------------------------------------------------------------- fold-aware encoder

class SiteTargetEncoder(BaseEstimator, TransformerMixin):
    """Replace the `Site` column with a smoothed mean of the *training* target per site.

    Unseen sites (leave-site-out folds) receive the global training mean. m-estimate smoothing:
    te = (n * mean_site + m * mean_global) / (n + m).
    """

    def __init__(self, site_col: str = SITE_COL, m: float = 5.0):
        self.site_col = site_col
        self.m = m

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_mean_ = float(y.mean())
        if self.site_col in X.columns:
            stats = y.groupby(X[self.site_col].astype(str)).agg(["sum", "count"])
            self.mapping_ = ((stats["sum"] + self.m * self.global_mean_) / (stats["count"] + self.m)).to_dict()
        else:
            self.mapping_ = {}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.site_col in X.columns:
            te = X[self.site_col].astype(str).map(self.mapping_).astype(float).fillna(self.global_mean_)
            X = X.drop(columns=[self.site_col])
            X["site_te"] = te.to_numpy()
        return X

    def get_feature_names_out(self, input_features=None):
        cols = [c for c in (input_features or []) if c != self.site_col]
        return np.array(cols + ["site_te"]) if input_features and self.site_col in input_features else np.array(cols)


if __name__ == "__main__":
    for t in TARGETS:
        for tier in TIERS:
            X, y, meta = build(t, tier)
            print(t, "tier", tier, X.shape, "| NaN share per col (max):", round(X.isna().mean().max(), 2))
