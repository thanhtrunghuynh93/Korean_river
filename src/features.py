"""Feature tiers and experiment feature sets for the THMFP / HAAFP predictors.

Tier 1  chemistry   core-10 water-quality variables (log10 where skewed)
Tier 2  context     + site group, treatment stage, WTP, main-stem position, month sin/cos,
                    + site target-encoding (fit inside each training fold by `SiteTargetEncoder`)
Tier 3  network     + previous-month same-site values, same-site history mean, upstream same-month values

Improvement-round feature blocks (see `src/experiments.py`), all on top of tier 3:
  CHEM_EXTRA_COLS  LC-OCD fractions as ratios to TOC, Temp x log TOC, sparse extra variables (NaN-tolerant)
  XLAG_COLS        other target's lag / history / upstream values, own lag-2, upstream site's lag-1
  BASIN_COLS       leave-one-out same-month means over the other river sites / own group, downstream
                   neighbour, plant raw-water target

Every lag feature uses only *earlier dates* at the same site; every same-month feature uses only *other sites*
(the row's own target is never included). Rows are identified by the index of `base_frame()`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing.load import (MAIN_STEM_ORDER, STAGE_ORDER, TARGETS, WTP_RAW_SITES, load_clean)

CORE10 = ["Temp", "pH", "Turbidity", "EC", "Br", "TOC", "Biopolymer", "HS", "BB", "LMWN"]
EXTRA_CHEM_RAW = ["COD", "SS", "DO", "BOD", "NH3N", "SUVA", "Aromaticity", "MolWeight"]
SKEWED = ["Turbidity", "EC", "SS", "Br", "BOD", "COD", "NH3N", "TOC", "Biopolymer", "HS", "BB", "LMWN"]
GROUPS = ["main_stem", "tributary", "reservoir", "treatment"]
WTPS = ["Gumi", "Goryeong", "Bansong"]
SITE_COL = "Site"                 # carried in tier >= 2 matrices for the fold-aware encoder
TIERS = (1, 2, 3)
MIN_BASIN_N = 3


def _feat_name(col: str) -> str:
    return f"log_{col}" if col in SKEWED else col


T1_COLS = [_feat_name(c) for c in CORE10]
T2_EXTRA = ([f"group_{g}" for g in GROUPS] + [f"wtp_{w}" for w in WTPS]
            + ["stage_idx", "ms_position", "month_sin", "month_cos", SITE_COL])
T3_EXTRA = ["lag1_y", "lag1_log_TOC", "lag1_log_HS", "hist_mean_y", "up_y", "up_log_TOC"]

RATIO_BASES = ["HS", "BB", "Biopolymer", "LMWN", "Br"]
CHEM_RATIOS = [f"ratio_{c}_TOC" for c in RATIO_BASES] + ["Temp_x_logTOC"]
CHEM_EXTRA_COLS = CHEM_RATIOS + [_feat_name(c) for c in EXTRA_CHEM_RAW]
XLAG_COLS = ["lag1_y_other", "hist_mean_y_other", "up_y_other", "lag2_y", "up_lag1_y"]
BASIN_COLS = ["basin_y", "basin_log_TOC", "basin_log_HS", "grp_y", "down_y", "down_log_TOC", "plant_raw_y"]


def tier_columns(tier: int) -> list[str]:
    if tier == 1:
        return list(T1_COLS)
    if tier == 2:
        return T1_COLS + T2_EXTRA
    if tier == 3:
        return T1_COLS + T2_EXTRA + T3_EXTRA
    raise ValueError(f"tier must be one of {TIERS}")


def feature_columns(spec: int | str) -> list[str]:
    """Column list for an int tier or a named experiment (see src/experiments.py)."""
    if isinstance(spec, str) and spec.isdigit():
        spec = int(spec)
    if isinstance(spec, int):
        return tier_columns(spec)
    from src.experiments import EXPERIMENTS          # local import: experiments.py imports this module
    if spec not in EXPERIMENTS:
        raise ValueError(f"unknown feature spec {spec!r}; tiers {TIERS} or one of {list(EXPERIMENTS)}")
    return list(EXPERIMENTS[spec])


def spec_label(spec: int | str) -> str:
    return f"tier{spec}" if isinstance(spec, int) or str(spec).isdigit() else str(spec)


def target_col(target: str) -> str:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    return f"y_{target}"


def other_target(target: str) -> str:
    return [t for t in TARGETS if t != target][0]


# --------------------------------------------------------------------------- site graph

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


def downstream_map(df: pd.DataFrame) -> dict[str, str]:
    """Site -> downstream site (same month): the inverse of `upstream_map`.

    Raw-water intakes coded as main-stem sites (M9, M14) keep their river downstream neighbour, not the
    treatment train, so the map stays one-to-one on the main stem."""
    down: dict[str, str] = {}
    for b, a in upstream_map(df).items():
        if a in MAIN_STEM_ORDER and b not in MAIN_STEM_ORDER:
            continue                                   # M9 -> Goryeong train, M14 -> Bansong train: skip
        down[a] = b
    return down


def _same_month_lookup(df: pd.DataFrame, site_col: str, value_col: str) -> np.ndarray:
    """value_col of (df[site_col], df.Date) looked up in df itself; NaN where the partner row is absent."""
    key = df.set_index(["Site", "Date"])[value_col]
    key = key[~key.index.duplicated()]
    return key.reindex(pd.MultiIndex.from_arrays([df[site_col], df["Date"]])).to_numpy()


def _loo_mean(values: pd.Series, by: pd.Series, member: pd.Series, min_n: int = MIN_BASIN_N) -> pd.Series:
    """Leave-one-out mean of `values` within groups `by`, computed over rows where `member` is True.

    Rows that are members have their own value removed; non-member rows get the plain group mean.
    NaN when fewer than `min_n` other values are available."""
    v = values.where(member)
    g = v.groupby(by)
    total, count = g.transform("sum"), g.transform("count")
    own = member & values.notna()
    total = total - values.where(own, 0.0)
    count = count - own.astype(int)
    out = total / count
    return out.where(count >= min_n)


# --------------------------------------------------------------------------- base frame

def base_frame() -> pd.DataFrame:
    """Cleaned rows + all target-independent features + log10 targets. Drops the 3 fully blank rows."""
    df = load_clean()
    df = df[df.n_measured > 2].copy()
    df["Site"] = df["Site"].astype(str)
    df = df.sort_values(["Site", "Date"]).reset_index(drop=True)

    for t in TARGETS:
        df[target_col(t)] = _log10_pos(df[t])

    # tier 1 + sparse extras
    for c in CORE10 + EXTRA_CHEM_RAW:
        df[_feat_name(c)] = _log10_pos(df[c]) if c in SKEWED else df[c]
    for c in RATIO_BASES:
        df[f"ratio_{c}_TOC"] = df[_feat_name(c)] - df["log_TOC"]
    df["Temp_x_logTOC"] = df["Temp"] * df["log_TOC"]

    # tier 2 (except the site target-encoding, which is fold-aware)
    for g in GROUPS:
        df[f"group_{g}"] = (df.group.astype(str) == g).astype(int)
    for w in WTPS:
        df[f"wtp_{w}"] = (df.wtp == w).astype(int)
    df["stage_idx"] = df["stage"].map(lambda s: STAGE_ORDER.index(s) if isinstance(s, str) else 0).astype(int)
    df["ms_position"] = df["Site"].map({s: i + 1 for i, s in enumerate(MAIN_STEM_ORDER)})
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)

    # tier 3 / network, target-independent parts
    g = df.groupby("Site", sort=False)
    df["lag1_log_TOC"] = g["log_TOC"].shift(1)
    df["lag1_log_HS"] = g["log_HS"].shift(1)
    df["up_site"] = df["Site"].map(upstream_map(df))
    df["down_site"] = df["Site"].map(downstream_map(df))
    df["plant_raw_site"] = df["wtp"].map({v: k for k, v in WTP_RAW_SITES.items()})
    df.loc[df["Site"].isin(WTP_RAW_SITES), "plant_raw_site"] = np.nan      # raw rows: would be their own target
    df["up_log_TOC"] = _same_month_lookup(df, "up_site", "log_TOC")
    df["down_log_TOC"] = _same_month_lookup(df, "down_site", "log_TOC")
    river = df.group.astype(str) != "treatment"
    df["basin_log_TOC"] = _loo_mean(df["log_TOC"], df["Date"], river)
    df["basin_log_HS"] = _loo_mean(df["log_HS"], df["Date"], river)
    return df


def add_target_context(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Target-dependent network columns for `target` (own lags, history, upstream, downstream, basin, plant)."""
    out = df.copy()
    y = target_col(target)
    g = out.groupby("Site", sort=False)[y]
    out["lag1_y"] = g.shift(1)
    out["lag2_y"] = g.shift(2)
    out["hist_mean_y"] = g.transform(lambda s: s.shift(1).expanding().mean())
    out["up_y"] = _same_month_lookup(out, "up_site", y)
    out["up_lag1_y"] = _same_month_lookup(out, "up_site", "lag1_y")
    out["down_y"] = _same_month_lookup(out, "down_site", y)
    out["plant_raw_y"] = _same_month_lookup(out, "plant_raw_site", y)
    river = out.group.astype(str) != "treatment"
    out["basin_y"] = _loo_mean(out[y], out["Date"], river)
    out["grp_y"] = _loo_mean(out[y], out["Date"].astype(str) + "|" + out["group"].astype(str),
                             pd.Series(True, index=out.index))
    return out


def add_cross_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Other target's lag-1, history mean and upstream same-month value, suffixed `_other`."""
    other = add_target_context(df, other_target(target))
    out = df.copy()
    for c in ["lag1_y", "hist_mean_y", "up_y"]:
        out[f"{c}_other"] = other[c].to_numpy()
    return out


def build(target: str, spec: int | str, df: pd.DataFrame | None = None):
    """Return X (DataFrame), y (Series, log10 target), meta (Site, Date, group, ...) with target present."""
    df = base_frame() if df is None else df
    df = add_cross_target(add_target_context(df, target), target)
    y = df[target_col(target)]
    keep = y.notna()
    X = df.loc[keep, feature_columns(spec)].copy()
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
    from src.experiments import EXPERIMENT_ORDER
    frame = base_frame()
    for t in TARGETS:
        for spec in list(TIERS) + EXPERIMENT_ORDER:
            X, y, meta = build(t, spec, frame)
            print(f"{t} {spec_label(spec):9s} {X.shape}  max NaN share {X.isna().mean().max():.2f}")
