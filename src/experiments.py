"""Pre-registered feature experiments for improvement round 1. All extend tier 3.

Selection rule (fixed in advance): best mean pooled R² over the `month` and `site` schemes.
Deltas smaller than the seed-to-seed sd (~0.03) are reported as no change.
"""
from src.features import BASIN_COLS, CHEM_EXTRA_COLS, XLAG_COLS, tier_columns

_T3 = tier_columns(3)

EXPERIMENTS: dict[str, list[str]] = {
    "t3": _T3,                                                   # baseline under the multi-seed protocol
    "t3_chem": _T3 + CHEM_EXTRA_COLS,                            # composition ratios + sparse extras
    "t3_xlag": _T3 + XLAG_COLS,                                  # other target's lags, own lag-2, upstream lag-1
    "t3_basin": _T3 + BASIN_COLS,                                # same-month network means, downstream, plant raw
    "t4_all": _T3 + CHEM_EXTRA_COLS + XLAG_COLS + BASIN_COLS,    # everything
}
EXPERIMENT_ORDER = list(EXPERIMENTS)

NEW_FEATURE_BLOCK = {c: "chem" for c in CHEM_EXTRA_COLS} | {c: "xlag" for c in XLAG_COLS} | {c: "basin" for c in BASIN_COLS}
