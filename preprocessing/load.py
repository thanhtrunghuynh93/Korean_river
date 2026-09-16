"""Load and clean `data/20260909 DBP Viet.xlsx` (sheet `Total`).

Raw layout: 4 header rows (banner / `Var n` / short name / sub-name), data from row 5,
25 columns. 704 rows = 44 sampling points x 16 monthly dates (2017-05 .. 2018-08).

Usage
-----
>>> from preprocessing.load import load_raw, clean
>>> df = clean(load_raw())
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "20260909 DBP Viet.xlsx"
PROCESSED_PATH = ROOT / "data" / "processed" / "dbp_clean.csv"

NONDETECT = "불검출"  # Korean: "not detected"

COLUMNS = [
    "Date", "Site", "Sample_KR", "Name_EN", "Position",
    "Temp", "pH", "Turbidity", "EC", "DO", "SS",            # Var 1-6  field measurements
    "Br", "BOD", "COD", "NH3N", "TOC",                       # Var 7-11 lab measurements
    "SUVA", "Aromaticity", "MolWeight",                      # Var 12-14 lab (UV / size)
    "Biopolymer", "HS", "BB", "LMWN",                        # Var 15-18 LC-OCD fractions (CDOC)
    "THMFP", "HAAFP",                                        # Var 19-20 outputs
]

VAR_GROUPS: dict[str, list[str]] = {
    "field": ["Temp", "pH", "Turbidity", "EC", "DO", "SS"],
    "lab": ["Br", "BOD", "COD", "NH3N", "TOC"],
    "uv_size": ["SUVA", "Aromaticity", "MolWeight"],
    "lcocd": ["Biopolymer", "HS", "BB", "LMWN"],
    "target": ["THMFP", "HAAFP"],
}
PREDICTORS = VAR_GROUPS["field"] + VAR_GROUPS["lab"] + VAR_GROUPS["uv_size"] + VAR_GROUPS["lcocd"]
TARGETS = VAR_GROUPS["target"]
NUMERIC = PREDICTORS + TARGETS

# Units as inferred from typical Korean water-quality reporting; confirm with the data owner.
UNITS = {
    "Temp": "°C", "pH": "-", "Turbidity": "NTU", "EC": "µS/cm", "DO": "mg/L", "SS": "mg/L",
    "Br": "mg/L", "BOD": "mg/L", "COD": "mg/L", "NH3N": "mg/L", "TOC": "mg/L",
    "SUVA": "L/mg·m", "Aromaticity": "-", "MolWeight": "Da",
    "Biopolymer": "µg/L C", "HS": "µg/L C", "BB": "µg/L C", "LMWN": "µg/L C",
    "THMFP": "mg/L", "HAAFP": "mg/L",
}

# Site codes are blank in the raw file for Goryeong / Bansong treatment-train rows.
# The mapping below (Name_EN -> code) is derived from the rows that do carry a code
# plus the sequential numbering already used for the Gumi train (6-11).
SITE_CODE_BY_NAME: dict[str, str] = {
    "Miryang Reservoir": "1", "Unmun Reservoir": "2",
    "Namgang Reservoir / Namgang Dam Reservoir": "3",
    "Yeongcheon Reservoir (Jukjangcheon)": "4", "Hapcheon Reservoir": "5",
    "Gumi Raw Water": "6", "Gumi Settled Water": "7", "Gumi Filtered Water": "8",
    "Gumi Ozonated Water": "9", "Gumi GAC-Treated Water": "10", "Gumi Finished Water": "11",
    "Goryeong Settled Water": "12", "Goryeong Filtered Water": "13",
    "Goryeong Ozonated Water": "14", "Goryeong GAC-Treated Water": "15",
    "Goryeong Finished Water": "16",
    "Bansong Ozonated Water": "17", "Bansong Settled Water": "18",
    "Bansong F/A-Treated Water": "19", "Bansong Finished Water": "20",
}

# Upstream -> downstream order for the main stem (M1..M18), then tributaries, reservoirs, WTP trains.
MAIN_STEM_ORDER = [f"M{i}" for i in range(1, 19)]
TRIBUTARY_ORDER = [f"T{i}" for i in range(1, 7)]
RESERVOIR_ORDER = ["1", "2", "3", "4", "5"]
TREATMENT_ORDER = [str(i) for i in range(6, 21)]
SITE_ORDER = MAIN_STEM_ORDER + TRIBUTARY_ORDER + RESERVOIR_ORDER + TREATMENT_ORDER

GROUP_ORDER = ["main_stem", "tributary", "reservoir", "treatment"]
GROUP_LABEL = {
    "main_stem": "Main stem (M1–M18)",
    "tributary": "Tributary (T1–T6)",
    "reservoir": "Reservoir (1–5)",
    "treatment": "WTP train (6–20)",
}

STAGE_ORDER = ["raw", "settled", "filtered", "ozonated", "GAC", "F/A", "finished"]
_STAGE_KEYWORDS = [
    ("Raw Water", "raw"), ("Settled", "settled"), ("Filtered", "filtered"),
    ("Ozonated", "ozonated"), ("GAC", "GAC"), ("F/A", "F/A"), ("Finished", "finished"),
]
# Raw-water intakes of the three WTPs are coded as main-stem sites in the file.
WTP_RAW_SITES = {"6": "Gumi", "M9": "Goryeong", "M14": "Bansong"}

SEASON_BY_MONTH = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
SEASON_ORDER = ["spring", "summer", "autumn", "winter"]


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Read the `Total` sheet, skipping the 4 header rows, and name the 25 columns."""
    df = pd.read_excel(path, sheet_name="Total", header=None, skiprows=4)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} columns, got {df.shape[1]}")
    df.columns = COLUMNS
    df["Date"] = pd.to_datetime(df["Date"])
    df["Site"] = df["Site"].astype("string")
    return df


def _group_of(site: str) -> str:
    if site.startswith("M"):
        return "main_stem"
    if site.startswith("T"):
        return "tributary"
    if site in RESERVOIR_ORDER:
        return "reservoir"
    return "treatment"


def _stage_of(name: str) -> str | None:
    for kw, stage in _STAGE_KEYWORDS:
        if kw in name:
            return stage
    return None


def clean(df: pd.DataFrame, nondetect: str = "nan") -> pd.DataFrame:
    """Return a typed, annotated copy of the raw frame.

    nondetect: how to encode the Korean non-detect marker in numeric columns.
        "nan"  -> NaN (default; keeps a `<var>_nd` flag column so the info is not lost)
        "zero" -> 0.0
    """
    if nondetect not in {"nan", "zero"}:
        raise ValueError("nondetect must be 'nan' or 'zero'")
    out = df.copy()

    # 1. non-detects -> flag + NaN/0, then coerce to float
    for col in NUMERIC:
        is_nd = out[col].astype("string").str.strip().eq(NONDETECT).fillna(False)
        if is_nd.any():
            out[f"{col}_nd"] = is_nd.astype(bool)
            log.info("%s: %d non-detect values", col, int(is_nd.sum()))
        out[col] = pd.to_numeric(out[col].where(~is_nd), errors="coerce")
        if nondetect == "zero" and is_nd.any():
            out.loc[is_nd, col] = 0.0

    # 2. literal 0 in all three UV/size variables at once = placeholder, not a measurement
    uv = VAR_GROUPS["uv_size"]
    zero_triplet = (out[uv] == 0).all(axis=1)
    if zero_triplet.any():
        log.info("%d rows with 0 in all of %s -> set to NaN", int(zero_triplet.sum()), uv)
        out.loc[zero_triplet, uv] = np.nan
    out["uv_zero_placeholder"] = zero_triplet

    # 3. fill missing site codes from the English name
    missing_code = out["Site"].isna()
    out.loc[missing_code, "Site"] = out.loc[missing_code, "Name_EN"].map(SITE_CODE_BY_NAME)
    if out["Site"].isna().any():
        unknown = out.loc[out["Site"].isna(), "Name_EN"].unique()
        raise ValueError(f"no site code for: {unknown}")
    out["Site"] = pd.Categorical(out["Site"].astype(str), categories=SITE_ORDER, ordered=True)

    # 4. grouping / treatment-train annotations
    out["group"] = pd.Categorical(out["Site"].astype(str).map(_group_of), categories=GROUP_ORDER, ordered=True)
    out["wtp"] = out["Name_EN"].str.extract(r"^(Gumi|Goryeong|Bansong) ")[0]
    out.loc[out["Site"].astype(str).isin(WTP_RAW_SITES), "wtp"] = out["Site"].astype(str).map(WTP_RAW_SITES)
    out["stage"] = out["Name_EN"].map(_stage_of)
    out["stage"] = pd.Categorical(out["stage"], categories=STAGE_ORDER, ordered=True)

    # 5. calendar helpers
    out["month"] = out["Date"].dt.month
    out["season"] = pd.Categorical(out["month"].map(SEASON_BY_MONTH), categories=SEASON_ORDER, ordered=True)

    # 6. row-level completeness
    out["n_measured"] = out[NUMERIC].notna().sum(axis=1)
    return out


def load_clean(path: str | Path = RAW_PATH, **kw) -> pd.DataFrame:
    return clean(load_raw(path), **kw)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = load_clean()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(df.shape, "->", PROCESSED_PATH)
