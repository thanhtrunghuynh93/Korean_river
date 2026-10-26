"""Non-detect handling in preprocessing/load.py."""
import numpy as np
import pandas as pd
import pytest

from preprocessing.load import COLUMNS, DETECTION_LIMITS, NONDETECT, clean


def _raw() -> pd.DataFrame:
    row = {c: 1.0 for c in COLUMNS}
    row.update(Date=pd.Timestamp("2017-05-01"), Site="M1", Sample_KR="", Name_EN="x", Position="")
    df = pd.DataFrame([row, row, row])
    df["Site"] = df["Site"].astype("string")
    df["NH3N"] = df["NH3N"].astype(object)
    df["Br"] = df["Br"].astype(object)
    df.loc[0, "NH3N"] = NONDETECT
    df.loc[1, "Br"] = NONDETECT
    return df


def test_half_lod_substitutes_known_lod_only():
    out = clean(_raw())
    assert out.loc[0, "NH3N"] == pytest.approx(DETECTION_LIMITS["NH3N"] / 2) == pytest.approx(0.0025)
    assert bool(out.loc[0, "NH3N_nd"])
    assert np.isnan(out.loc[1, "Br"])          # no LOD known -> stays missing
    assert out.loc[2, "NH3N"] == 1.0


def test_nan_mode_keeps_previous_behaviour():
    out = clean(_raw(), nondetect="nan")
    assert np.isnan(out.loc[0, "NH3N"])
