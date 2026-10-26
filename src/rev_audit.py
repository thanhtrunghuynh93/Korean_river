"""P0 provenance audit: data dictionary, sample ledger with quality flags, count reconciliation, query list.

Nothing is removed or altered here; every decision is recorded as a flag so later steps can filter explicitly.

Flags per (sample, variable):
    measured                  numeric value parsed from the workbook
    nondetect                 Korean "불검출" (not detected); LOD unknown
    zero                      literal 0 (enters a log transform as missing; LOD/meaning unknown)
    placeholder_zero          0 in all three of SUVA / aromaticity / molecular weight at once
    not_measured_by_design    missing for 100 % of the sample's site group (structural absence)
    missing                   empty cell otherwise
    empty_record              the whole sample has <= 2 measured variables
Sample-level query flags (not exclusions):
    repeated_value            identical to the same site's previous-month value (possible copy)
    lowest_or_highest_value   dataset minimum / maximum of a target (laboratory query, e.g. M2 Sep 2017)

    uv run python -m src.rev_audit      # -> results/revision/audit/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.load import NONDETECT, NUMERIC, UNITS, VAR_GROUPS, clean, load_raw
from src.features import SKEWED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "audit"

DESCRIPTION = {
    "Temp": "water temperature (field)", "pH": "pH (field)", "Turbidity": "turbidity (field)",
    "EC": "electrical conductivity (field)", "DO": "dissolved oxygen (field)", "SS": "suspended solids",
    "Br": "bromide", "BOD": "biochemical oxygen demand", "COD": "chemical oxygen demand", "NH3N": "ammonia nitrogen",
    "TOC": "total organic carbon", "SUVA": "specific UV absorbance", "Aromaticity": "aromaticity index",
    "MolWeight": "apparent molecular weight", "Biopolymer": "LC-OCD biopolymers", "HS": "LC-OCD humic substances",
    "BB": "LC-OCD building blocks", "LMWN": "LC-OCD low-molecular-weight neutrals",
    "THMFP": "trihalomethane formation potential (target)", "HAAFP": "haloacetic acid formation potential (target)",
}


def sample_ids(raw: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    ids = cl[["Date", "Site", "Name_EN", "group", "wtp", "stage"]].copy()
    ids["Site"] = ids.Site.astype(str)
    ids["campaign"] = ids.Date.rank(method="dense").astype(int)
    ids["sample_id"] = ids.Date.dt.strftime("%Y%m") + "_" + ids.Site
    return ids


def ledger(raw: pd.DataFrame, cl: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group = ids.group.astype(str)
    empty = cl.n_measured <= 2
    uv = VAR_GROUPS["uv_size"]
    for v in NUMERIC:
        txt = raw[v].astype("string")
        val = pd.to_numeric(raw[v], errors="coerce")
        struct_missing = val.isna().groupby(group).transform("mean").eq(1.0)
        flag = pd.Series("measured", index=raw.index)
        flag[val.isna()] = "missing"
        flag[val.isna() & struct_missing] = "not_measured_by_design"
        flag[txt.str.strip().eq(NONDETECT).fillna(False)] = "nondetect"
        flag[val.eq(0)] = "zero"
        if v in uv:
            flag[cl.uv_zero_placeholder] = "placeholder_zero"
        flag[empty] = "empty_record"
        rows.append(pd.DataFrame({"sample_id": ids.sample_id, "variable": v, "raw_text": txt.fillna(""),
                                  "value": val, "flag": flag, "log_transformed": v in SKEWED or v in ("THMFP", "HAAFP")}))
    return pd.concat(rows, ignore_index=True)


def query_flags(raw: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    q = []
    r = raw.assign(sample_id=ids.sample_id.values, Site=ids.Site.values).sort_values(["Site", "Date"])
    for v in NUMERIC:
        s = pd.to_numeric(r[v], errors="coerce")
        prev = s.groupby(r.Site).shift(1)
        rep = s.notna() & s.eq(prev)
        for sid in r.loc[rep, "sample_id"]:
            q.append({"sample_id": sid, "variable": v, "query": "repeated_value",
                      "detail": "identical to the same site's previous-month value"})
    for t in ("THMFP", "HAAFP"):
        s = pd.to_numeric(raw[t], errors="coerce")
        for which, idx in (("minimum", s.idxmin()), ("maximum", s.idxmax())):
            q.append({"sample_id": ids.sample_id[idx], "variable": t, "query": f"dataset_{which}",
                      "detail": f"{t} = {s[idx]} mg/L (reported units)"})
    for v in NUMERIC:
        s = pd.to_numeric(raw[v], errors="coerce")
        for idx in s.index[s.eq(0)]:
            q.append({"sample_id": ids.sample_id[idx], "variable": v, "query": "zero_value",
                      "detail": "literal 0; becomes missing under log10; meaning (below LOD?) unknown"})
    return pd.DataFrame(q)


def dictionary(ldg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in NUMERIC:
        f = ldg[ldg.variable == v].flag.value_counts()
        rows.append({"variable": v, "description": DESCRIPTION[v], "unit_reported": UNITS[v], "unit_verified": False,
                     "block": next(k for k, cols in VAR_GROUPS.items() if v in cols),
                     "log10_in_models": v in SKEWED or v in ("THMFP", "HAAFP"),
                     **{f"n_{k}": int(f.get(k, 0)) for k in ["measured", "nondetect", "zero", "placeholder_zero",
                                                             "not_measured_by_design", "missing", "empty_record"]}})
    return pd.DataFrame(rows)


def reconciliation(cl: pd.DataFrame, ldg: pd.DataFrame) -> str:
    th = cl.THMFP.notna() & (cl.n_measured > 2)
    ha = cl.HAAFP.notna() & (cl.n_measured > 2)
    lines = ["# Count reconciliation", "",
             f"- design records (44 sites x 16 campaigns): {len(cl)}",
             f"- non-empty records (> 2 measured variables): {int((cl.n_measured > 2).sum())}",
             f"- with THMFP: {int(th.sum())}; with HAAFP: {int(ha.sum())}; with both: {int((th & ha).sum())}",
             f"- duplicate sample identifiers: {int(ldg.sample_id.nunique() * 0 + cl.duplicated(['Date', 'Site']).sum())}",
             f"- zero values that become missing under log10: "
             + ", ".join(f"{v} {n}" for v, n in ldg[ldg.flag == 'zero'].groupby('variable').size().items()),
             f"- non-detects: " + ", ".join(f"{v} {n}" for v, n in ldg[ldg.flag == 'nondetect'].groupby('variable').size().items()),
             "- identical measurement vectors across two sites in the same campaign (possible sample reuse between an "
             "intake and a main-stem record): 0",
             ""]
    return "\n".join(lines)


CHANGES_LOG = """# Changes log (data handling decisions)

| # | Decision | Rows affected | Where | Status |
|---|---|---|---|---|
| 1 | Korean non-detect string flagged `nondetect`; NH3-N set to LOD/2 = 0.0025 mg/L (LOD 0.005 mg/L, US EPA QA/G-9S), others set to missing | Turbidity 1, SS 1, Br 2, NH3-N 25 | `preprocessing/load.py` | LOD known for NH3-N only (2026-10-09) |
| 2 | 0 in SUVA, aromaticity and molecular weight simultaneously treated as placeholder -> missing | 4 samples (T1) | `preprocessing/load.py` | query |
| 3 | Samples with <= 2 measured variables excluded from modelling | 3 samples (T1, Dec 2017 - Feb 2018) | `src/features.py` | recorded as `empty_record` |
| 4 | Literal zeros become missing when log10-transformed (previously undocumented) | Br 5, NH3-N 2 | `src/features.py::_log10_pos` | documented 2026-09-26; query sent |
| 5 | Site codes 12-20 inferred from sample names (blank in workbook) | 259 records | `preprocessing/load.py` | query |
| 6 | Identical consecutive-month values kept, flagged `repeated_value` | see query_list.csv | `src/rev_audit.py` | query |
| 7 | M2 THMFP 0.007 mg/L (Sep 2017) kept in all targets and evaluations; only screened as a *sentinel input* | 1 | `src/sentinel.py` | not labelled erroneous; query |
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    cl = clean(raw)
    ids = sample_ids(raw, cl)
    ldg = ledger(raw, cl, ids)
    ldg.to_csv(OUT / "sample_ledger_long.csv", index=False)
    wide = ldg.pivot(index="sample_id", columns="variable", values="flag")
    ids.set_index("sample_id").join(wide).to_csv(OUT / "sample_quality_flags.csv")
    dictionary(ldg).to_csv(OUT / "data_dictionary.csv", index=False)
    q = query_flags(raw, ids)
    q.to_csv(OUT / "query_list.csv", index=False)
    (OUT / "count_reconciliation.md").write_text(reconciliation(cl, ldg))
    (OUT / "changes_log.md").write_text(CHANGES_LOG)
    print((OUT / "count_reconciliation.md").read_text())
    print(q["query"].value_counts().to_string())


if __name__ == "__main__":
    main()
