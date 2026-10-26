# Changes log (data handling decisions)

| # | Decision | Rows affected | Where | Status |
|---|---|---|---|---|
| 1 | Korean non-detect string flagged `nondetect`; NH3-N set to LOD/2 = 0.0025 mg/L (LOD 0.005 mg/L, US EPA QA/G-9S), others set to missing | Turbidity 1, SS 1, Br 2, NH3-N 25 | `preprocessing/load.py` | LOD known for NH3-N only (2026-10-09) |
| 2 | 0 in SUVA, aromaticity and molecular weight simultaneously treated as placeholder -> missing | 4 samples (T1) | `preprocessing/load.py` | query |
| 3 | Samples with <= 2 measured variables excluded from modelling | 3 samples (T1, Dec 2017 - Feb 2018) | `src/features.py` | recorded as `empty_record` |
| 4 | Literal zeros become missing when log10-transformed (previously undocumented) | Br 5, NH3-N 2 | `src/features.py::_log10_pos` | documented 2026-09-26; query sent |
| 5 | Site codes 12-20 inferred from sample names (blank in workbook) | 259 records | `preprocessing/load.py` | query |
| 6 | Identical consecutive-month values kept, flagged `repeated_value` | see query_list.csv | `src/rev_audit.py` | query |
| 7 | M2 THMFP 0.007 mg/L (Sep 2017) kept in all targets and evaluations; only screened as a *sentinel input* | 1 | `src/sentinel.py` | not labelled erroneous; query |
