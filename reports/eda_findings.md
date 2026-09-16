# EDA findings — Nakdong River DBP formation potential (THMFP / HAAFP)

Source: `data/20260909 DBP Viet.xlsx`, sheet `Total`. Notebook: `notebooks/eda.ipynb`. Figures: `figures/`.
Cleaning rules: `preprocessing/load.py`. Written 2026-09-16.

## 1. What the data is

- **704 rows = 44 sampling points × 16 monthly dates** (May 2017 – Aug 2018). The panel is complete: every
  point has exactly one row per month, no duplicates.
- **18 predictors + 2 targets.** Field (Temp, pH, Turbidity, EC, DO, SS), lab (Br, BOD, COD, NH3-N, TOC),
  UV/size (SUVA, Aromaticity, Mol Weight), LC-OCD fractions (Biopolymer, HS, BB, LMWN); outputs THMFP, HAAFP.
- **Four site groups** that behave differently and should be modelled with a group indicator or separately:

| Group | Codes | Sites | Rows | Median THMFP | Median HAAFP |
|---|---|---|---|---|---|
| Main stem (Andong → estuary) | M1–M18 | 18 | 288 | 0.108 | 0.122 |
| Tributaries | T1–T6 | 6 | 96 | 0.114 | 0.130 |
| Reservoirs | 1–5 | 5 | 80 | 0.058 | 0.062 |
| WTP treatment trains (Gumi, Goryeong, Bansong) | 6–20 | 15 | 240 | 0.052 | 0.049 |

Units are not given in the workbook; the ones used in the notebook are assumptions to confirm with the data owner.

## 2. Data quality and how it was handled

| Issue | Extent | Handling in `load.py` |
|---|---|---|
| Korean non-detect string `불검출` in numeric columns | Turbidity 1, SS 1, Br 2, NH3-N 25 | Set to NaN, boolean `<var>_nd` flag kept. `nondetect="zero"` option exists; ½·LOD needs the LOD values. |
| Site code blank for Goryeong/Bansong train rows | 259 rows | Filled from the English name (codes 12–20 are consistent with the Gumi numbering 6–11). |
| SUVA / Aromaticity / Mol Weight literal 0 in all three | 4 rows (Wicheon T1, Dec 2017 – Mar 2018) | Treated as placeholders → NaN, flagged `uv_zero_placeholder`. |
| Rows with no measurements at all | 3 rows (Wicheon T1, Dec 2017 – Feb 2018) | Kept, `n_measured` column allows dropping. |
| Missing values | see below | Not imputed; complete-case counts reported in §6. |

**Missingness is structural, not random** (`figures/02_missingness.png`):

- SUVA / Aromaticity / Mol Weight: **100 % missing for reservoirs and every treatment-train sample**, 25–29 %
  missing for main-stem and tributary rows. Only 284 of 704 rows have them.
- DO, SS, BOD, COD, NH3-N: **60–95 % missing for treatment-train rows**, ≤ 6 % for river rows (NH3-N 32 % for
  reservoirs).
- Everything else (Temp, pH, Turbidity, EC, Br, TOC, the four LC-OCD fractions, both targets) is ≥ 98 % complete.

**Outliers** (robust z > 3.5 on log scale, `figures/09–10`, outlier table in notebook §4): one sampling event
dominates. **Hoecheon T4, Aug 2018** has Biopolymer 6102 µg/L (median 164), BB 5329, COD 60.9 and BOD 13.6,
all record values. Geumho River T3 has EC 1541–1811 µS/cm (Dec 2017, Apr 2018). The largest target outliers are in
the treatment trains: **Gumi raw water Jul 2018 HAAFP 0.389** and Gumi/Bansong settled water Aug 2018 (0.24–0.31)
against a train median around 0.05. Summer 2018 also shows a jump in the main-stem monthly median (THMFP 0.32 in
Aug 2018, HAAFP 0.32 in Jul 2018, versus 0.08–0.21 in other months). These should be confirmed as real events
(2018 heat wave / bloom) rather than analytical batch effects before modelling.

## 3. Target behaviour

- Both targets are right-skewed (skew ≈ 2) and roughly log-normal; **model log10(THMFP), log10(HAAFP)**
  (`figures/03_target_hist.png`). Range 0.007–0.551 mg/L, medians 0.081 / 0.087.
- THMFP and HAAFP are correlated (Spearman 0.67 overall; 0.59–0.69 within river groups, 0.45 within trains) but
  not interchangeable — a joint or multi-output model is reasonable, separate models are also justified.
- **Spatial signal:** river and tributary sites are ~2× reservoirs and treatment trains. Along the main stem,
  targets rise from Andong (M1) downstream and peak around M9–M11 (Goryeong reach) in summer
  (`figures/18_site_profile.png`). Site alone explains η² = 0.33 (THMFP) / 0.59 (HAAFP) of log-target variance
  on all rows, 0.24 / 0.37 on river rows.
- **Seasonal signal:** summer maximum for river rows, winter minimum; Date explains η² = 0.38 (THMFP) / 0.15
  (HAAFP) on all rows, 0.47 / 0.29 on river rows (`figures/06`, `figures/17`). Sampling month or Temp must be in
  the feature set; a random train/test split across months will leak season.
- **Treatment trains** (`figures/08_treatment_train.png`): coagulation/settling removes about half of both
  formation potentials (median raw ≈ 0.09–0.13 → settled ≈ 0.05–0.06); ozonation slightly raises them at all
  three plants; GAC / F/A brings them to ≈ 0.03–0.05 in finished water. TOC falls from ≈ 2.6 to ≈ 1.7–2.0.

## 4. Strongest predictors (Spearman ρ, `figures/12_corr_ranked.png`)

| Predictor | THMFP all / river / train | HAAFP all / river / train | n (all) |
|---|---|---|---|
| HS (humic substances) | 0.66 / 0.62 / 0.46 | **0.78** / 0.73 / 0.52 | 693 |
| TOC | 0.66 / 0.60 / **0.64** | 0.68 / 0.61 / **0.68** | 691 |
| Turbidity | 0.56 / 0.45 / 0.38 | 0.65 / 0.38 / 0.51 | 692 |
| BB (building blocks) | 0.55 / 0.53 / 0.49 | 0.59 / 0.61 / 0.48 | 693 |
| COD | 0.53 / 0.52 / (n=16) | 0.51 / 0.51 / (n=16) | 469 |
| SS | 0.53 / 0.53 / (n=16) | 0.47 / 0.47 / (n=16) | 468 |
| Biopolymer | 0.46 / 0.30 / 0.19 | 0.49 / 0.17 / 0.29 | 693 |
| SUVA | 0.41 / 0.41 / — | 0.53 / 0.53 / — | 280 |
| Temp | 0.37 / 0.50 / 0.34 | 0.23 / 0.34 / 0.09 | 693 |

- **Organic-carbon quantity dominates:** TOC and HS carry the same information (ρ = 0.78 between them); HS is
  the better single predictor of HAAFP, TOC the more robust one across groups.
- **Br is weak on its own** (ρ ≈ 0.2 overall, ≈ 0.1 for river rows) and adds little after TOC is accounted for
  (Spearman of log-log TOC residuals with log Br: +0.15 THMFP, +0.09 HAAFP). It may still matter for speciation
  (brominated species), which these totals cannot show.
- **EC, LMWN, NH3-N, BOD, DO, Mol Weight** are near zero or unstable in sign across groups; low priority.
- Turbidity/SS/COD correlate with targets mostly through particulate and organic load; Turbidity is the only one
  of the three that is complete for treatment-train rows.
- Correlation with the targets is weaker inside the treatment trains than in river water for every predictor
  except TOC and COD — treated water is a different regime.

## 5. Collinearity (`figures/16_predictor_corr.png`, VIF table in notebook §6)

- |ρ| > 0.7 pairs (river rows): HS–BB 0.81, TOC–HS 0.78, EC–Br 0.74, TOC–BB 0.72.
- Aromaticity, Mol Weight and SUVA form a block that is negatively related to EC / Br / DO / NH3-N (ρ ≈ −0.5),
  i.e. they separate reservoir/upstream water from downstream water rather than adding DBP information.
- VIF with all 18 predictors on river complete cases: HS 8.4, BB 8.5, SUVA 6.5, TOC 5.6, Mol Weight 5.6. Dropping
  the UV/size trio brings every VIF below 5. Tree models will not care; linear models should use TOC **or**
  HS+BB, not all three.

## 6. Recommended feature sets and row counts

| Set | Predictors | Rows with both targets | main stem | tributary | reservoir | train |
|---|---|---|---|---|---|---|
| A: all 18 | everything | **277 / 704 (39 %)** | 211 | 66 | **0** | **0** |
| B: drop SUVA, Aromaticity, Mol Weight | 15 | 439 (62 %) | 283 | 91 | 52 | 13 |
| C: core 10 = Temp, pH, Turbidity, EC, Br, TOC, Biopolymer, HS, BB, LMWN | 10 | **685 (97 %)** | 283 | 91 | 74 | 237 |

- **Set C is the only set that covers all four groups.** It keeps the two strongest predictors (TOC, HS) plus
  Turbidity, Temp and the remaining LC-OCD fractions, loses nothing with |ρ| > 0.55 except COD/SS, which are
  absent for treatment-train rows anyway.
- Set B is the river-only option if COD/SS/DO/BOD/NH3-N are wanted (main stem + tributary ≈ 374 rows).
- Set A (with SUVA/Aromaticity/Mol Weight) is a **main-stem/tributary-only** model on ~277 rows; use it only if
  those UV descriptors are the scientific question.
- Add `group` (or `Site`) and month/season as features; use grouped or time-blocked cross-validation (by Date,
  or leave-site-out) because Site and Date each explain more variance than any single predictor.

## 7. Open questions for the data owner

1. Units for every variable (assumed mg/L for THMFP/HAAFP, µg/L C for LC-OCD fractions, µS/cm for EC).
2. Detection limits for the `불검출` values (needed for ½·LOD substitution), especially NH3-N (25 values).
3. Are the four all-zero SUVA/Aromaticity/Mol Weight rows and the three fully blank Wicheon (T1) rows for
   Dec 2017 – Mar 2018 a sampling gap (frozen / dry tributary)?
4. Is the Aug 2018 Hoecheon (T4) event (Biopolymer 6102, COD 60.9) real? Are the Jul–Aug 2018 treatment-train
   target spikes (Gumi raw HAAFP 0.389) real?
5. Confirm the inferred site codes 12–20 for the Goryeong and Bansong trains, and that M9 / M14 are the raw-water
   intakes of Goryeong and Bansong (they are labelled that way in the workbook).
6. Should the treatment-train rows be part of the same predictive model as river water, or is the model for
   source water only? Correlation structure differs between the two regimes.
