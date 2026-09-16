# THMFP and HAAFP predictors — results and recommendation

Code: `src/` (features, cv, models, train, interpret). Results: `results/experiments.csv` (96 cells),
`results/predictions.csv` (out-of-fold predictions), `results/final_models.csv`, `results/importance.csv`.
Figures: `figures/20_…27_….png`, rendered by `notebooks/modeling.ipynb`. Written 2026-09-16.

## 1. Setup

- **Targets:** log10(THMFP), log10(HAAFP) in mg/L; 693 / 692 rows across all 44 sites (main stem, tributaries,
  reservoirs, three WTP treatment trains).
- **Three feature tiers**, nested:
  - **T1 chemistry** — the core-10 water-quality variables (Temp, pH, Turbidity, EC, Br, TOC, Biopolymer, HS, BB,
    LMWN; log10 where skewed). Needs one water sample.
  - **T2 + context** — site group, treatment stage, WTP, main-stem position, month (sin/cos) and a site
    target-encoding fitted inside every training fold. Needs where/when.
  - **T3 + network** — previous-month same-site target, TOC and HS; the site's mean of earlier targets; the
    upstream site's same-month target and TOC (M(k−1) on the main stem, previous stage in a treatment train).
    Needs monitoring history.
- **Models:** ridge (median-impute → scale → RidgeCV), sklearn HistGradientBoosting, XGBoost. Baselines: global
  median, training-fold site median, previous-month persistence.
- **Validation:** four schemes, all nested (hyper-parameters tuned by a 12-draw random search whose inner folds
  are grouped like the outer scheme).

| Scheme | Holds out | Answers |
|---|---|---|
| random | random rows, 5×2 | optimistic reference |
| month | 2 whole months × 8 folds | can we predict a new month at known sites? |
| site | 4 whole sites × 11 folds | can we predict a site we never sampled? |
| forward | May–Aug 2018 after training on the first 12 months | strict temporal extrapolation |

Leakage safeguards, all tested in `tests/`: lag features use only earlier dates; upstream features use other
sites at the same date; the site encoding is fit on the training index only; folds never share a Date (month
scheme) or a Site (site scheme); hyper-parameter search is grouped like the outer fold.

## 2. Headline numbers — pooled out-of-fold R² on the log10 target

| Target | Tier / model | random | month | site | forward | medAPE (site) |
|---|---|---|---|---|---|---|
| THMFP | T1 xgb | 0.60 | 0.40 | 0.58 | −0.20 | 18 % |
| THMFP | T1 ridge | 0.55 | **0.48** | 0.55 | −0.03 | |
| THMFP | T3 hgb | 0.71 | 0.52 | **0.71** | −0.10 | 15 % |
| THMFP | **T3 xgb (final)** | 0.72 | 0.53 | 0.70 | −0.17 | 15 % |
| THMFP | T3 ridge | 0.65 | **0.58** | 0.64 | **0.22** | |
| HAAFP | **T1 xgb** | 0.72 | 0.56 | **0.72** | 0.46 | 20 % |
| HAAFP | T2 xgb | 0.75 | 0.54 | 0.61 | 0.44 | 24 % |
| HAAFP | **T3 xgb (final)** | 0.77 | 0.61 | 0.68 | 0.52 | 20 % |
| HAAFP | T3 ridge | 0.71 | **0.65** | 0.48 | **0.53** | |
| baseline | persistence (T3 info only) | 0.33 / 0.33 | 0.33 / 0.33 | 0.33 / 0.30 | 0.21 / 0.06 | THMFP / HAAFP |
| baseline | site median | 0.20 / 0.52 | 0.18 / 0.52 | −0.01 / −0.02 | −0.85 / 0.41 | THMFP / HAAFP |

Full table: notebook §1 or `results/experiments.csv`. Fold-to-fold spread is large under the month scheme
(sd of R² 0.2–0.3), so differences below ≈0.05 between models are not meaningful.

## 3. What we learned

**Selected models** (rule: best mean R² over the two honest schemes, month and site) are **T3 XGBoost for both
targets**: THMFP R² 0.53 (month) / 0.70 (site), HAAFP 0.61 / 0.68. Median absolute error on the original scale
is 15–20 %. Saved to `models/THMFP_tier3_xgb.joblib` and `models/HAAFP_tier3_xgb.joblib`.

**Spatial/temporal context helps, but differently for the two targets** (`figures/21_tier_gain.png`):

- *THMFP* gains +0.10 (month), +0.14 (site) and +0.25 (forward) R² from the network tier. Permutation
  importance ranks the previous-month same-site THMFP first, then the month-of-year encodings and Temp, ahead
  of TOC and HS. THMFP behaves like a seasonal, site-persistent quantity that chemistry alone under-explains.
- *HAAFP* is chemistry-driven: HS (humic substances) is by far the most important feature, and chemistry alone
  already reaches R² 0.72 on held-out sites. The network tier adds +0.08 on new months and +0.07 on the forward
  holdout but **loses 0.04 on new sites**, and the T2 site encoding loses 0.10 there: for a site never sampled,
  the encoder falls back to the global mean, which is worse than letting the trees use HS directly.
  → For predicting HAAFP at a **new site**, use the T1 XGBoost (`--tier 1 --model xgb`); for new months at
  monitored sites, use T3.
- The T2 encodings alone (group, stage, position, month, site encoding) add almost nothing (≤ 0.03) on top of
  chemistry under honest validation. Site identity is largely already carried by the chemistry.

**Temporal extrapolation is the weak spot** (`figures/20`, `figures/27`). Training on May 2017–Apr 2018 and
predicting May–Aug 2018 gives negative R² for THMFP with tree models and only 0.22 with T3 ridge. Those four
months contain the record THMFP values seen in the EDA (main-stem monthly median 0.32 in Aug 2018 vs ≤ 0.21 in
every other month); trees cannot extrapolate beyond the training range and the site-median baseline collapses
to −0.85. HAAFP, whose summer 2018 was less extreme, extrapolates at R² 0.46–0.53. Any deployment must expect
degradation in unusually hot / high-organic summers, and should retrain as new months arrive.

**Linear vs. trees.** Ridge is competitive or better whenever the test distribution differs from training
(month and forward schemes), trees win on random and site schemes. A ridge–XGBoost average would be a
reasonable robust choice; not built here.

**Per group** (`figures/23`): treatment-train rows are predicted about as well as river rows on new sites
(R² ≈ 0.6 for THMFP); reservoirs are hardest (THMFP 0.14 with chemistry, 0.54 with network). HAAFP on the main
stem is poorly predicted across months (R² 0.08 with chemistry, 0.34 with network) because within-site
variation there is mostly seasonal and small relative to between-site differences.

**Partial dependence** (`figures/25`) is monotone and physically plausible: predicted HAAFP rises steadily with
HS and TOC (≈0.072 → 0.12 mg/L over the observed HS range) and with the site's past and upstream HAAFP; predicted
THMFP rises with last month's THMFP and with temperature above ≈10 °C, and the month encodings place the maximum
in late spring / summer.

**Residuals** (`figures/26`) show no site with a systematic bias except wider spread at Hwang River (T5),
Yeongcheon Reservoir (4) and Miryang Reservoir (1); by month, THMFP is over-predicted in Feb–Mar 2018 and
strongly under-predicted in May–Jun 2018, consistent with the seasonal anomaly.

## 4. Recommendation

| Use case | Model | Expected R² (log10) | medAPE |
|---|---|---|---|
| New month at a monitored site, both targets | `models/*_tier3_xgb.joblib` | THMFP 0.53, HAAFP 0.61 | 15–20 % |
| New site, THMFP | T3 xgb/hgb (needs upstream + previous month) | 0.70 | 15 % |
| New site, HAAFP | T1 xgb (chemistry only) | 0.72 | 20 % |
| Single sample, no history, either target | T1 xgb | THMFP 0.58, HAAFP 0.72 (site) | 18–20 % |

## 5. Limitations and next steps

- 16 months = one seasonal cycle; the forward test is a single 4-month period with an anomaly. More years are
  the only cure for the extrapolation weakness.
- Hyper-parameter search is small (12 draws); results are stable enough for model ranking, not for squeezing
  the last 0.02 of R².
- Not tried: ridge + XGBoost blend; multi-output model sharing information between THMFP and HAAFP
  (Spearman 0.67); quantile / interval predictions; explicit bromide-incorporation terms (Br added little here,
  but speciation was not the target).
- Data owner questions from the EDA still apply (units, non-detect limits, the Aug 2018 Hoecheon event).
