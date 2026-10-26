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

## 5. Improvement round 1 — richer features (2026-09-16)

Scope chosen by the user: feature changes only (basin-wide same-month features; richer chemistry and
cross-target lags). Models, nested search and folds are unchanged, so gains are attributable to features.
Five pre-registered feature sets on top of tier 3 (`src/experiments.py`), evaluated with ridge and XGBoost under
month / site / forward; site folds repeated over 3 seeds (seed sd ≤ 0.024). Selection rule fixed in advance:
best mean R² over month and site. Full table: notebook §10, `results/experiments.csv` (rows `t3…t4_all`).

| Experiment | Adds | THMFP ridge month / site / fwd | THMFP xgb | HAAFP ridge | HAAFP xgb |
|---|---|---|---|---|---|
| `t3` (baseline) | — | 0.62 / 0.68 / 0.32 | 0.53 / 0.70 / −0.17 | 0.65 / 0.54 / 0.51 | 0.61 / 0.65 / 0.52 |
| `t3_chem` | LC-OCD ratios to TOC, Temp×TOC, sparse extras | 0.63 / 0.67 / 0.23 | 0.55 / 0.72 / −0.03 | 0.61 / 0.63 / 0.50 | 0.63 / 0.69 / 0.53 |
| `t3_xlag` | other target's lags, lag-2, upstream lag-1 | 0.59 / 0.68 / 0.37 | 0.54 / 0.73 / 0.01 | 0.65 / 0.59 / 0.51 | 0.62 / 0.63 / 0.50 |
| `t3_basin` | same-month network means, downstream, plant raw | **0.68 / 0.68 / 0.42** | 0.60 / 0.72 / −0.38 | 0.72 / 0.62 / 0.62 | 0.63 / 0.67 / 0.55 |
| `t4_all` | union | 0.66 / 0.65 / 0.43 | 0.60 / 0.75 / −0.16 | **0.71 / 0.66 / 0.59** | 0.66 / 0.68 / 0.56 |

**Result.** Both final models change to **ridge** with network features:

| Target | Round 0 final | Round 1 final | month | site | forward | honest mean |
|---|---|---|---|---|---|---|
| THMFP | T3 xgb: 0.53 / 0.70 / −0.17 | **`t3_basin` ridge** | 0.68 | 0.68 | 0.42 | 0.615 → **0.682 (+0.07)** |
| HAAFP | T3 xgb: 0.61 / 0.68 / 0.52 | **`t4_all` ridge** | 0.71 | 0.66 | 0.59 | 0.646 → **0.686 (+0.04)** |

THMFP meets the ≥ 0.05 success criterion; HAAFP falls just short of it but the gain is above the seed noise
(0.02) and consistent across all three schemes. Saved as `models/THMFP_t3_basin_ridge.joblib` and
`models/HAAFP_t4_all_ridge.joblib` (the round-0 XGBoost models remain in `models/`).

**Why it works.** The basin features remove the shared monthly offset: the share of month-scheme residual
variance explained by a per-month mean residual falls from 26 % to 11 % (THMFP, ridge) and 22 % to 8 % (HAAFP)
(notebook §10). Permutation importance of the finals (`figures/29`) is led by the group / basin same-month means,
downstream and upstream targets, then HS and lag-1. Forward extrapolation into summer 2018 improves for ridge
(THMFP 0.32 → 0.42, HAAFP 0.51 → 0.59) because the basin mean of that month carries the anomaly into the inputs;
trees do the opposite for THMFP (xgb −0.17 → −0.38) because they cannot extrapolate a feature beyond its
training range.

**Negative and mixed results, kept on record.**
- `t3_xlag` (cross-target lags, lag-2) is noise-level everywhere (|Δ| ≤ 0.03 on honest schemes).
- `t3_chem` helps HAAFP on new sites (ridge +0.09, xgb +0.03) but hurts THMFP forward for ridge (−0.10); it is in
  the HAAFP final via `t4_all` and excluded from the THMFP final.
- XGBoost improves on the site scheme with every block (THMFP `t4_all` 0.75, the best single site-scheme
  number) but never on the month scheme by more than 0.07, so the honest mean favours ridge.
- HGB on the winners tracked xgb (THMFP `t3_basin` 0.65 / 0.71; HAAFP `t4_all` 0.65 / 0.61) and did not win.
- One confound to disclose: the ridge pipeline gained a missing-value indicator in this round so the sparse extras
  would not be imputed silently. That alone raised the ridge `t3` baseline versus round 0 (THMFP site 0.64 → 0.68,
  HAAFP site 0.48 → 0.54). All round-1 comparisons are against the re-run `t3`, so the feature deltas above are
  like-for-like; the round-0 → round-1 totals include this model-side effect.

**Deployment assumption.** Basin, group, downstream and plant-raw features need the *other* sites of the network
sampled in the same month (the same assumption as the upstream feature already in tier 3). For a single sample
without network context, the tier-1 chemistry models from round 0 still apply (THMFP 0.58, HAAFP 0.72 on new sites).

**Updated recommendation**

| Use case | Model | month / site / forward R² |
|---|---|---|
| Monitoring network sampled this month, either target | `models/THMFP_t3_basin_ridge.joblib`, `models/HAAFP_t4_all_ridge.joblib` | 0.68 / 0.68 / 0.42 and 0.71 / 0.66 / 0.59 |
| New site inside the network, THMFP | `t4_all` xgb (site 0.75) or the ridge final (0.68) | |
| New site, HAAFP, chemistry only | round-0 T1 xgb (site 0.72) | |
| Single sample, no network or history | round-0 T1 xgb | THMFP 0.58, HAAFP 0.72 (site) |

## 6. Sentinel sampling design (2026-09-25)

**Why.** Round 1's gain came almost entirely from other sites' same-month DBP, and THMFP shares 57 % of its
within-site month-to-month variation across the whole basin (HAAFP 36 %). The sentinel design uses that shared
signal while cutting lab work. Chemistry is measured everywhere each month, but THMFP / HAAFP are lab-tested
only at *k* sentinel sites.

**Setup** (`src/sentinel.py`, tests in `tests/test_sentinel.py`).

- The 16-month dataset acts as a calibration campaign that gives each site's normal log level.
- A non-sentinel site is predicted by ridge regression from:
  - its chemistry and context (tier 2 without the site code);
  - its calibration mean;
  - the sentinels' mean deviation from their own normal level that month, for both targets;
  - the deviation of the nearest main-stem sentinel.
- Non-sentinel sites have no DBP history after calibration, so own-lag features are not used.
- Evaluation is leave-2-months-out (8 folds).
- Sentinels are chosen by greedy forward selection inside each outer fold, scored on an inner
  leave-months-out over the training months only.
- Scores are pooled R² on the non-sentinel rows of the held-out months.

**QA screen, added after the first run.** In the first run THMFP collapsed at two and three sentinels (R² 0.31
and 0.37). The cause was one reading: Imha Reservoir (M2), September 2017, THMFP 0.007 mg/L. That is the
dataset minimum and 1.2 log units below the site's normal level. Whenever M2 was a sentinel, that reading
dragged every site's prediction down by about a factor of three for the month.

A standard lab-QA rule was added. A sentinel reading is dropped if it lies more than 4 robust SDs from that
site's normal level, where the robust SD is 1.4826 × MAD of its training-month deviations, floored at 0.05
log10. Thresholds use training months only. They were set once from the rule and not tuned on scores. Both runs
are kept in `results/sentinel_summary.csv` (column `qa`). The rule was introduced after seeing test-fold
results, so treat the screened numbers as slightly optimistic.

| Sentinels / month | Lab share | THMFP R², QA | THMFP R², no QA | HAAFP R², QA | HAAFP R², no QA |
|---|---|---|---|---|---|
| 0 | 0 % | 0.49 | 0.49 | 0.58 | 0.58 |
| 1 | 2 % | 0.63 | 0.56 | 0.68 | 0.67 |
| 2 | 5 % | 0.58 | 0.31 | 0.69 | 0.71 |
| 4 | 9 % | 0.65 | 0.53 | 0.68 | 0.71 |
| 8 | 18 % | 0.68 | 0.67 | 0.72 | 0.70 |
| all (round-1 gap-fill finals) | ~100 % | 0.68 | | 0.71 | |

Fold-to-fold sd is 0.14–0.21 throughout (`figures/30_sentinel_design.png`).

**Findings.**

- **A single sentinel recovers much of the network benefit.** THMFP rises from 0.49 to 0.63, HAAFP from 0.58 to
  0.68, with 2 % of the lab tests.
- **About 4–8 sentinels (9–18 % of lab tests) match full-network gap-filling for THMFP.** For HAAFP, 1–2
  sentinels are enough. Differences between k = 1 and k = 8 are within fold noise for HAAFP.
- **Useful sentinels differ by target.** For HAAFP, Bansong raw water (M14) is picked first in 5 of 8 folds, and M4,
  M14, M15 and T4 dominate the first four picks. For THMFP, the first pick alternates between Namgang Reservoir (3; 4
  folds) and M14 (3 folds), and Imha (M2) is among the first four in 7 of 8 folds once the QA screen is on.
  *(Corrected 2026-09-25: an earlier version said M14 was picked first for both targets.)*
- **Treatment-train stages are the weakest group** (THMFP 0.41, HAAFP 0.63 at k = 4), because their
  deviation from normal depends on plant operation as well as the river.
- **Operational lesson.** A sentinel design needs a QA screen on sentinel results, because one bad reading
  spreads to every site. The M2 September 2017 value should go to the data owner for confirmation.

**Limits.**

- Only the month scheme is tested. A brand-new site needs a calibration period to learn its normal level.
- The sentinel set changes between folds. A fixed operational set should be picked by the same greedy
  procedure on all 16 months, then validated on new data.
- Ridge only. Tree models were not tried here.

## 7. Round 2: model-side levers, honest selection and intervals (2026-09-25)

**Scope.** Six models targeting diagnosed weaknesses, added to `src/models.py` with tests in
`tests/test_models.py`:

| Model | What it does | Weakness targeted |
|---|---|---|
| spline-ridge | cubic splines with linear extrapolation on key chemistry | nonlinearity |
| monotone XGBoost | increasing constraints on organic carbon and network features | tree extrapolation |
| specific-FP XGBoost | predicts log(FP) − log(TOC) | the TOC scale trees cannot extend |
| ridge + XGBoost blend | equal-weight average of the two | combining their strengths |
| random-intercept model | empirical-Bayes site intercept | site offsets |
| chained model | a ridge prediction of the other DBP target fed to XGBoost | shared THMFP/HAAFP signal |

Each was run on the single-sample specs (tier 1, tier 2) and the gap-fill spec (THMFP `t3_basin`, HAAFP
`t4_all`), under the month, site (3 seeds) and forward schemes: 138 cells, all with the same folds and nested
search.

The random-intercept model uses least-squares fixed effects plus a site intercept shrunk by a
method-of-moments variance ratio. statsmodels MixedLM did not converge on this collinear design: it returned
an intercept near 0, and R² was −12 in smoke tests.

The tier 1/2 ridge and XGBoost rows were re-run with 3 site seeds, so they replace the single-seed round-0 rows.

**Honest selection** (`src/select.py`, tests in `tests/test_select.py`).

- Within each deployment scenario, the model for each outer fold is chosen on the *other* folds only.
- 90 % intervals come from a block bootstrap over months (month and forward schemes) or sites (site scheme),
  with 2000 resamples.
- Pairwise differences are bootstrapped on paired blocks.
- Outputs: `results/headline.csv`, `results/headline_diffs.csv`, `results/conformal.csv`,
  `figures/31_round2_paired_diffs.png`.

**Selection-adjusted headline R²** (90 % CI). These are the numbers the paper should quote:

| Scenario | THMFP month | THMFP site | THMFP forward | HAAFP month | HAAFP site | HAAFP forward |
|---|---|---|---|---|---|---|
| S1 single sample | 0.46 [0.29, 0.58] | 0.60 [0.52, 0.66] | −0.03 [−1.19, 0.47] | 0.50 [0.28, 0.69] | 0.70 [0.63, 0.75] | 0.44 [−0.13, 0.78] |
| S2 sentinel, k = 1 | 0.63 [0.54, 0.71] | – | – | 0.68 [0.54, 0.78] | – | – |
| S2 sentinel, k = 4 | 0.65 [0.56, 0.72] | – | – | 0.68 [0.50, 0.80] | – | – |
| S3 network gap-fill | 0.68 [0.58, 0.75] | 0.75 [0.68, 0.80] | 0.42 [0.18, 0.53] | 0.71 [0.58, 0.80] | 0.75 [0.70, 0.79] | 0.59 [0.21, 0.81] |

Sentinel rows are month-scheme only, with sentinels chosen inside each fold. The intervals for S1, S2 and S3
overlap on the month scheme for HAAFP, so the network benefit for HAAFP is suggestive, not established. For
THMFP, S3 is above S1 on new months, and the S1 upper bound of 0.58 sits at the S3 lower bound.

**Paired comparisons** (a − b, 90 % CI; only intervals excluding 0 are claimed):

- **Ridge + XGBoost blend beats XGBoost on tier 2.**
  - THMFP: +0.10 [0.03, 0.20] on new months, +0.02 [0.00, 0.03] on new sites, +0.19 [0.05, 0.50] forward.
  - HAAFP: +0.03 [0.00, 0.05] on new months.
  - Mechanism: the blend inherits ridge's extrapolation.
- **Ridge with context (tier 2) beats chemistry-only XGBoost for THMFP** on new months (+0.09 [0.02, 0.17]) and
  forward (+0.19 [0.08, 0.36]). The reverse holds for HAAFP on new sites (−0.21 [−0.25, −0.17]), where
  chemistry-only XGBoost stays best.
- **Monotone constraints help XGBoost slightly for THMFP**: +0.02 on new sites, +0.05 [0.00, 0.15] forward. No
  effect for HAAFP.
- **The random-intercept model beats ridge on tier 2 for HAAFP on new sites** (+0.14 [0.09, 0.18]), but still
  trails chemistry-only XGBoost (−0.07). It is no better than ridge elsewhere. With the 54-feature `t4_all` set
  its unregularised fixed effects fail on new sites (R² −0.02), so it is not recommended with large feature sets.
- **Spline-ridge is worse than ridge on new months** (THMFP −0.04, HAAFP −0.12) but better for HAAFP on new
  sites (+0.08).
- **The specific-FP target and the chained model show no clear difference** anywhere. The chained model's
  +0.006 for HAAFP forward is negligible.

**Success criteria from the round-2 plan.**

- **S1 missed.** The selection-adjusted honest mean is THMFP 0.53 against a target of 0.54 (round-0 tier-1
  XGBoost 0.49) and HAAFP 0.60 against a target of 0.69 (baseline 0.64). The HAAFP figure falls below the best
  single model because honest selection sometimes picks a model that transfers poorly.
- **THMFP forward R² above 0 without network DBP: missed.** The selection-adjusted value is −0.03; the best
  single model, the random intercept on tier 2, reaches only 0.01.
- **Conformal intervals: met.** Coverage of 90 % intervals is 0.88–0.89 overall and 0.86–0.90 per site group.
  Interval widths are ×/÷ 1.6–2.0 for S1 and ×/÷ 1.5–1.7 for S3. Reservoirs are widest (×/÷ 2.0–2.6).

**What this means for the paper.**

1. Report selection-adjusted numbers with intervals.
2. Chemistry alone predicts DBP formation potential at about R² 0.5–0.7 in this basin. More model complexity
   does not move that ceiling.
3. The practical gains come from information, not algorithms: context and network measurements, especially
   sentinels.
4. Among models, prefer ridge or the ridge + XGBoost blend where extrapolation matters (new months, anomalous
   summers), and chemistry-only XGBoost for HAAFP at new sites.
5. Report factor-of-two prediction intervals alongside point predictions.

## 8. Limitations and next steps

- 16 months = one seasonal cycle; the forward test is a single 4-month period with an anomaly. More years are
  the only cure for the extrapolation weakness.
- Hyper-parameter search is small (12 draws); results are stable enough for model ranking, not for squeezing
  the last 0.02 of R².
- Not tried: ridge + XGBoost blend; multi-output model sharing information between THMFP and HAAFP
  (Spearman 0.67); quantile / interval predictions; explicit bromide-incorporation terms (Br added little here,
  but speciation was not the target).
- Data owner questions from the EDA still apply (units, non-detect limits, the Aug 2018 Hoecheon event).
