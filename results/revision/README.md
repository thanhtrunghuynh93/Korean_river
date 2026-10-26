# Revision analyses (2026-09-26): outputs and analysis manifest

Produced by the revision plan (P0–P2). Everything here supersedes `results/experiments.csv`, `results/headline*.csv`
and `results/sentinel_*.csv` for the manuscript; those earlier files are kept as the development record (Text S4).

## How to reproduce

```bash
uv run python -m src.rev_audit       # audit/: ledger, flags, dictionary, query list, changes log
uv run python -m src.rev_eval        # eval/: nested predictions for S1a, S1b, S3a, S3b x 6 schemes
uv run python -m src.rev_sentinel    # sentinel/: frozen protocol, ladder, random sets, robustness
uv run python -m src.rev_analyses    # analyses/: performance, tiers, associations, treatment, coherence, decisions
uv run python -m src.rev_paper       # manuscript figures rev_fig2-7 and tables rev_*.tex
uv run pytest -q                     # includes tests/test_rev_eval.py (withholding tests)
```

## Manifest: manuscript claims -> source

| Claim (abstract / results) | File | Rows / columns |
|---|---|---|
| 704 / 701 / 693 / 692 / 692 records; zeros, non-detects | `audit/count_reconciliation.md` | all |
| 429 repeated values, 19 zeros, extreme targets queried | `audit/query_list.csv` | `query` |
| Within-site rho ~0.5 (TOC, HS); bromide -0.32 within site | `analyses/associations.csv` | level = within site |
| HS coefficient 0.36 (THMFP), 0.54 (HAAFP) with site + campaign FE | `analyses/fe_coefficients.csv` | term = log_HS |
| HS beyond TOC: dR2 ~0.09 HAAFP, 0.02-0.08 THMFP | `analyses/tier_diffs.csv` | a = Q4 +HS, b = Q3 +bromide |
| S1a R2 0.49 / 0.64 (THMFP), 0.54 / 0.75 (HAAFP) | `analyses/performance.csv` | scenario S1a, family selected, scheme month / site |
| THMFP 2018/2017 = x1.94 at 95 % of sites; HAAFP x1.01; TOC x1.12; HS x1.27 | `analyses/year_over_year.csv` | variable |
| 8 % of holdout samples with any input out of range; 10 % THMFP above max | `analyses/shift.csv` | last row; y_THMFP |
| Forward THMFP S1a R2 0.00; S3b ridge 0.42 vs XGBoost -0.62 | `analyses/performance.csv` | scheme forward |
| 90 % interval coverage 0.88-0.91 (grouped), 0.53-0.56 THMFP forward S1 | `analyses/performance.csv` | cov90 |
| Co-variation 0.81-0.83 (THMFP), 0.62-0.65 (HAAFP); LOO ranges; OOF residuals | `analyses/coherence.csv`, `analyses/coherence_leave_one_campaign.csv` | analysis, rule |
| Frozen sentinels: THMFP 0.27 (k=4), 0.33 (k=8); random median 0.29 / 0.31; HAAFP no gain | `sentinel/results.csv`, `sentinel/random_sets.csv` | experiment forward_greedy, comparator L5 |
| Screen lowers THMFP (0.27 vs 0.31); protects HAAFP under bias (0.36 vs 0.25) | `sentinel/results.csv` | experiment robust_screen, bias_one_x2 |
| Carbon-normalized HAAFP -53..56 %, THMFP -14..34 % (finished vs raw) | `analyses/treatment_paired.csv` | stage finished, variable sp_* |
| Pre-ozonation at Bansong: THMFP -24 %, HAAFP -32 %, TOC 0 % | `analyses/treatment_paired.csv` | plant Bansong, step 1 |
| Screening: sensitivity 0.70, precision 0.75 (month, S1a); ranking rho 0.84 / 0.87 | `analyses/decisions.csv` | scheme month, family ridge |

## Decisions and caveats recorded during the revision

- Bansong stage order follows site numbering (17 ozonated before 18 settled = pre-ozonation). The first ordering
  (18 -> 17) produced a chemically implausible 22 % TOC increase across ozonation and was abandoned. To be verified.
- The sentinel quality screen and its threshold were introduced after an observed failure in the development analyses;
  the frozen test reports screened, unscreened and median-aggregated results.
- The frozen test is retrospective (May–Aug 2018 had been examined). A new campaign is required for confirmation.
- Units, detection limits, formation-potential protocol and batch structure are unverified (see `audit/query_list.csv`).
- 2026-10-09 (co-author feedback on the Word draft): NH3-N non-detects (25, 5.3 %) are set to LOD/2 = 0.0025 mg/L
  (LOD 0.005 mg/L; US EPA QA/G-9S 2006, Sec. 4.7.1) in `preprocessing/load.py`; other non-detects stay missing.
  NH3-N is not a model input, so only Table 1, the input figures and `audit/changes_log.md` changed. Aromaticity and
  molecular weight are dropped from the paper (still loaded): Q6 is now +SUVA only (`python -m src.rev_analyses --q6`).
