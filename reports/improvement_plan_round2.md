# Improvement plan, round 2 (proposal, 2026-09-25)

## 1. Revision of round 1

**The round-1 gain comes almost entirely from other sites' same-month DBP measurements.** A diagnostic
re-run with ridge (not saved to `results/`) adds input groups one at a time, ordered by what each needs at
prediction time:

| Inputs | Needs at prediction time | THMFP month / site / fwd | HAAFP month / site / fwd |
|---|---|---|---|
| A. Tier 2: chemistry + where/when | one water sample | 0.49 / 0.56 / −0.02 | 0.58 / 0.52 / 0.48 |
| B. + same-month chemistry of the network | chemistry at the other sites | 0.43 / 0.65 / −0.27 | 0.52 / 0.46 / 0.43 |
| C. + own past DBP (lag-1, history) | last month's lab DBP at this site | 0.53 / 0.68 / −0.18 | 0.52 / 0.52 / 0.43 |
| D. + other sites' same-month DBP | this month's lab DBP at neighbours / basin | **0.68 / 0.68 / 0.42** | **0.72 / 0.61 / 0.62** |

What this means:

- The current final models are **network gap-fillers**. They are accurate when THMFP/HAAFP were measured
  at most other sites in the same month, and that is also what drives the better forward extrapolation.
  They are not predictors of DBP from cheap water-quality measurements.
- Network chemistry alone (TOC, HS at other sites) adds nothing, and it hurts on new months.
- Last month's DBP at the same site helps THMFP (+0.08 honest mean) but not HAAFP.
- For the realistic use case, predicting DBP from a water sample, the best numbers are still round 0
  tier-1 XGBoost: THMFP 0.40 / 0.57, HAAFP 0.56 / 0.72 (month / site).

Other weaknesses to fix:

- **Selection optimism.** The finals were chosen from about 20 configurations on the same folds used to
  report them, so the reported R² is slightly optimistic.
- **Noise.** Month-fold R² sd is 0.15–0.30 and the forward test is a single anomalous summer. Differences
  below about 0.05 are not evidence.
- **Confound.** The ridge missing-value indicator changed alongside the features in round 1.

## 2. Proposed round 2

> Update 2026-09-25: the sentinel design (report §6) and steps 2–4 (model levers, honest selection, conformal
> intervals; report §7) are done. The S1 and S2 success criteria were missed; the conformal criterion was met.

Goal: improve the deployable scenarios, and report every scenario separately with honest selection.

### Step 1: reframe results by deployment scenario
Report a headline model per scenario instead of one "best":

- **S1 single sample**: inputs A (or tier 1).
- **S2 monitored site**: inputs C, adding last month's DBP.
- **S3 network gap-fill**: inputs D, which is the current final.

The HTML page and report currently present S3 as "the best model". That wording needs changing.

### Step 2: model-side levers for S1 and S2
These were out of scope in round 1. Each targets a diagnosed weakness.

| Lever | Targets | Cost |
|---|---|---|
| Spline-ridge on TOC, HS, Temp, Turbidity (`SplineTransformer` + ridge) | nonlinear chemistry, extrapolates linearly | low |
| XGBoost with monotone constraints (+ on TOC, HS, BB, lag-1) | tree extrapolation beyond training range | low |
| Tree on specific FP, log(FP/TOC), back-transformed | removes the TOC scale the trees cannot extend | low |
| Out-of-fold stack of ridge + XGBoost (+ spline-ridge) | combine extrapolation with site nonlinearity | medium |
| Mixed-effects model: chemistry + random site intercept nested in group (`statsmodels` MixedLM) | site offsets, 31 % of HAAFP site-scheme error | medium |
| Chained model: predict THMFP, feed the prediction to HAAFP (and the reverse) | shared signal, Spearman 0.67 | medium |

All levers run under the same folds, three seeds, and the nested 12-draw search, with the ridge indicator
fixed on in every arm.

### Step 3: honest selection
Choose among candidates inside an outer leave-months-out and leave-sites-out loop, so the reported score is
selection-adjusted. Add block-bootstrap 90 % intervals over months and sites for every headline R².

### Step 4: prediction intervals
Add group-conditional split-conformal intervals on the log scale for the S1 and S2 winners, and report
coverage. Operators need a range, not just a point.

### Outside modelling
These affect the result more than any model choice:

- More years of data. One seasonal cycle limits every scheme.
- Detection limits for the 29 non-detects.
- Confirmation of the Aug-2018 Hoecheon event and the summer-2018 treatment-train spikes.

## 3. Success criteria

- S1: selection-adjusted honest mean R² at least 0.05 above round 0 tier-1 XGBoost
  (THMFP 0.49 → ≥ 0.54, HAAFP 0.64 → ≥ 0.69), with the gain larger than the bootstrap interval width.
- S2: THMFP forward R² above 0 without other sites' same-month DBP.
- Conformal intervals reach 80–90 % coverage within each site group.
- If a criterion is missed, the report says so and keeps the earlier model.

## 4. Deliverables

- `src/models.py`: new model factories.
- `src/experiments.py`: scenario feature sets A–D.
- `src/select.py`: nested selection and bootstrap intervals.
- `src/conformal.py`: prediction intervals.
- Tests for each new component.
- Notebook section 12 and report section 8.
- HTML page reworded around the three scenarios.
- Estimated compute: about 30–45 min for the grid, run in the background.
