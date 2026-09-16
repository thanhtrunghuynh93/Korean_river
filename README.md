# Korean River — Nakdong River DBP formation potential

Data: `data/20260909 DBP Viet.xlsx` (sheet `Total`). 44 sampling points × 16 monthly dates
(May 2017 – Aug 2018) on the Nakdong River basin, South Korea: 18 main-stem sites (M1–M18),
6 tributaries (T1–T6), 5 reservoirs (1–5) and 15 treatment-train samples at three water treatment
plants (Gumi, Goryeong, Bansong; codes 6–20).

| Block | Variables |
|---|---|
| Field (Var 1–6) | Temp, pH, Turbidity, EC, DO, SS |
| Lab (Var 7–14) | Br, BOD, COD, NH3-N, TOC, SUVA, Aromaticity, Mol Weight |
| LC-OCD fractions (Var 15–18) | Biopolymer, HS, BB, LMWN |
| Outputs (Var 19–20) | THMFP, HAAFP |

## Layout

```
data/                 raw workbook + data/processed/dbp_clean.csv (written by preprocessing/load.py)
preprocessing/load.py cleaning rules (non-detects, zero placeholders, site codes, groups, stages)
preprocessing/style.py figure palette and rcParams
notebooks/eda.ipynb   exploratory analysis, writes figures/
notebooks/build_eda_nb.py  regenerates eda.ipynb from source (edit cells here, then re-execute)
figures/              PNG figures numbered in notebook order
reports/eda_findings.md  written summary
docs/index.html       self-contained HTML summary of the EDA and the best models (figures embedded)
```

## Run the EDA

```bash
uv sync
uv run python -m preprocessing.load                 # -> data/processed/dbp_clean.csv
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/eda.ipynb
```

## Predictors for THMFP and HAAFP

Three feature tiers are compared under four validation schemes (see `reports/modeling_findings.md`):

| Tier | Features | Needs at prediction time |
|---|---|---|
| 1 chemistry | core-10 water-quality variables (log10 where skewed) | one water sample |
| 2 context | + site group, treatment stage, WTP, main-stem position, month, fold-fit site encoding | where and when |
| 3 network | + previous-month same-site target/TOC/HS, site history mean, upstream same-month target/TOC | monitoring history |

Schemes: `random` (reference), `month` (leave 2 months out), `site` (leave 4 sites out), `forward`
(train to Apr 2018, test May–Aug 2018). Models: `ridge`, `hgb` (sklearn), `xgb`; baselines: median,
site median, persistence. Hyper-parameters are tuned by a nested random search grouped like the outer scheme.

```
src/features.py      tier builders, upstream map, fold-aware SiteTargetEncoder
src/cv.py            splits(), inner_groups(), metrics()
src/models.py        make_model(name, tier) -> (Pipeline, search space)
src/train.py         CLI; writes results/experiments.csv and results/predictions.csv
src/interpret.py     picks the best tier/model per target, refits, saves models/*.joblib,
                     results/importance.csv (permutation, held-out sites), results/pdp.csv
notebooks/modeling.ipynb   result figures 20–27 (regenerate with notebooks/build_modeling_nb.py)
tests/               leakage and fold-integrity tests
```

Improvement round 1 adds pre-registered feature experiments on top of tier 3 (`src/experiments.py`):
`t3` (baseline), `t3_chem` (fraction ratios, sparse extras), `t3_xlag` (other target's lags, lag-2),
`t3_basin` (same-month network means, downstream neighbour, plant raw water), `t4_all`. Selection rule is the
mean R² over the `month` and `site` schemes; `site`/`random` schemes are repeated over 3 seeds.

```bash
uv run pytest -q
uv run python -m src.train --all                    # ~3 min, 96 cells, single seed
uv run python -m src.train --ablation               # ~10 min, experiments x targets x {ridge,xgb} x {month,site,forward}
uv run python -m src.train --target THMFP --tier 3 --model xgb --scheme site
uv run python -m src.train --target HAAFP --exp t3_basin --model xgb --scheme month --seeds 3
uv run python -m src.interpret                      # picks the best spec/model per target (or --exp/--tier + --model)
uv run python notebooks/build_modeling_nb.py && uv run jupyter nbconvert --execute --inplace notebooks/modeling.ipynb
```

Loading a saved model:

```python
import joblib; from src.features import build
m = joblib.load("models/THMFP_tier3_xgb.joblib")
X, y, meta = build("THMFP", m["tier"])
log10_pred = m["pipeline"].predict(X[m["features"]])
```
