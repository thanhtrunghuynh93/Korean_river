"""Build docs/index.html: self-contained summary of the EDA and the best THMFP / HAAFP models."""
import base64
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "docs" / "index.html"


def img(name: str, alt: str, caption: str) -> str:
    data = base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{alt}" loading="lazy">'
            f"<figcaption>{caption}</figcaption></figure>")


exp = pd.read_csv(ROOT / "results" / "experiments.csv", dtype={"tier": str})
final = pd.read_csv(ROOT / "results" / "final_models.csv")


def r2(target, tier, model, scheme):
    v = exp[(exp.target == target) & (exp.tier == tier) & (exp.model == model) & (exp.scheme == scheme)].r2_log
    return f"{v.iloc[0]:.2f}" if len(v) else "–"


def medape(target, tier, model, scheme):
    v = exp[(exp.target == target) & (exp.tier == tier) & (exp.model == model) & (exp.scheme == scheme)].medape
    return f"{v.iloc[0]:.0%}" if len(v) else "–"


rows = [
    ("THMFP", "1", "xgb", "T1 chemistry · XGBoost"),
    ("THMFP", "3", "xgb", "T3 network · XGBoost (round 0 final)"),
    ("THMFP", "t3_basin", "ridge", "T3 + basin · ridge (final)"),
    ("HAAFP", "1", "xgb", "T1 chemistry · XGBoost"),
    ("HAAFP", "3", "xgb", "T3 network · XGBoost (round 0 final)"),
    ("HAAFP", "t4_all", "ridge", "T3 + all blocks · ridge (final)"),
]
result_rows = "\n".join(
    f"<tr class=\"{'final' if 'final)' in label and 'round 0' not in label else ''}\"><td>{t}</td><td>{label}</td>"
    f"<td>{r2(t, tier, m, 'month')}</td><td>{r2(t, tier, m, 'site')}</td><td>{r2(t, tier, m, 'forward')}</td>"
    f"<td>{medape(t, tier, m, 'site')}</td></tr>"
    for t, tier, m, label in rows)

base_rows = "\n".join(
    f"<tr><td>{t}</td><td>baseline · {b.replace('_', ' ')}</td><td>{r2(t, 'base', b, 'month')}</td>"
    f"<td>{r2(t, 'base', b, 'site')}</td><td>{r2(t, 'base', b, 'forward')}</td><td>{medape(t, 'base', b, 'site')}</td></tr>"
    for t in ("THMFP", "HAAFP") for b in ("site_median", "persistence"))

def card(target, tier, model, feats, note):
    f = final[final.target == target].iloc[0]
    return f"""
<article class="model">
  <div class="model-head"><span class="eyebrow">Best model</span><h3>{target}</h3></div>
  <dl class="scores">
    <div><dt>New months</dt><dd>{f.r2_month:.2f}</dd></div>
    <div><dt>New sites</dt><dd>{f.r2_site:.2f}</dd></div>
    <div><dt>Forward (May–Aug 2018)</dt><dd>{f.r2_forward:.2f}</dd></div>
  </dl>
  <p class="spec"><strong>{model}</strong> on feature set <code>{tier}</code> · saved as <code>{f.path}</code></p>
  <p>{feats}</p>
  <p class="note">{note}</p>
</article>"""

thmfp_card = card("THMFP", "t3_basin", "Ridge regression (median impute → scale → RidgeCV)",
                  "Core-10 chemistry (log-scaled), site group, treatment stage, main-stem position, month, fold-fit site "
                  "encoding, last month's THMFP / TOC / HS at the site, upstream and downstream same-month values, and the "
                  "leave-one-out same-month mean of the other river sites and of the site's own group.",
                  "Up from 0.53 / 0.70 / −0.17 (round-0 XGBoost). The basin mean carries the shared monthly signal, which is "
                  "what let a linear model overtake the trees and extrapolate into the summer-2018 anomaly.")
haafp_card = card("HAAFP", "t4_all", "Ridge regression (median impute → scale → RidgeCV)",
                  "Everything in the THMFP set plus LC-OCD fractions as ratios to TOC, Temp × log TOC, the sparse extras "
                  "(COD, SS, DO, BOD, NH3-N, SUVA, Aromaticity, Mol. weight, NaN-flagged) and last month's THMFP.",
                  "Up from 0.61 / 0.68 / 0.52. Humic substances remain the top chemistry driver; the downstream and upstream "
                  "same-month HAAFP values rank alongside it. For a brand-new site with no network context the chemistry-only "
                  "XGBoost (site R² 0.72) is still the safer pick.")

html = f"""<title>Nakdong DBP Predictors</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --bg:#f7f8f9; --surface:#ffffff; --ink:#1d2530; --ink-2:#55606d; --ink-3:#8a93a0; --rule:#e2e6ea;
  --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; --violet:#4a3aa7; --blue-soft:#e4eefb; --violet-soft:#ece9f7;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#15191e; --surface:#1d2229; --ink:#eef1f4; --ink-2:#b4bcc6; --ink-3:#7f8894; --rule:#2c333c;
    --blue:#5a9be6; --orange:#f0825a; --aqua:#3cc394; --violet:#9085e9; --blue-soft:#1f2c40; --violet-soft:#2a2540;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#15191e; --surface:#1d2229; --ink:#eef1f4; --ink-2:#b4bcc6; --ink-3:#7f8894; --rule:#2c333c;
  --blue:#5a9be6; --orange:#f0825a; --aqua:#3cc394; --violet:#9085e9; --blue-soft:#1f2c40; --violet-soft:#2a2540;
}}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans); font-size:16px; line-height:1.55; }}
.wrap {{ max-width:960px; margin:0 auto; padding-inline:clamp(16px,4vw,32px); padding-block:32px 64px; }}
h1,h2,h3 {{ text-wrap:balance; line-height:1.2; margin:0; }}
h1 {{ font-size:clamp(28px,4.5vw,40px); font-weight:600; letter-spacing:-0.01em; }}
h2 {{ font-size:22px; font-weight:600; margin-bottom:12px; }}
h3 {{ font-size:20px; font-weight:600; }}
p {{ max-width:70ch; margin:0 0 12px; }}
.eyebrow {{ display:block; font-family:var(--mono); font-size:12px; letter-spacing:0.08em; text-transform:uppercase; color:var(--ink-3); margin-bottom:6px; }}
header.top {{ display:grid; gap:16px; padding-bottom:24px; border-bottom:1px solid var(--rule); }}
header.top p.lede {{ font-size:18px; color:var(--ink-2); max-width:62ch; margin:0; }}
.meta {{ font-family:var(--mono); font-size:13px; color:var(--ink-3); }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:24px 0 8px; }}
.kpi {{ background:var(--surface); border:1px solid var(--rule); border-radius:6px; padding:14px 16px; }}
.kpi .v {{ font-family:var(--mono); font-size:26px; font-weight:500; font-variant-numeric:tabular-nums; }}
.kpi .l {{ font-size:13px; color:var(--ink-2); }}
section {{ padding-top:40px; }}
section + section {{ border-top:1px solid var(--rule); margin-top:40px; }}
ul.findings {{ padding-left:0; list-style:none; display:grid; gap:10px; max-width:75ch; margin:0 0 16px; }}
ul.findings li {{ padding-left:18px; position:relative; }}
ul.findings li::before {{ content:""; position:absolute; left:0; top:0.62em; width:8px; height:8px; border-radius:2px; background:var(--blue); }}
ul.findings li strong {{ font-weight:600; }}
figure {{ margin:20px 0; }}
figure img {{ width:100%; height:auto; display:block; border-radius:4px; border:1px solid var(--rule); background:#fcfcfb; }}
figcaption {{ font-size:13px; color:var(--ink-2); margin-top:8px; max-width:80ch; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:20px; }}
.tablewrap {{ overflow-x:auto; margin:16px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:14.5px; min-width:560px; }}
th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--rule); vertical-align:top; }}
th {{ font-weight:600; color:var(--ink-2); font-size:13px; letter-spacing:0.02em; }}
td:nth-child(n+3), th:nth-child(n+3) {{ font-family:var(--mono); font-variant-numeric:tabular-nums; text-align:right; }}
tr.final td {{ background:var(--blue-soft); font-weight:500; }}
.models {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:20px; margin-top:16px; }}
.model {{ background:var(--surface); border:1px solid var(--rule); border-top:4px solid var(--blue); border-radius:6px; padding:20px 22px; }}
.model:nth-child(2) {{ border-top-color:var(--orange); }}
.model-head {{ margin-bottom:12px; }}
.scores {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:0 0 14px; }}
.scores div {{ background:var(--bg); border-radius:4px; padding:10px 12px; }}
.scores dt {{ font-size:12px; color:var(--ink-2); }}
.scores dd {{ margin:2px 0 0; font-family:var(--mono); font-size:24px; font-weight:500; font-variant-numeric:tabular-nums; }}
.spec {{ font-size:14px; color:var(--ink-2); }}
.note {{ font-size:14px; color:var(--ink-2); border-left:2px solid var(--rule); padding-left:10px; margin-bottom:0; }}
code {{ font-family:var(--mono); font-size:0.9em; background:var(--bg); padding:1px 5px; border-radius:3px; }}
pre {{ font-family:var(--mono); font-size:13.5px; background:var(--surface); border:1px solid var(--rule); border-radius:6px; padding:14px 16px; overflow-x:auto; }}
.tiers th:first-child, .tiers td:first-child {{ white-space:nowrap; }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px; vertical-align:middle; margin-right:6px; }}
footer {{ margin-top:48px; padding-top:20px; border-top:1px solid var(--rule); font-size:13px; color:var(--ink-3); }}
a {{ color:var(--blue); }}
a:focus-visible {{ outline:2px solid var(--blue); outline-offset:2px; }}
@media (max-width:480px) {{ .scores {{ grid-template-columns:1fr; }} table {{ min-width:520px; }} }}
</style>

<div class="wrap">
<header class="top">
  <div>
    <span class="eyebrow">Nakdong River basin · South Korea · May 2017 – Aug 2018</span>
    <h1>Predicting disinfection by-product formation potential</h1>
  </div>
  <p class="lede">What the monitoring data says about THMFP and HAAFP across 44 sampling points, and which models predict them
  best when tested on months and sites they never saw.</p>
  <div class="meta">Source workbook <code>20260909 DBP Viet.xlsx</code> · summary generated 2026-09-16 · full detail in
  <code>reports/eda_findings.md</code> and <code>reports/modeling_findings.md</code></div>
</header>

<div class="kpis">
  <div class="kpi"><div class="v">704</div><div class="l">rows = 44 sites × 16 months</div></div>
  <div class="kpi"><div class="v">18 → 2</div><div class="l">predictors → targets (THMFP, HAAFP)</div></div>
  <div class="kpi"><div class="v">0.68</div><div class="l">THMFP R² on unseen months and sites</div></div>
  <div class="kpi"><div class="v">0.71 / 0.66</div><div class="l">HAAFP R² on unseen months / sites</div></div>
</div>

<section id="revision" style="border:1px solid var(--orange);border-radius:6px;padding:16px 20px;margin-top:24px;background:var(--surface)">
  <span class="eyebrow">Revision, 26 September 2026</span>
  <h2>Revised analysis: some results below are superseded</h2>
  <p>A fully nested re-analysis (all tuning and selection inside grouped folds), a frozen sentinel test on May–August 2018
  and new robustness checks changed several conclusions. The manuscript and <code>results/revision/</code> hold the current numbers.</p>
  <ul class="findings">
    <li><strong>THMFP roughly doubled in May–August 2018 versus 2017</strong> (×1.94 at 95% of sites) while HAAFP was unchanged and TOC rose 12%.
    Water-quality predictors failed for THMFP in this period (R² ≈ 0), and 90% intervals covered only 53–56% of observations.
    An analytical cause has not been ruled out.</li>
    <li><strong>The sentinel benefit is smaller than stated in section 6.</strong> With sentinels frozen on earlier months, four to eight sentinels
    raised THMFP R² for summer 2018 to 0.27–0.33; random sites did as well as optimized ones; HAAFP did not benefit.</li>
    <li><strong>Humic substances add information beyond TOC</strong>, most for HAAFP (ΔR² ≈ 0.09); organic carbon tracks change within sites (ρ ≈ 0.5).</li>
    <li><strong>Treatment removes HAA-forming carbon selectively:</strong> carbon-normalized HAAFP falls 53–56% versus 14–34% for THMFP.
    The earlier statement that ozonation raised formation potential at Bansong came from an incorrect stage order (pre-ozonation precedes settling).</li>
  </ul>
</section>

<section id="data">
  <span class="eyebrow">1 · The data</span>
  <h2>A complete monthly panel over four kinds of water</h2>
  <p>Every sampling point has exactly one row per month. Eighteen water-quality inputs are grouped into field measurements
  (Temp, pH, Turbidity, EC, DO, SS), lab measurements (Br, BOD, COD, NH3-N, TOC, SUVA, Aromaticity, Mol. weight) and LC-OCD
  carbon fractions (Biopolymer, Humic substances, Building blocks, LMW neutrals). The outputs are trihalomethane and
  haloacetic-acid formation potentials in mg/L.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Site group</th><th>Codes</th><th>Sites</th><th>Rows</th><th>Median THMFP</th><th>Median HAAFP</th></tr></thead>
    <tbody>
      <tr><td><span class="swatch" style="background:var(--blue)"></span>Main stem, Andong → estuary</td><td>M1–M18</td><td>18</td><td>288</td><td>0.108</td><td>0.122</td></tr>
      <tr><td><span class="swatch" style="background:var(--orange)"></span>Tributaries</td><td>T1–T6</td><td>6</td><td>96</td><td>0.114</td><td>0.130</td></tr>
      <tr><td><span class="swatch" style="background:var(--aqua)"></span>Reservoirs</td><td>1–5</td><td>5</td><td>80</td><td>0.058</td><td>0.062</td></tr>
      <tr><td><span class="swatch" style="background:var(--violet)"></span>Treatment trains (Gumi, Goryeong, Bansong)</td><td>6–20</td><td>15</td><td>240</td><td>0.052</td><td>0.049</td></tr>
    </tbody>
  </table></div>
  {img("02_missingness", "Heatmap of missing values by variable and site group", "Missingness is structural, not random. SUVA, Aromaticity and Mol. weight are absent for every reservoir and treatment-train sample; DO, SS, BOD, COD and NH3-N are mostly absent for treated water. Ten variables are at least 98 % complete everywhere and form the core feature set.")}
</section>

<section id="eda">
  <span class="eyebrow">2 · Key EDA findings</span>
  <h2>Organic carbon quantity, season and place</h2>
  <ul class="findings">
    <li><strong>Both targets are log-normal</strong> (medians 0.08 mg/L, range 0.007–0.55) and correlate with each other at Spearman 0.67. All modelling is on log10.</li>
    <li><strong>Humic substances and TOC are the strongest predictors</strong>: Spearman 0.66–0.78 with the targets. Bromide is weak on its own (≈0.2) and adds little once TOC is known.</li>
    <li><strong>Site and month each explain more variance than any single predictor</strong>: site η² up to 0.59 for HAAFP, date η² up to 0.47 for THMFP on river rows. A random train/test split would leak both.</li>
    <li><strong>River water carries about twice the formation potential of reservoirs and treated water.</strong> Along the main stem, values rise downstream and peak around the Goryeong reach in summer.</li>
    <li><strong>Treatment removes roughly half at coagulation/settling</strong>, ozonation slightly raises it, and GAC or F/A brings finished water to 0.03–0.05 mg/L.</li>
    <li><strong>Summer 2018 is anomalous</strong>: main-stem monthly medians reached 0.32 mg/L against ≤0.21 in every other month, and one tributary event (Hoecheon, Aug 2018) holds record values for several variables.</li>
    <li><strong>Data-quality handling</strong>: 29 Korean non-detect strings set to NaN with flags, 4 zero placeholders in the UV/size trio removed, blank site codes for two treatment trains recovered from names.</li>
  </ul>
  <div class="grid2">
    {img("06_target_monthly", "Monthly median THMFP and HAAFP by site group", "Monthly medians by site group. River water (blue, orange) sits well above reservoirs and treated water and peaks in summer; the 2018 summer spike is visible in both targets.")}
    {img("12_corr_ranked", "Spearman correlation of each predictor with THMFP and HAAFP", "Spearman correlation of every predictor with each target, for all rows, river rows and treatment-train rows. Correlations are weaker inside treatment trains for almost every predictor.")}
  </div>
  {img("08_treatment_train", "TOC, THMFP and HAAFP through the treatment train at three plants", "Median and interquartile range through the treatment train at the three water treatment plants.")}
</section>

<section id="models">
  <span class="eyebrow">3 · Modelling</span>
  <h2>Three feature tiers, four ways of holding data out</h2>
  <p>Features were added in nested tiers so the value of each kind of information could be measured. Every model was tuned
  by a nested random search whose inner folds share the outer scheme's grouping, so held-out months and sites never leak into
  tuning. Ridge, HistGradientBoosting and XGBoost were compared against median, site-median and persistence baselines.</p>
  <div class="tablewrap"><table class="tiers">
    <thead><tr><th>Tier</th><th>Adds</th><th>Needs at prediction time</th></tr></thead>
    <tbody>
      <tr><td>T1 chemistry</td><td>Core-10 water-quality variables, log-scaled where skewed</td><td>one water sample</td></tr>
      <tr><td>T2 context</td><td>site group, treatment stage, plant, main-stem position, month, fold-fit site encoding</td><td>where and when</td></tr>
      <tr><td>T3 network</td><td>previous-month target / TOC / HS at the site, site history mean, upstream same-month target and TOC</td><td>monitoring history</td></tr>
      <tr><td>Round 1 blocks</td><td>basin-wide same-month means, downstream neighbour, plant raw water; LC-OCD ratios to TOC and sparse extras; other target's lags</td><td>the network sampled that month</td></tr>
    </tbody>
  </table></div>
  <div class="tablewrap"><table>
    <thead><tr><th>Scheme</th><th>Holds out</th><th>Answers</th></tr></thead>
    <tbody>
      <tr><td>month</td><td>2 whole months × 8 folds</td><td>can we predict a new month at known sites?</td></tr>
      <tr><td>site</td><td>4 whole sites × 11 folds, 3 seeds</td><td>can we predict a site we never sampled?</td></tr>
      <tr><td>forward</td><td>May–Aug 2018 after training on the first 12 months</td><td>strict temporal extrapolation</td></tr>
      <tr><td>random</td><td>random rows</td><td>optimistic reference only</td></tr>
    </tbody>
  </table></div>
  {img("20_cv_r2_by_tier_model", "Out-of-fold R² by feature tier and model under each scheme", "Round 0: out-of-fold R² by tier and model. Random splits flatter every model; the month and site schemes are the honest ones. Trees win on new sites, ridge holds up better when the test months differ from training.")}
  {img("28_ablation_delta_r2", "Gain in R² over tier 3 for each round-1 feature block", "Round 1: gain over tier 3 from each feature block. The basin-wide same-month features (+basin) lift the month scheme for both targets and are the only block that helps forward extrapolation for ridge; cross-target lags (+xlag) are noise-level.")}
  <div class="tablewrap"><table>
    <thead><tr><th>Target</th><th>Model</th><th>R² month</th><th>R² site</th><th>R² forward</th><th>Median abs. % error (site)</th></tr></thead>
    <tbody>
      {base_rows}
      {result_rows}
    </tbody>
  </table></div>
  <p class="note">R² is on the log10 target and pooled over out-of-fold predictions; the site scheme is averaged over 3 seeds
  for round-1 rows. Highlighted rows are the current final models.</p>
</section>

<section id="best">
  <span class="eyebrow">4 · Best models</span>
  <h2>One recommendation per target</h2>
  <p>Selection rule fixed before the experiments: the highest mean R² over the month and site schemes. Both winners are ridge
  regressions on network-aware features; the round-0 XGBoost models are kept in <code>models/</code> for the chemistry-only case.</p>
  <div class="models">
    {thmfp_card}
    {haafp_card}
  </div>
  <div class="grid2">
    {img("22_pred_vs_obs_site", "Predicted versus observed on held-out sites for each tier", "Round 0, leave-sites-out: predicted vs observed on log axes, each point from a model that never saw its site. Spread tightens from chemistry-only (left) to the network tier (right).")}
    {img("29_new_feature_importance", "Permutation importance of the round-1 final models", "Permutation importance of the final ridge models on held-out sites, coloured by feature block. Same-month network values (violet) lead for both targets, followed by humic substances and last month's value.")}
  </div>
  <h3 style="margin-top:24px">Which model for which situation</h3>
  <div class="tablewrap"><table>
    <thead><tr><th>Situation</th><th>Use</th><th>Expected R² (month / site)</th></tr></thead>
    <tbody>
      <tr><td>The monitoring network was sampled this month</td><td>final ridge models above</td><td>THMFP 0.68 / 0.68 · HAAFP 0.71 / 0.66</td></tr>
      <tr><td>A new site inside the network, THMFP</td><td><code>t4_all</code> XGBoost or the ridge final</td><td>site 0.75 / 0.68</td></tr>
      <tr><td>A new site, HAAFP, chemistry only</td><td>round-0 tier-1 XGBoost</td><td>site 0.72</td></tr>
      <tr><td>One sample, no history or network</td><td>round-0 tier-1 XGBoost</td><td>THMFP 0.58 · HAAFP 0.72 (site)</td></tr>
    </tbody>
  </table></div>
</section>

<section id="why-ridge">
  <span class="eyebrow">5 · Why ridge beat XGBoost</span>
  <h2>A linear model won because the task became one of propagating a measured shift</h2>
  <p>In round 0 the tree model led on unseen sites, where the job was to learn site-specific nonlinearities. Round 1 added the
  network's same-month values, and with them the strongest signal became a shift that has to be carried through to the
  prediction. That is a linear job, and it exposed a structural limit of trees.</p>
  <ul class="findings">
    <li><strong>Trees cannot predict beyond the training range.</strong> A boosted tree returns a constant inside each leaf, so its
    output never exceeds the largest value it was trained on. In the forward holdout one in ten THMFP observations sits above the
    training maximum of 0.29 mg/L. XGBoost tops out at 0.15 mg/L there and under-predicts those points by about a factor of four;
    ridge keeps extending along its slope, reaches 0.38 mg/L, and halves that error.</li>
    <li><strong>The gap is concentrated in anomalous months.</strong> Within the leave-months-out scheme the two models are close on
    typical months. On the high months of 2018 ridge still explains part of the variance while XGBoost explains almost none.</li>
    <li><strong>The new features are nearly linear in the log target.</strong> Ridge spreads its weight almost evenly across the
    group mean, basin mean, site encoding and last month's value, so when the whole basin rises every input pushes the prediction
    up together. A tree must rediscover each of those as step functions from about 500 rows.</li>
    <li><strong>Strong regularisation suits a small, collinear sample.</strong> Ridge chose a heavy penalty and the inputs are highly
    correlated (TOC with humic substances 0.78, humic substances with building blocks 0.81). Shrinking correlated coefficients
    together is what ridge does well; the trees show the larger fold-to-fold spread.</li>
    <li><strong>XGBoost still wins on unseen sites inside known months.</strong> With all feature blocks it reaches site R² 0.75 for
    THMFP against 0.68 for ridge, so the tree model stays the recommendation for a new site in a known period.</li>
  </ul>
  <div class="tablewrap"><table>
    <thead><tr><th>THMFP, basin feature set</th><th>Model</th><th>Max prediction, forward holdout</th><th>Error on points above training max</th><th>R² typical months</th><th>R² anomalous months</th></tr></thead>
    <tbody>
      <tr><td>observed maximum 0.551 mg/L</td><td>ridge</td><td>0.379 mg/L</td><td>−0.20 log10 (÷1.6)</td><td>0.70</td><td>0.28</td></tr>
      <tr><td></td><td>XGBoost</td><td>0.146 mg/L</td><td>−0.59 log10 (÷3.9)</td><td>0.65</td><td>0.03</td></tr>
    </tbody>
  </table></div>
  <p class="note">Anomalous months are those whose basin-wide median THMFP exceeds 0.15 mg/L (176 of 693 rows). Two caveats: the
  ridge pipeline also gained a missing-value indicator in round 1, which raised its own baseline, and the forward test rests on a
  single anomalous summer. Monotone constraints or a ridge–tree blend would likely narrow the gap; both were outside this round's scope.</p>
</section>

<section id="sentinel">
  <span class="eyebrow">6 · Sentinel sampling</span>
  <h2>Lab-test a few sites, predict the rest</h2>
  <p>The best models above need THMFP and HAAFP measured at most other sites in the same month, which makes them
  gap-fillers. They work because the basin moves together: 57 % of THMFP's month-to-month variation within a site is
  shared across all sites that month (HAAFP 36 %), and main-stem sites twelve steps apart still correlate at 0.72 for
  THMFP. A sentinel design exploits that signal at a fraction of the lab cost. Chemistry is measured everywhere, DBP
  only at <em>k</em> sentinel sites, and every other site is predicted from its chemistry, its normal level from this
  16-month campaign, and how far the sentinels sit from their own normal level that month.</p>
  {img("30_sentinel_design", "R-squared on non-sentinel sites as the number of sentinel sites grows", "R² on non-sentinel sites in held-out months as sentinels are added. Sentinels were chosen by greedy selection inside each fold, never on the test months. The dotted line is the run without the QA screen; the dashed line is full-network gap-filling.")}
  <div class="tablewrap"><table>
    <thead><tr><th>Sentinels per month</th><th>Share of lab tests</th><th>THMFP R²</th><th>HAAFP R²</th></tr></thead>
    <tbody>
      <tr><td>0 (chemistry + calibration level)</td><td>0 %</td><td>0.49</td><td>0.58</td></tr>
      <tr><td>1</td><td>2 %</td><td>0.63</td><td>0.68</td></tr>
      <tr><td>2</td><td>5 %</td><td>0.58</td><td>0.69</td></tr>
      <tr><td>4</td><td>9 %</td><td>0.65</td><td>0.68</td></tr>
      <tr><td>8</td><td>18 %</td><td>0.68</td><td>0.72</td></tr>
      <tr class="final"><td>all sites (current finals)</td><td>≈100 %</td><td>0.68</td><td>0.71</td></tr>
    </tbody>
  </table></div>
  <ul class="findings">
    <li><strong>One sentinel recovers most of the benefit</strong> with 2 % of the lab tests; four to eight match
    full-network gap-filling for THMFP, and one or two are enough for HAAFP.</li>
    <li><strong>Useful sentinels differ by target.</strong> For HAAFP, Bansong raw water (M14) is chosen first in five of
    eight folds; for THMFP the first choice alternates between Namgang Reservoir and M14.</li>
    <li><strong>A QA screen is essential.</strong> Without it, one reading (Imha Reservoir, September 2017, THMFP 0.007 mg/L,
    the dataset minimum) dragged every site's prediction down by about a factor of three, and THMFP R² fell to 0.31
    with two sentinels. Sentinel readings more than four robust SDs from the site's normal level are now dropped. The
    screen was added after seeing that failure, so the screened numbers are slightly optimistic.</li>
    <li><strong>Limits.</strong> Tested on new months only; a new site needs a calibration period. Treated-water stages are
    the weakest group. A fixed operational sentinel set still has to be chosen and validated on new data.</li>
  </ul>
</section>

<section id="caveats">
  <span class="eyebrow">7 · Caveats</span>
  <h2>What to keep in mind</h2>
  <ul class="findings">
    <li><strong>One seasonal cycle.</strong> Sixteen months is a single year plus four; the forward test is one anomalous summer. Expect degradation in unusual years and retrain as months arrive.</li>
    <li><strong>Network features need the network.</strong> Basin, group, downstream and plant-raw features assume the other sites were sampled the same month. Without them, fall back to the chemistry-only models.</li>
    <li><strong>Units are assumed</strong> (mg/L for the targets, µg/L C for LC-OCD fractions, µS/cm for EC); the workbook does not state them. Detection limits for the 29 non-detects are also unknown.</li>
    <li><strong>One model-side change slipped into round 1</strong>: ridge gained a missing-value indicator, which raised its own baseline. Feature gains are measured against the re-run baseline, so they are like-for-like.</li>
    <li><strong>Not yet tried</strong>: ridge–XGBoost blending, a joint two-target model, prediction intervals, robust losses.</li>
  </ul>
</section>

<footer>
  Reproduce: <code>uv sync</code> → <code>uv run python -m src.train --all</code> → <code>uv run python -m src.train --ablation</code> →
  <code>uv run python -m src.interpret</code>. Notebooks <code>notebooks/eda.ipynb</code> and <code>notebooks/modeling.ipynb</code> render every figure.
  Repository: <a href="https://github.com/thanhtrunghuynh93/Korean_river">github.com/thanhtrunghuynh93/Korean_river</a>.
</footer>
</div>
"""

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html, encoding="utf-8")
print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
