"""LaTeX tables for the manuscript, generated from results/ so every number traces to a CSV.

    uv run python -m src.paper_tables     # -> Paper_templates/Template_paper/contents/tables/*.tex
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.load import GROUP_LABEL, SITE_ORDER, load_clean

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
P = R / "paper"
OUT = ROOT / "Paper_templates" / "Template_paper" / "contents" / "tables"

LABEL = {"Temp": "Temperature", "pH": "pH", "Turbidity": "Turbidity", "EC": "EC", "DO": "DO", "SS": "SS", "Br": "Br$^-$",
         "BOD": "BOD", "COD": "COD", "NH3N": "NH$_3$-N", "TOC": "TOC", "SUVA": "SUVA", "Aromaticity": "Aromaticity",
         "MolWeight": "Mol.\\ weight", "Biopolymer": "Biopolymers", "HS": "Humic substances", "BB": "Building blocks",
         "LMWN": "LMW neutrals", "THMFP": "THMFP", "HAAFP": "HAAFP"}
UNIT = {"°C": "$^\\circ$C", "-": "--", "NTU": "NTU", "µS/cm": "$\\mu$S/cm", "mg/L": "mg/L", "L/mg·m": "L/(mg$\\cdot$m)",
        "Da": "Da", "µg/L C": "$\\mu$g C/L"}


def fmt(v: float) -> str:
    if pd.isna(v):
        return "--"
    a = abs(v)
    if a >= 100: return f"{v:.0f}"
    if a >= 10: return f"{v:.1f}"
    if a >= 1: return f"{v:.2f}"
    return f"{v:.3f}"


def pfmt(p: float) -> str:
    if pd.isna(p): return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def table1() -> str:
    d = pd.read_csv(P / "table1_descriptive.csv")
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Descriptive statistics of the 16 input variables and the two targets over all 701 samples. "
             r"Group columns give medians; Kruskal--Wallis tests compare the four site groups and, for river samples "
             r"only, the four seasons. Units are as reported in the monitoring workbook \todo{confirm units with the data provider}.}",
             r"\label{tab:descriptive}", r"\footnotesize", r"\setlength{\tabcolsep}{3.2pt}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule",
             r" & & & & & & \multicolumn{4}{c}{Median by site group} & \multicolumn{2}{c}{Kruskal--Wallis $p$} \\",
             r"\cmidrule(lr){7-10}\cmidrule(lr){11-12}",
             r"Variable & Unit & $n$ & Median & Range & Skew. & Main stem & Tributary & Reservoir & WTP train & Group & Season \\",
             r"\midrule"]
    for _, r in d.iterrows():
        if r.variable == "THMFP":
            lines.append(r"\midrule")
        rng = f"{fmt(r['min'])}--{fmt(r['max'])}"
        lines.append(" & ".join([LABEL[r.variable], UNIT.get(r.unit, r.unit), f"{int(r.n)}", fmt(r["median"]), rng, f"{r.skewness:.2f}",
                                 fmt(r.median_main_stem), fmt(r.median_tributary), fmt(r.median_reservoir), fmt(r.median_treatment),
                                 pfmt(r.kw_group_p), pfmt(r.kw_season_p)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def table4() -> str:
    t = pd.read_csv(P / "table_scenarios.csv")
    curve = pd.read_csv(P / "sentinel_curve.csv")

    def cell(target, scen, scheme):
        e = t[(t.target == target) & (t.scenario == scen) & (t.scheme == scheme)]
        if e.empty: return "--"
        e = e.iloc[0]
        return f"{e.r2_selected:.2f} [{e.ci_lo:.2f}, {e.ci_hi:.2f}]"

    def pi(target, scen):
        e = t[(t.target == target) & (t.scenario == scen) & t.scheme.isin(["month", "site"])].dropna(subset=["fold_change"])
        if e.empty: return "--"
        return f"{e.fold_change.min():.1f}--{e.fold_change.max():.1f}"

    rows = []
    for scen, lab in [("S1 single sample", "S1 single sample")]:
        rows.append((lab, *[cell(tg, scen, s) for tg in ["THMFP", "HAAFP"] for s in ["month", "site", "forward"]]))
    for k in (1, 4, 8):
        vals = []
        for tg in ["THMFP", "HAAFP"]:
            c = curve[(curve.target == tg) & (curve.k == k)].iloc[0]
            vals += [f"{c.r2:.2f} [{c.ci_lo:.2f}, {c.ci_hi:.2f}]", "--", "--"]
        rows.append((f"S2 sentinel, $k={k}$ sites", *vals))
    rows.append(("S3 network gap-fill", *[cell(tg, "S3 network gap-fill", s) for tg in ["THMFP", "HAAFP"] for s in ["month", "site", "forward"]]))
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Selection-adjusted out-of-fold $R^2$ (log$_{10}$ scale) with 90\% block-bootstrap confidence intervals "
             r"for the three deployment scenarios. New months: leave-two-months-out; new sites: leave-four-sites-out "
             r"(three seeds); forward: trained on May 2017--April 2018 and tested on May--August 2018. Sentinel "
             r"scenarios were evaluated on new months only. The last two rows give the width of group-conditional "
             r"90\% conformal prediction intervals as a multiplicative factor ($\times/\div$) and their empirical coverage.}",
             r"\label{tab:scenarios}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{THMFP} & \multicolumn{3}{c}{HAAFP} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Scenario & New months & New sites & Forward & New months & New sites & Forward \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    cov = t.dropna(subset=["coverage"])
    lines += [r"\midrule",
              f"90\\% PI width, S1 ($\\times/\\div$) & \\multicolumn{{3}}{{c}}{{{pi('THMFP', 'S1 single sample')}}} & \\multicolumn{{3}}{{c}}{{{pi('HAAFP', 'S1 single sample')}}} \\\\",
              f"90\\% PI width, S3 ($\\times/\\div$) & \\multicolumn{{3}}{{c}}{{{pi('THMFP', 'S3 network gap-fill')}}} & \\multicolumn{{3}}{{c}}{{{pi('HAAFP', 'S3 network gap-fill')}}} \\\\",
              f"PI coverage (nominal 0.90) & \\multicolumn{{6}}{{c}}{{{cov.coverage.min():.2f}--{cov.coverage.max():.2f}}} \\\\",
              r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def si_sites() -> str:
    df = load_clean()
    s = df[["Site", "Name_EN", "Sample_KR", "group", "Position"]].drop_duplicates("Site")
    s["Site"] = pd.Categorical(s.Site.astype(str), SITE_ORDER, ordered=True)
    s = s.sort_values("Site")
    esc = lambda x: str(x).replace("&", r"\&").replace("%", r"\%").replace("–", "--").replace(";", ",")
    lines = [r"\begin{longtable}{p{1.1cm}p{4.4cm}p{2.6cm}p{6.2cm}}",
             r"\caption{Sampling sites (codes as in the monitoring workbook; site codes 12--20 were inferred from sample names).}\label{tab:s_sites}\\",
             r"\toprule Code & Name & Group & Position \\ \midrule \endfirsthead", r"\toprule Code & Name & Group & Position \\ \midrule \endhead",
             r"\bottomrule \endfoot"]
    for _, r in s.iterrows():
        lines.append(f"{r.Site} & {esc(r.Name_EN)} & {GROUP_LABEL[str(r.group)].split(' (')[0]} & {esc(r.Position)} \\\\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def si_treatment() -> str:
    t = pd.read_csv(P / "treatment_changes.csv")
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{Change relative to the plant's raw water (median of 16 paired months, \% with interquartile range) and "
             r"Wilcoxon signed-rank $p$.}", r"\label{tab:s_treatment}", r"\footnotesize",
             r"\begin{tabular}{llrrrr}", r"\toprule", r"WTP & Stage & Variable & Median change (\%) & IQR (\%) & $p$ \\", r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.wtp} & {r.stage} & {r.variable} & {r.median_pct_change:+.1f} & {r.q1_pct:+.0f} to {r.q3_pct:+.0f} & {pfmt(r.wilcoxon_p)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def si_diffs() -> str:
    d = pd.read_csv(R / "headline_diffs.csv")
    name = {"2|mixed": "random intercept (T2)", "2|ridge": "ridge (T2)", "1|xgb": "XGBoost (T1)", "2|spline_ridge": "spline-ridge (T2)",
            "2|xgb_mono": "monotone XGBoost (T2)", "2|xgb": "XGBoost (T2)", "2|xgb_spec": "specific-FP XGBoost (T2)",
            "2|blend": "ridge+XGBoost blend (T2)", "1|chained": "chained (T1)", "t3_basin|ridge": "ridge (S3 basin)",
            "t3_basin|xgb": "XGBoost (S3 basin)", "t4_all|ridge": "ridge (S3 all)", "t4_all|xgb": "XGBoost (S3 all)",
            "t3_basin|mixed": "random intercept (S3 basin)", "t4_all|mixed": "random intercept (S3 all)"}
    sch = {"month": "new months", "site": "new sites", "forward": "forward"}
    m = lambda v: f"{v:+.3f}".replace("-", "$-$")
    head = r"\toprule Target & a vs b & Scheme & $\Delta R^2$ & 90\% CI \\ \midrule"
    lines = [r"{\footnotesize\begin{longtable}{lllrl}",
             r"\caption{Paired differences in $R^2$ between model variants (a $-$ b) with 90\% block-bootstrap confidence "
             r"intervals (Fig.~S7). T1: chemistry only; T2: chemistry and context; S3: network gap-fill feature sets.}"
             r"\label{tab:s_diffs}\\",
             head + r" \endfirsthead", head + r" \endhead", r"\bottomrule \endfoot"]
    for _, r in d.iterrows():
        if r.a not in name or r.b not in name: continue
        lines.append(f"{r.target} & {name[r.a]} vs {name[r.b]} & {sch[r.scheme]} & {m(r.delta_r2)} & [{m(r.ci_lo)}, {m(r.ci_hi)}] \\\\")
    lines += [r"\end{longtable}}"]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table1_descriptive.tex").write_text(table1())
    t4 = table4()
    for a, b in [("[-", "[$-$"), (" -0", " $-$0"), (" -1", " $-$1"), ("& -", "& $-$")]:
        t4 = t4.replace(a, b)
    (OUT / "table4_scenarios.tex").write_text(t4)
    (OUT / "si_sites.tex").write_text(si_sites())
    (OUT / "si_treatment.tex").write_text(si_treatment())
    (OUT / "si_diffs.tex").write_text(si_diffs())
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
