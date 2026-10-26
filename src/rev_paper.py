"""Revised manuscript figures and tables from results/revision (journal size).

    uv run python -m src.rev_paper     # -> Paper_templates/Template_paper/assets/figs/rev_*.pdf|png, contents/tables/rev_*.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from preprocessing import style
from preprocessing.load import GROUP_LABEL, GROUP_ORDER, PREDICTORS, SITE_ORDER, UNITS, load_clean
from src.features import base_frame
from src.paper_figures import MM, W2, panel, setup

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "revision" / "analyses"
S = ROOT / "results" / "revision" / "sentinel"
FIG = ROOT / "Paper_templates" / "Template_paper" / "assets" / "figs"
TAB = ROOT / "Paper_templates" / "Template_paper" / "contents" / "tables"
TC = style.TARGET_COLORS
GC = style.GROUP_COLORS
TARGETS = ("THMFP", "HAAFP")
FAM = {"ridge": style.GROUP_COLORS["main_stem"], "xgb": style.GROUP_COLORS["tributary"], "blend": style.GROUP_COLORS["reservoir"],
       "selected": style.GROUP_COLORS["treatment"]}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(FIG / f"{name}.png", dpi=300); plt.close(fig); print("saved", name)


# ------------------------------------------------------------------------------------------ Fig 2: variation
def fig_variation():
    bf = base_frame(); bf["Site"] = bf.Site.astype(str)
    yoy = pd.read_csv(A / "year_over_year.csv")
    fig = plt.figure(figsize=(W2, 150 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=0.45, wspace=0.25)
    order = [s for s in SITE_ORDER if s in set(bf.Site)]
    for j, t in enumerate(TARGETS):
        ax = fig.add_subplot(gs[0, j])
        y = f"y_{t}"
        a = bf.assign(anom=bf[y] - bf.groupby("Site")[y].transform("mean")).pivot_table(index="Site", columns="Date", values="anom").reindex(order)
        im = ax.imshow(a.to_numpy(), cmap=style.DIV_CMAP, vmin=-0.6, vmax=0.6, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=4.8); ax.grid(False)
        ax.set_xticks(range(0, a.shape[1], 2)); ax.set_xticklabels([d.strftime("%Y-%m") for d in a.columns[::2]], rotation=60, fontsize=5.5)
        for g_end in (17.5, 23.5, 28.5):
            ax.axhline(g_end, color=style.SURFACE, lw=1.2)
        ax.set_title(f"{t}: site-centred log$_{{10}}$ anomaly", loc="left"); panel(ax, "ab"[j])
        cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02); cb.ax.tick_params(labelsize=5.5); cb.set_label("log$_{10}$ anomaly", fontsize=6)
    ax = fig.add_subplot(gs[1, :])
    v = yoy[yoy.variable != "Temp"].reset_index(drop=True)
    x = np.arange(len(v))
    ax.bar(x, v.median_site_change, color=[TC.get(n, style.TEXT_2) for n in v.variable], width=0.55)
    ax.errorbar(x, v.median_site_change, yerr=[v.median_site_change - v.ci_lo, v.ci_hi - v.median_site_change], fmt="none", ecolor=style.TEXT, capsize=2, lw=0.8)
    for xi, r in zip(x, v.itertuples()):
        ax.text(xi, max(r.ci_hi, 0) + 0.015, f"×{10 ** r.median_site_change:.2f}\n{r.share_sites_increase:.0%} of sites ↑", ha="center", va="bottom", fontsize=5.6)
    ax.axhline(0, color=style.TEXT_2, lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(v.variable); ax.set_ylabel("median site change, log$_{10}$(2018 / 2017)")
    ax.set_ylim(-0.05, 0.45)
    temp = yoy[yoy.variable == "Temp"].iloc[0]
    ax.set_title(f"Same months (May–Aug), 2018 vs 2017 (90% CI over sites); water temperature {temp.median_site_change:+.1f} °C", loc="left"); panel(ax, "c")
    save(fig, "rev_fig2_variation")


# ------------------------------------------------------------------------------------------ Fig 3: precursors
def fig_precursors():
    asc = pd.read_csv(A / "associations.csv"); tiers = pd.read_csv(A / "tiers.csv"); fe = pd.read_csv(A / "fe_coefficients.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 66 * MM), gridspec_kw={"width_ratios": [1.2, 1.15, 0.8], "wspace": 0.75})
    ax = axes[0]
    preds = ["log_TOC", "log_HS", "log_Br", "Temp"]; levels = ["pooled", "site means", "within site"]
    lc = {"pooled": style.TEXT_2, "site means": GC["main_stem"], "within site": GC["tributary"]}
    for i, t in enumerate(TARGETS):
        for j, lev in enumerate(levels):
            d = asc[(asc.target == t) & (asc.level == lev)].set_index("predictor").reindex(preds)
            y = np.arange(len(preds)) + i * (len(preds) + 1)
            ax.errorbar(d.rho, y + (j - 1) * 0.25, xerr=[d.rho - d.ci_lo, d.ci_hi - d.rho], fmt="o", color=lc[lev], ms=3, lw=0.8,
                        label=lev if i == 0 else None)
    ticks = [f"{t} · {p.replace('log_', '')}" for t in TARGETS for p in preds]
    ax.set_yticks([k + i * (len(preds) + 1) for i in range(2) for k in range(len(preds))]); ax.set_yticklabels(ticks, fontsize=6)
    ax.axvline(0, color=style.TEXT_2, lw=0.7); ax.set_xlabel("Spearman ρ (90% CI, site bootstrap)"); ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", ncol=3, bbox_to_anchor=(0.45, -0.18)); ax.set_title("Association at three levels", loc="left"); panel(ax, "a")
    ax = axes[1]
    tier_names = ["Q1 field", "Q2 +TOC", "Q3 +bromide", "Q4 +HS", "Q5 +all LC-OCD"]
    for t in TARGETS:
        for scheme, ls in (("month", "-"), ("site", "--")):
            d = tiers[(tiers.target == t) & (tiers.scheme == scheme) & (tiers.family == "ridge")].set_index("tier").reindex(tier_names)
            ax.errorbar(range(5), d.r2, yerr=[d.r2 - d.ci_lo, d.ci_hi - d.r2], color=TC[t], ls=ls, marker="o", ms=3, lw=1.1, capsize=1.5,
                        label=f"{t}, {'new months' if scheme == 'month' else 'new sites'}")
    ax.set_xticks(range(5)); ax.set_xticklabels([n.split(" ", 1)[1] for n in tier_names], rotation=30, ha="right")
    ax.set_ylabel("R² (ridge, common cohort)"); ax.legend(frameon=False, fontsize=5.6, loc="lower right")
    ax.set_title("Measurement tiers", loc="left"); panel(ax, "b")
    ax = axes[2]
    for i, t in enumerate(TARGETS):
        d = fe[fe.target == t].set_index("term")
        for j, term in enumerate(["log_TOC", "log_HS"]):
            yy = i * 3 + j
            ax.errorbar(d.loc[term, "estimate"], yy, xerr=[[d.loc[term, "estimate"] - d.loc[term, "ci_lo"]], [d.loc[term, "ci_hi"] - d.loc[term, "estimate"]]],
                        fmt="o", color=TC[t], ms=3.5, lw=1)
    ax.set_yticks([0, 1, 3, 4]); ax.set_yticklabels(["THM·TOC", "THM·HS", "HAA·TOC", "HAA·HS"], fontsize=6)
    ax.axvline(0, color=style.TEXT_2, lw=0.7); ax.invert_yaxis(); ax.set_xlabel("coefficient (log–log), 90% CI")
    ax.set_title("Site + campaign FE", loc="left"); panel(ax, "c")
    save(fig, "rev_fig3_precursors")


# ------------------------------------------------------------------------------------------ Fig 4: treatment
def fig_treatment():
    t = pd.read_csv(A / "treatment_paired.csv")
    vars_ = ["TOC", "HS", "THMFP", "HAAFP", "sp_THMFP", "sp_HAAFP"]
    lab = {"TOC": "TOC", "HS": "HS", "THMFP": "THMFP", "HAAFP": "HAAFP", "sp_THMFP": "THMFP/TOC", "sp_HAAFP": "HAAFP/TOC"}
    fig, axes = plt.subplots(1, 3, figsize=(W2, 70 * MM), sharey=False, gridspec_kw={"wspace": 0.08})
    for ax, plant, letter in zip(axes, ["Gumi", "Goryeong", "Bansong"], "abc"):
        d = t[(t.plant == plant) & t.variable.isin(vars_)]
        steps = d.drop_duplicates("step").sort_values("step")
        mat = d.pivot_table(index="variable", columns="step", values="adjacent_reduction_pct").reindex(vars_)
        im = ax.imshow(mat.to_numpy(), cmap=style.DIV_CMAP.reversed(), vmin=-60, vmax=60, aspect="auto")
        for (r, c), v in np.ndenumerate(mat.to_numpy()):
            if not np.isnan(v): ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=5.8, color="white" if abs(v) > 38 else style.TEXT)
        ax.set_xticks(range(len(steps))); ax.set_xticklabels([f"→{s}" for s in steps.stage], rotation=35, ha="right", fontsize=6)
        ax.set_yticks(range(len(vars_))); ax.grid(False)
        ax.set_yticklabels([lab[v] for v in vars_] if plant == "Gumi" else [])
        fin = d[d.stage == "finished"].set_index("variable").raw_relative_reduction_pct
        ax.set_title(plant, loc="left"); panel(ax, letter)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01); cb.set_label("reduction vs previous stage (%)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    fig.suptitle("Adjacent-stage reduction (%) along the treatment trains (median of paired campaigns)", x=0.01, ha="left", fontsize=7.5, fontweight="bold")
    save(fig, "rev_fig4_treatment")


# ------------------------------------------------------------------------------------------ Fig 5: prediction
def fig_prediction():
    p = pd.read_csv(A / "performance.csv")
    scen_lab = {"S1a": "S1a water quality", "S1b": "S1b + site level", "S3b": "S3 network gap-fill"}
    schemes = [("month", "new months"), ("site", "new sites"), ("forward", "forward"), ("rolling", "rolling origin")]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 72 * MM), sharey=True)
    for ax, t, letter in zip(axes, TARGETS, "ab"):
        yl = []
        for i, (scen, sl) in enumerate(scen_lab.items()):
            for j, (sch, schl) in enumerate(schemes):
                yy = i * (len(schemes) + 1) + j
                yl.append((yy, f"{sl} · {schl}"))
                for k, fam in enumerate(["ridge", "xgb", "blend", "selected"]):
                    e = p[(p.target == t) & (p.scenario == scen) & (p.scheme == sch) & (p.family == fam)]
                    if e.empty: continue
                    e = e.iloc[0]; off = (k - 1.5) * 0.18
                    lo = max(e.ci_lo, -0.6)
                    ax.hlines(yy + off, lo, e.ci_hi, color=FAM[fam], lw=0.9)
                    ax.plot(max(e.r2, -0.6), yy + off, "o", color=FAM[fam], ms=2.8)
        ax.set_yticks([a for a, _ in yl]); ax.set_yticklabels([b for _, b in yl], fontsize=5.6); ax.invert_yaxis()
        ax.axvline(0, color=style.TEXT_2, lw=0.7); ax.set_xlim(-0.62, 0.9); ax.set_xlabel("out-of-fold R² (log$_{10}$), 90% CI; clipped at −0.6")
        ax.set_title(t, loc="left"); panel(ax, letter)
    fig.legend(handles=[Line2D([], [], color=c, marker="o", lw=1, label={"ridge": "ridge (grouped tuning)", "xgb": "XGBoost (grouped tuning)",
                                                                         "blend": "ridge–XGBoost blend", "selected": "nested selection"}[f]) for f, c in FAM.items()],
               frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.2, wspace=0.05, left=0.2)
    save(fig, "rev_fig5_prediction")


# ------------------------------------------------------------------------------------------ Fig 6: coherence + sentinel
def fig_sentinel():
    coh = pd.read_csv(A / "coherence.csv"); loo = pd.read_csv(A / "coherence_leave_one_campaign.csv")
    r = pd.read_csv(S / "results.csv"); rs = pd.read_csv(S / "random_sets.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 64 * MM), gridspec_kw={"width_ratios": [1.1, 1, 1], "wspace": 0.4})
    ax = axes[0]
    rows = [("all other sites", "leave-self-out reference", "all_others"), ("river sites only", "leave-self-out reference", "river_only"),
            ("plant-weighted", "leave-self-out reference", "unit_weighted"), ("without Apr–Aug 2018", "excluding Apr-Aug 2018 (sensitivity)", "unit_weighted"),
            ("out-of-fold residuals", "out-of-fold chemistry residuals", "unit_weighted")]
    for i, t in enumerate(TARGETS):
        vals = [coh[(coh.target == t) & (coh.analysis == a) & (coh.rule == rule)].median_corr.iloc[0] for _, a, rule in rows]
        y = np.arange(len(rows)) + (i - 0.5) * 0.36
        ax.barh(y, vals, height=0.34, color=TC[t], label=t)
        l = loo[loo.target == t].median_corr
        ax.errorbar(vals[2], y[2], xerr=[[vals[2] - l.min()], [l.max() - vals[2]]], fmt="none", ecolor=style.TEXT, capsize=2, lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([a for a, _, _ in rows], fontsize=6); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("median site correlation with reference"); ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.set_title("Monthly co-variation checks", loc="left"); panel(ax, "a")
    for ax, t, letter in zip(axes[1:], TARGETS, "bc"):
        q = rs[rs.target == t].groupby("k").r2.quantile([0.1, 0.5, 0.9]).unstack()
        ax.fill_between(q.index, q[0.1], q[0.9], color=style.GRID, alpha=0.9, lw=0, label="random sets (10–90%)")
        ax.plot(q.index, q[0.5], color=style.TEXT_2, lw=0.9, ls=":", label="random sets (median)")
        for strat, col, lab in (("forward_greedy", TC[t], "greedy (target)"), ("forward_greedy_joint", GC["reservoir"], "greedy (joint)"),
                                ("forward_fixed", GC["treatment"], "fixed geographic")):
            d = r[(r.experiment == strat) & (r.target == t) & (r.comparator == "L5 chemistry+calibration+sentinel")].sort_values("k")
            if strat != "forward_greedy":
                d = pd.concat([r[(r.experiment == "forward_greedy") & (r.target == t) & (r.k == 0) & (r.comparator == "L5 chemistry+calibration+sentinel")], d])
            ax.plot(d.k, d.r2, marker="o", ms=3, color=col, lw=1.2, label=lab)
        k0 = r[(r.experiment == "forward_greedy") & (r.target == t) & (r.k == 0)].set_index("comparator").r2
        ax.axhline(k0["L1 chemistry"], color=style.TEXT, lw=0.6, ls="--")
        ax.text(8, k0["L1 chemistry"], "chemistry only", fontsize=5.5, ha="right", va="bottom", color=style.TEXT)
        ax.set_xticks([0, 1, 2, 4, 8]); ax.set_xlabel("sentinel sites (frozen, calibration months)"); ax.set_ylabel("R², fixed recipient panel, May–Aug 2018")
        ax.set_title(f"{t}: frozen sentinel test", loc="left"); panel(ax, letter)
        ax.set_ylim(-0.6, 0.7)
    axes[1].legend(frameon=False, fontsize=5.3, loc="lower right")
    save(fig, "rev_fig6_sentinel")


# ------------------------------------------------------------------------------------------ Fig 7: failure + uncertainty
def fig_failure():
    pred = pd.read_csv(ROOT / "results/revision/eval/predictions.csv", parse_dates=["Date"])
    perf = pd.read_csv(A / "performance.csv"); pc = pd.read_csv(A / "per_campaign.csv", parse_dates=["Date"])
    bf = base_frame()
    fig, axes = plt.subplots(1, 3, figsize=(W2, 66 * MM), gridspec_kw={"wspace": 0.55})
    ax = axes[0]
    tmax = bf.loc[bf.Date <= "2018-04-01", "y_THMFP"].max()
    for scen, col in (("S1a", GC["tributary"]), ("S3b", GC["main_stem"])):
        d = pred[(pred.target == "THMFP") & (pred.scenario == scen) & (pred.scheme == "forward")]
        ax.scatter(10 ** d.y, 10 ** d.yhat_ridge, s=6, color=col, alpha=0.7, edgecolors="none", label={"S1a": "S1a water quality", "S3b": "S3 gap-fill"}[scen])
    lim = [0.015, 0.7]; ax.plot(lim, lim, color=style.TEXT_2, lw=0.7)
    ax.axvline(10 ** tmax, color=style.TEXT_2, ls=":", lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mpl.ticker.FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5])); axis.set_minor_formatter(mpl.ticker.NullFormatter())
        axis.set_major_formatter(mpl.ticker.FixedFormatter(["0.02", "0.05", "0.1", "0.2", "0.5"]))
    ax.set_xlabel("observed THMFP (mg/L)"); ax.set_ylabel("predicted (ridge), mg/L"); ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax.set_title("THMFP, May–Aug 2018", loc="left"); panel(ax, "a")
    ax = axes[1]
    for t in TARGETS:
        d = pc[(pc.target == t) & (pc.scenario == "S1a") & (pc.scheme == "month") & (pc.family == "ridge")].sort_values("Date")
        ax.plot(d.Date, d.bias_log, color=TC[t], marker="o", ms=2.5, lw=1.1, label=t)
    ax.axhline(0, color=style.TEXT_2, lw=0.7); ax.axvspan(pd.Timestamp("2018-04-16"), pd.Timestamp("2018-08-16"), color=GC["treatment"], alpha=0.07, lw=0)
    ax.set_ylabel("mean residual, log$_{10}$ (pred − obs)"); ax.tick_params(axis="x", rotation=40)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m")); ax.legend(frameon=False, fontsize=6)
    ax.set_title("S1a bias by held-out campaign", loc="left"); panel(ax, "b")
    ax = axes[2]
    sc = [("month", "new months"), ("site", "new sites"), ("forward", "forward"), ("rolling", "rolling")]
    for i, t in enumerate(TARGETS):
        for j, scen in enumerate(["S1a", "S3b"]):
            v = [perf[(perf.target == t) & (perf.scenario == scen) & (perf.scheme == s) & (perf.family == "selected")].cov90.iloc[0] for s, _ in sc]
            x = np.arange(len(sc)) + (i * 2 + j - 1.5) * 0.18
            ax.bar(x, v, width=0.17, color=TC[t], alpha=1.0 if scen == "S1a" else 0.5, label=f"{t} {scen}")
    ax.axhline(0.9, color=style.TEXT, lw=0.8, ls="--"); ax.set_ylim(0.3, 1.0)
    ax.set_xticks(range(len(sc))); ax.set_xticklabels([b for _, b in sc], rotation=25, ha="right")
    ax.set_ylabel("coverage of nominal 90% interval"); ax.legend(frameon=False, fontsize=5.3, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Interval coverage", loc="left"); panel(ax, "c")
    save(fig, "rev_fig7_failure")


# ------------------------------------------------------------------------------------------ tables
def fmt_ci(r, lo, hi, d=2):
    s = f"{r:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
    return s.replace("-", "$-$")


def table_performance():
    p = pd.read_csv(A / "performance.csv")
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Out-of-fold $R^2$ ($\log_{10}$ scale; 90\% block-bootstrap CI) of the prespecified model families and of nested "
             r"selection. All tuning and selection used inner folds grouped like the outer scheme. Blocks are campaigns (new months, "
             r"forward, rolling origin) or sites (new sites). Coverage refers to nominal 90\% intervals of the selected model.}",
             r"\label{tab:performance}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llllllll}", r"\toprule",
             r"Target & Scenario & Validation & Ridge & XGBoost & Blend & Nested selection & Coverage (90\%) \\", r"\midrule"]
    sl = {"month": "new months", "site": "new sites", "forward": "forward (May--Aug 2018)", "rolling": "rolling origin"}
    scl = {"S1a": "S1a water quality", "S1b": "S1b + site level", "S3b": "S3 network gap-fill"}
    for t in TARGETS:
        for scen in ("S1a", "S1b", "S3b"):
            for sch in ("month", "site", "forward", "rolling"):
                e = p[(p.target == t) & (p.scenario == scen) & (p.scheme == sch)].set_index("family")
                if e.empty: continue
                cells = [fmt_ci(e.loc[f, "r2"], e.loc[f, "ci_lo"], e.loc[f, "ci_hi"]) for f in ("ridge", "xgb", "blend", "selected")]
                lines.append(f"{t} & {scl[scen]} & {sl[sch]} & " + " & ".join(cells) + f" & {e.loc['selected', 'cov90']:.2f} \\\\")
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def table_sentinel():
    r = pd.read_csv(S / "results.csv")
    comps = ["L1 chemistry", "L2 calibration", "L4 chemistry+calibration", "L4c conventional+calibration",
             "L3 calibration+sentinel", "L5 chemistry+calibration+sentinel", "L5c conventional+calibration+sentinel"]
    lab = {"L1 chemistry": "Chemistry (incl.\\ LC-OCD) only", "L2 calibration": "Site calibration level only",
           "L4 chemistry+calibration": "Chemistry + level ($k=0$)", "L4c conventional+calibration": "Conventional chemistry + level ($k=0$)",
           "L3 calibration+sentinel": "Level + sentinel anomaly", "L5 chemistry+calibration+sentinel": "Chemistry + level + sentinel",
           "L5c conventional+calibration+sentinel": "Conventional chemistry + level + sentinel"}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Frozen sentinel protocol, retrospective temporal stress test: $R^2$ ($\log_{10}$) on the fixed recipient panel "
             r"(29 sites never eligible as sentinels) in May--August 2018. Sentinels (target-specific greedy selection), penalty and "
             r"screening rule were fixed on May 2017--April 2018. 90\% CI from site-block bootstrap. Random sets: median of 200 random "
             r"$k$-site networks from the same pool.}", r"\label{tab:sentinel}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lccc}", r"\toprule", r"Comparator & $k$ & THMFP & HAAFP \\", r"\midrule"]
    for comp in comps:
        ks = [0] if not ("sentinel" in comp) else [4, 8]
        for k in ks:
            cells = []
            for t in TARGETS:
                e = r[(r.experiment == "forward_greedy") & (r.target == t) & (r.comparator == comp) & (r.k == k)].iloc[0]
                cells.append(fmt_ci(e.r2, e.ci_lo, e.ci_hi))
            lines.append(f"{lab[comp]} & {k} & " + " & ".join(cells) + r" \\")
    rs = pd.read_csv(S / "random_sets.csv")
    for k in (4, 8):
        cells = [f"{rs[(rs.target == t) & (rs.k == k)].r2.median():.2f}".replace("-", "$-$") for t in TARGETS]
        lines.append(f"Random {k}-site networks (median) & {k} & " + " & ".join(cells) + r" \\")
    b = r[r.experiment == "forward_baseline"].set_index("target").r2
    lines.append("Last calibration-period value & -- & " + " & ".join(f"{b[t]:.2f}".replace("-", "$-$") for t in TARGETS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_treatment():
    t = pd.read_csv(A / "treatment_paired.csv")
    fin = t[t.stage == "finished"]
    vars_ = ["TOC", "HS", "THMFP", "HAAFP", "sp_THMFP", "sp_HAAFP"]
    lab = {"TOC": "TOC", "HS": "HS", "THMFP": "THMFP", "HAAFP": "HAAFP", "sp_THMFP": "THMFP/TOC", "sp_HAAFP": "HAAFP/TOC"}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Reduction of finished relative to raw water (median of paired campaigns, \%, with 90\% CI from campaign "
             r"bootstrap). Carbon-normalised formation potential is FP/TOC in the reported units (TOC, not DOC).}",
             r"\label{tab:treatment}", r"\footnotesize", r"\begin{tabular}{lccc}", r"\toprule", r"Variable & Gumi & Goryeong & Bansong \\", r"\midrule"]
    for v in vars_:
        cells = []
        for p in ("Gumi", "Goryeong", "Bansong"):
            e = fin[(fin.plant == p) & (fin.variable == v)]
            cells.append("--" if e.empty else f"{e.raw_relative_reduction_pct.iloc[0]:.0f} [{e.raw_ci_lo.iloc[0]:.0f}, {e.raw_ci_hi.iloc[0]:.0f}]".replace("-", "$-$"))
        lines.append(f"{lab[v]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def si_tables():
    t = pd.read_csv(A / "treatment_paired.csv")
    lab = {"TOC": "TOC", "HS": "HS", "BB": "BB", "Biopolymer": "Biopolymers", "LMWN": "LMWN", "THMFP": "THMFP", "HAAFP": "HAAFP",
           "sp_THMFP": "THMFP/TOC", "sp_HAAFP": "HAAFP/TOC"}
    m = lambda v: f"{v:.0f}".replace("-", "$-$")
    L = [r"{\footnotesize\begin{longtable}{lllrrrr}",
         r"\caption{Paired treatment changes: raw-relative and adjacent-stage reduction (\%, median over paired campaigns, 90\% CI "
         r"from campaign bootstrap). Negative values are increases. Stage sequences inferred from stage names and site numbering.}"
         r"\label{tab:s_treatment}\\",
         r"\toprule Plant & Stage & Variable & $n$ & Raw-relative & Adjacent & 90\% CI (adjacent) \\ \midrule \endfirsthead",
         r"\toprule Plant & Stage & Variable & $n$ & Raw-relative & Adjacent & 90\% CI (adjacent) \\ \midrule \endhead", r"\bottomrule \endfoot"]
    for _, r in t.iterrows():
        L.append(f"{r.plant} & {r.stage} & {lab[r.variable]} & {r.n_campaigns} & {m(r.raw_relative_reduction_pct)} & {m(r.adjacent_reduction_pct)} & [{m(r.adj_ci_lo)}, {m(r.adj_ci_hi)}] \\\\")
    L.append(r"\end{longtable}}")
    (TAB / "rev_si_treatment.tex").write_text("\n".join(L))

    d = pd.read_csv(A / "tier_diffs.csv")
    f = lambda v: f"{v:+.3f}".replace("-", "$-$")
    sl = {"month": "new campaigns", "site": "new sites", "forward": "forward"}
    L = [r"{\footnotesize\begin{longtable}{llllrl}",
         r"\caption{Paired differences in $R^2$ between measurement tiers (a $-$ b) on a common cohort and identical folds, with 90\% "
         r"block-bootstrap CI.}\label{tab:s_tiers}\\",
         r"\toprule Target & a vs b & Validation & Family & $\Delta R^2$ & 90\% CI \\ \midrule \endfirsthead",
         r"\toprule Target & a vs b & Validation & Family & $\Delta R^2$ & 90\% CI \\ \midrule \endhead", r"\bottomrule \endfoot"]
    for _, r in d.iterrows():
        L.append(f"{r.target} & {r.a} vs {r.b} & {sl[r.scheme]} & {'ridge' if r.family == 'ridge' else 'XGBoost'} & {f(r.delta_r2)} & [{f(r.ci_lo)}, {f(r.ci_hi)}] \\\\")
    L.append(r"\end{longtable}}")
    (TAB / "rev_si_tiers.tex").write_text("\n".join(L))

    r = pd.read_csv(S / "results.csv")
    rb = r[r.experiment.str.startswith(("robust", "outage", "bias"))]
    g = rb.groupby(["experiment", "target", "screen", "agg"]).agg(r2=("r2", "mean"), sd=("r2", "std"), n=("r2", "size"),
                                                                   rej=("n_rejected", "mean"), reads=("n_sent_readings", "mean")).reset_index()
    exl = {"robust_screen": "as observed", "outage_one_missing": "one sentinel missing per campaign (50 trials)",
           "bias_one_x2": "one reading biased $\\times 2$ per campaign (50 trials)", "outage_lone_sentinel": "lone sentinel missing ($k = 1$)"}
    L = [r"\begin{table}[h]", r"\centering", r"\caption{Robustness of the frozen sentinel protocol (target-specific greedy network, $k = 4$ unless stated; "
         r"May--August 2018, fixed recipient panel). Screen: reject readings beyond four robust standard deviations.}", r"\label{tab:s_robust}",
         r"\footnotesize", r"\begin{tabular}{llllrr}", r"\toprule Condition & Target & Screen & Aggregation & $R^2$ (sd) & Rejected / readings \\", r"\midrule"]
    for _, x in g.iterrows():
        sd = "" if pd.isna(x.sd) else f" ({x.sd:.2f})"
        L.append(f"{exl[x.experiment]} & {x.target} & {x.screen} & {x["agg"]} & {x.r2:.2f}{sd} & {x.rej:.1f} / {x.reads:.0f} \\\\".replace("-0.", "$-$0."))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "rev_si_robust.tex").write_text("\n".join(L))

    st = pd.read_csv(A / "strata.csv"); de = pd.read_csv(A / "decisions.csv")
    st = st[(st.family == "ridge") & (st.scenario.isin(["S1a", "S3b"]))]
    L = [r"\begin{table}[h]", r"\centering", r"\caption{Error by concentration stratum (ridge) and screening at the operational threshold "
         r"(calibration-period 75th percentile). High FP: above the calibration-period 90th percentile.}", r"\label{tab:s_decisions}",
         r"\footnotesize", r"\begin{tabular}{lllllrrr}", r"\toprule Target & Scenario & Validation & Stratum & & $n$ & Bias ($\log_{10}$) & Coverage 90\% \\", r"\midrule"]
    for _, x in st.iterrows():
        L.append(f"{x.target} & {x.scenario} & {x.scheme} & {x.stratum} & & {x.n} & {x.bias_log:+.2f} & {x.cov90:.2f} \\\\".replace("-0.", "$-$0."))
    L += [r"\midrule", r"Target & Scenario & Validation & Rule & & Sensitivity & Precision & Review rate \\", r"\midrule"]
    dd = de[(de.decision == "screening") & (de.family == "ridge") & (de.scenario.isin(["S1a", "S3b"]))]
    for _, x in dd.iterrows():
        L.append(f"{x.target} & {x.scenario} & {x.scheme} & {x.rule.replace("%", chr(92) + "%")} & & {x.sensitivity:.2f} & {x.precision:.2f} & {x.review_rate:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "rev_si_decisions.tex").write_text("\n".join(L))
    cfg = pd.read_csv(ROOT / "results/revision/eval/xgb_configs.csv")
    L = [r"\begin{table}[h]", r"\centering", r"\caption{Prespecified XGBoost configurations (drawn once, seed 0, from the search space) and ridge penalties "
         r"used as candidates in every inner loop.}", r"\label{tab:s_configs}", r"\footnotesize", r"\begin{tabular}{lrrrrrrr}",
         r"\toprule Config & Trees & Learning rate & Max depth & Subsample & Col.\\ subsample & Min.\\ child weight & L2 \\", r"\midrule"]
    for _, x in cfg.iterrows():
        L.append(f"XGB {int(x.config)} & {int(x.n_estimators)} & {x.learning_rate:.3f} & {int(x.max_depth)} & {x.subsample:.2f} & {x.colsample_bytree:.2f} & {int(x.min_child_weight)} & {x.reg_lambda:.2f} \\\\")
    L += [r"\midrule", r"\multicolumn{8}{l}{Ridge penalties $\lambda \in \{0.3, 3, 30, 300\}$ on standardized inputs; blends: all 16 ridge--XGBoost pairs.} \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "rev_si_configs.tex").write_text("\n".join(L))


# ------------------------------------------------------------------------------------------ input variables
PAPER_INPUTS = [c for c in PREDICTORS if c not in ("Aromaticity", "MolWeight")]   # dropped from the paper (2026-10-09)


def fig_inputs():
    """Distributions of the 16 reported inputs and their Spearman correlations in river samples.
    NH3-N non-detects enter at LOD/2 (preprocessing/load.py)."""
    import seaborn as sns
    from matplotlib.ticker import NullFormatter
    df = load_clean(); df = df[df.n_measured > 2]
    skewed = [c for c in PAPER_INPUTS if df[c].dropna().skew() > 2]
    fig, axes = plt.subplots(4, 4, figsize=(W2, W2 * 0.8)); axes = axes.ravel()
    for ax, c in zip(axes, PAPER_INPUTS):
        s = df[c].dropna()
        if c in skewed:
            s = s[s > 0]; bins = np.logspace(np.log10(s.min()), np.log10(s.max()), 30); ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            bins = 30
        ax.hist(s, bins=bins, color=GC["main_stem"], edgecolor=style.SURFACE, linewidth=0.3)
        ax.set_title(f"{LABEL.get(c, c)}  (n = {len(s)})", fontsize=6.5)
        ax.set_xlabel(UNITS[c] if UNITS[c] != "-" else "", fontsize=6); ax.tick_params(labelsize=5.5); ax.grid(False)
    fig.tight_layout(); save(fig, "rev_fig_inputs")

    river = df[df.group.astype(str) != "treatment"]
    pc = river[PAPER_INPUTS].corr(method="spearman").rename(index=LABEL, columns=LABEL)
    g = sns.clustermap(pc, cmap=style.DIV_CMAP, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 5.5},
                       figsize=(W2, W2), linewidths=0.6, linecolor=style.SURFACE, dendrogram_ratio=0.08,
                       cbar_pos=(0.02, 0.85, 0.02, 0.12))
    g.ax_heatmap.tick_params(labelsize=6.5)
    g.savefig(FIG / "rev_fig_input_corr.pdf"); g.savefig(FIG / "rev_fig_input_corr.png", dpi=300); plt.close(g.fig)
    print("saved rev_fig_input_corr")


LABEL = {"Temp": "Temp", "Turbidity": "Turbidity", "Br": "Br⁻", "NH3N": "NH₃-N", "Biopolymer": "Biopolymers",
         "HS": "HS", "BB": "BB", "LMWN": "LMWN"}


def main():
    setup()
    if "--inputs" in sys.argv:
        fig_inputs(); return
    fig_inputs()
    fig_variation(); fig_precursors(); fig_treatment(); fig_prediction(); fig_sentinel(); fig_failure()
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "rev_performance.tex").write_text(table_performance())
    (TAB / "rev_sentinel.tex").write_text(table_sentinel())
    (TAB / "rev_treatment.tex").write_text(table_treatment())
    si_tables()
    print("tables written")


if __name__ == "__main__":
    main()
