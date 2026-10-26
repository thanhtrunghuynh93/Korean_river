"""Manuscript figures (journal size, vector PDF + 300 dpi PNG) -> Paper_templates/Template_paper/assets/figs/.

    uv run python -m src.paper_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from preprocessing import style
from preprocessing.load import GROUP_LABEL, GROUP_ORDER, SITE_ORDER, STAGE_ORDER, load_clean

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
P = R / "paper"
OUT = ROOT / "Paper_templates" / "Template_paper" / "assets" / "figs"
MM = 1 / 25.4
W2, W1 = 190 * MM, 90 * MM                     # double / single column widths
GC = style.GROUP_COLORS
TC = style.TARGET_COLORS
TARGETS = ["THMFP", "HAAFP"]


def setup() -> None:
    style.apply()
    mpl.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "lines.linewidth": 1.4, "lines.markersize": 4,
                         "axes.titleweight": "bold", "savefig.dpi": 300, "pdf.fonttype": 42})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", name)


def panel(ax, letter: str) -> None:
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


# ------------------------------------------------------------------------------------------------ Fig. 1
def fig1_network() -> None:
    """Schematic of the sampling network (not to scale; confluence positions approximate)."""
    fig, ax = plt.subplots(figsize=(W2, 92 * MM))
    ax.set_xlim(0.2, 19.4); ax.set_ylim(-4.3, 3.2); ax.axis("off")
    ax.plot([1, 18], [0, 0], color=GC["main_stem"], lw=2.2, solid_capstyle="round", zorder=1)
    ax.annotate("", xy=(18.9, 0), xytext=(18.2, 0), arrowprops=dict(arrowstyle="-|>", color=GC["main_stem"], lw=1.4))
    ax.text(18.95, 0.18, "Sea", fontsize=6.5, ha="left", color=style.TEXT_2)
    names = ["Andong Res.", "Imha Res.", "Yecheon", "Sangju", "Nakdan Weir", "Haepyeong", "Gumi", "Waegwan", "Goryeong raw",
             "Dalseong", "Goryeong", "Yongsan", "Namji", "Bansong raw", "Samnangjin", "Mulgeum", "Gupo", "Estuary Barrage"]
    for i, n in enumerate(names, start=1):
        ax.scatter(i, 0, s=26, color=GC["main_stem"], edgecolors=style.SURFACE, linewidths=0.8, zorder=3)
        ax.text(i, -0.28, f"M{i}", ha="center", va="top", fontsize=6, fontweight="bold", color=style.TEXT)
        ax.text(i + 0.05, -0.6, n, ha="right", va="top", fontsize=5.3, rotation=40, rotation_mode="anchor", color=style.TEXT_2)

    def trib(xj, top, label):
        ax.plot([xj, top[0]], [0, top[1]], color=GC["tributary"], lw=1.3, zorder=1)
        ax.scatter(*top, s=22, color=GC["tributary"], edgecolors=style.SURFACE, linewidths=0.8, zorder=3)
        ax.text(top[0], top[1] + 0.2, label, ha="center", va="bottom", fontsize=5.9, color=style.TEXT)

    def res(frm, at, label, ha):
        ax.plot([frm[0], at[0]], [frm[1], at[1]], color=GC["reservoir"], lw=0.9, zorder=1)
        ax.scatter(*at, s=30, marker="s", color=GC["reservoir"], edgecolors=style.SURFACE, linewidths=0.8, zorder=3)
        dx = -0.18 if ha == "right" else 0.18
        ax.text(at[0] + dx, at[1], label, ha=ha, va="center", fontsize=5.7, color=style.TEXT)

    trib(4.5, (4.0, 1.25), "T1 Wicheon")
    trib(6.5, (6.0, 1.25), "T2 Gamcheon")
    trib(9.5, (9.0, 1.25), "T3 Geumho R.")
    res((9.0, 1.25), (8.2, 2.05), "2 Unmun Res.", "right")
    res((9.0, 1.25), (9.1, 2.55), "4 Yeongcheon Res.", "right")
    trib(11.3, (11.0, 1.0), "T4 Hoecheon")
    trib(12.5, (12.2, 1.9), "T5 Hwang R.")
    res((12.2, 1.9), (12.8, 2.6), "5 Hapcheon Res.", "left")
    ax.plot([13.3, 13.7], [0, 1.0], color=GC["tributary"], lw=1.0, ls=(0, (2, 1.2)), zorder=1)
    res((13.7, 1.0), (13.7, 1.0), "3 Namgang Res.", "left")
    trib(15.3, (15.8, 1.6), "T6 Miryang R.")
    res((15.8, 1.6), (16.5, 2.5), "1 Miryang Res.", "left")

    # treatment trains in their own band, linked to their intakes
    trains = [(7, 3.4, "Gumi WTP", ["6 raw", "7 settled", "8 filtered", "9 O$_3$", "10 GAC", "11 finished"]),
              (9, 8.4, "Goryeong WTP (raw = M9)", ["12 settled", "13 filtered", "14 O$_3$", "15 GAC", "16 finished"]),
              (14, 13.6, "Bansong WTP (raw = M14)", ["18 settled", "17 O$_3$", "19 F/A", "20 finished"])]
    y0 = -2.75
    for xin, xs, lab, stages in trains:
        xe = xs + (len(stages) - 1) * 0.62
        ax.plot([xin, (xs + xe) / 2], [-0.35, y0 + 0.42], color=GC["treatment"], lw=0.8, ls=(0, (2, 1.5)), zorder=0)
        ax.text((xs + xe) / 2, y0 + 0.47, lab, fontsize=5.9, ha="center", va="bottom", color=style.TEXT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc=style.SURFACE, ec="none"), zorder=4)
        ax.plot([xs, xe], [y0, y0], color=GC["treatment"], lw=0.8, zorder=1)
        for k, st in enumerate(stages):
            xx = xs + k * 0.62
            ax.scatter(xx, y0, s=18, marker="D", color=GC["treatment"], edgecolors=style.SURFACE, linewidths=0.6, zorder=3)
            ax.text(xx, y0 - 0.22, st, fontsize=5.1, ha="center", va="top", rotation=90, color=style.TEXT_2)
    handles = [Line2D([], [], color=GC["main_stem"], marker="o", lw=2, label=GROUP_LABEL["main_stem"]),
               Line2D([], [], color=GC["tributary"], marker="o", lw=1.3, label=GROUP_LABEL["tributary"]),
               Line2D([], [], color=GC["reservoir"], marker="s", lw=0, label=GROUP_LABEL["reservoir"]),
               Line2D([], [], color=GC["treatment"], marker="D", lw=0, label=GROUP_LABEL["treatment"])]
    ax.legend(handles=handles, loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.0, 1.02))
    ax.text(19.3, -4.25, "Schematic, not to scale; confluence positions approximate", fontsize=5.5, ha="right",
            color=style.TEXT_2, style="italic")
    save(fig, "fig1_network")


# ------------------------------------------------------------------------------------------------ Fig. 2
def fig2_workflow() -> None:
    fig, ax = plt.subplots(figsize=(W2, 52 * MM))
    ax.set_xlim(0, 101.5); ax.set_ylim(0, 40); ax.axis("off")
    boxes = [
        ("Monitoring data", "44 sites × 16 months\nMay 2017 – Aug 2018\n18 inputs: field, lab,\nLC-OCD fractions\nTargets: THMFP, HAAFP", GC["main_stem"]),
        ("Processing", "Non-detects flagged\nPlaceholders removed\nlog$_{10}$ of skewed inputs\nQA screen (sentinels)", style.TEXT_2),
        ("Scenarios", "S1 single sample\nS2 sentinel network\nS3 network gap-fill", GC["tributary"]),
        ("Models", "Ridge, XGBoost, HGB\nspline, monotone,\nspecific-FP, blend,\nrandom intercept, chained\n+ 3 baselines", GC["reservoir"]),
        ("Evaluation", "Leave-months-out\nLeave-sites-out\nForward holdout\nSelection in outer folds\nBootstrap CI, conformal PI", GC["treatment"]),
    ]
    w, gap = 17.6, 2.4
    for i, (title, body, col) in enumerate(boxes):
        x = 1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 3), w, 32, boxstyle="round,pad=0.4,rounding_size=1.2", fc=style.SURFACE, ec=col, lw=1.2))
        ax.add_patch(FancyBboxPatch((x, 28.5), w, 6.5, boxstyle="round,pad=0.4,rounding_size=1.2", fc=col, ec=col, lw=1.2, alpha=0.15))
        ax.text(x + w / 2, 31.7, title, ha="center", va="center", fontsize=7, fontweight="bold", color=style.TEXT)
        ax.text(x + w / 2, 15.5, body, ha="center", va="center", fontsize=5.6, color=style.TEXT, linespacing=1.5)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.3, 19), xytext=(x + w + 0.5, 19),
                        arrowprops=dict(arrowstyle="-|>", color=style.TEXT_2, lw=1))
    save(fig, "fig2_workflow")


# ------------------------------------------------------------------------------------------------ Fig. 3
def fig3_spatiotemporal(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(W2, 150 * MM))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.95], hspace=0.62, wspace=0.18)
    river_sites = [s for s in SITE_ORDER if s in set(df.Site.astype(str)) and df.loc[df.Site.astype(str) == s, "group"].astype(str).iloc[0] != "treatment"]
    for r, t in enumerate(TARGETS):
        ax = fig.add_subplot(gs[r, :])
        data = [np.log10(df.loc[df.Site.astype(str) == s, t].dropna()) for s in river_sites]
        bp = ax.boxplot(data, positions=range(len(river_sites)), widths=0.6, patch_artist=True, showfliers=True,
                        flierprops={"markersize": 1.8, "markeredgecolor": style.TEXT_2}, medianprops={"color": style.TEXT, "lw": 1},
                        whiskerprops={"lw": 0.7}, capprops={"lw": 0.7}, boxprops={"lw": 0.4})
        for patch, s in zip(bp["boxes"], river_sites):
            patch.set_facecolor(GC[df.loc[df.Site.astype(str) == s, "group"].astype(str).iloc[0]]); patch.set_edgecolor(style.SURFACE)
        ax.set_xticks(range(len(river_sites))); ax.set_xticklabels(river_sites, fontsize=6)
        ax.set_xlim(-0.7, len(river_sites) - 0.3)
        ax.set_ylabel(f"log$_{{10}}$ {t} (mg/L)")
        ax.set_title(f"{t} by site: main stem upstream → estuary, then tributaries and reservoirs", loc="left")
        panel(ax, "ab"[r])
        if r == 0:
            ax.legend(handles=[Patch(color=GC[g], label=GROUP_LABEL[g]) for g in GROUP_ORDER[:3]], loc="lower right", ncol=3, frameon=False)
    monthly = df.groupby(["Date", "group"], observed=True)[TARGETS].median().reset_index()
    for c, t in enumerate(TARGETS):
        ax = fig.add_subplot(gs[2, c])
        for g in GROUP_ORDER:
            m = monthly[monthly.group.astype(str) == g]
            ax.plot(m.Date, m[t], color=GC[g], marker="o", markersize=2.5, lw=1.2, label=GROUP_LABEL[g])
        ax.set_ylabel(f"monthly median {t} (mg/L)"); ax.tick_params(axis="x", rotation=40)
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m"))
        ax.set_title(f"{t} by month and site group", loc="left"); panel(ax, "cd"[c])
        if c == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=5.8)
    save(fig, "fig3_spatiotemporal")


# ------------------------------------------------------------------------------------------------ Fig. 4
def fig4_treatment(df: pd.DataFrame) -> None:
    chg = pd.read_csv(P / "treatment_changes.csv")
    train = df[df.wtp.notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(W2, 64 * MM), gridspec_kw={"wspace": 0.42})
    for ax, v, letter in zip(axes, ["TOC", "THMFP", "HAAFP"], "abc"):
        for w, c in style.WTP_COLORS.items():
            g = train[train.wtp == w].groupby("stage", observed=True)[v]
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            x = [STAGE_ORDER.index(s) for s in med.index]
            ax.fill_between(x, q1.values, q3.values, color=c, alpha=0.10, lw=0)
            fin = chg[(chg.wtp == w) & (chg.variable == v) & (chg.stage == "finished")]
            lab = f"{w} ({fin.median_pct_change.iloc[0]:+.0f} %)" if len(fin) else w
            ax.plot(x, med.values, color=c, marker="o", markersize=3, label=lab)
        ax.set_xticks(range(len(STAGE_ORDER))); ax.set_xticklabels(STAGE_ORDER, rotation=35, ha="right")
        ax.set_xlim(-0.3, len(STAGE_ORDER) - 0.2)
        ax.set_ylabel(f"{v} ({'mg C/L' if v == 'TOC' else 'mg/L'})"); ax.set_title(f"{v} through treatment", loc="left"); panel(ax, letter)
        ax.legend(frameon=False, loc="upper right", fontsize=5.6, title="finished vs raw", title_fontsize=5.6)
    save(fig, "fig4_treatment")


# ------------------------------------------------------------------------------------------------ Fig. 5
def fig5_drivers() -> None:
    imp = pd.read_csv(P / "importance_consensus.csv")
    agree = pd.read_csv(P / "importance_agreement.csv").set_index("target").kendall_w
    exp_tab = pd.read_csv(R / "experiments.csv", dtype={"tier": str})
    fig = plt.figure(figsize=(W2, 72 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.55)
    # (a) Spearman rho of the core inputs with each target
    ax = fig.add_subplot(gs[0])
    th = imp[imp.target == "THMFP"].set_index("feature")
    order = th.sort_values("spearman_abs_rho").index
    y = np.arange(len(order))
    for k, t in enumerate(TARGETS):
        v = imp[imp.target == t].set_index("feature").spearman_abs_rho.reindex(order)
        ax.barh(y + (k - 0.5) * 0.38, v, height=0.36, color=TC[t], label=t)
    ax.set_yticks(y); ax.set_yticklabels([o.replace("log_", "") for o in order]); ax.set_xlabel("|Spearman ρ| with target")
    ax.set_title("Univariate association", loc="left"); ax.legend(frameon=False, loc="lower right"); panel(ax, "a")
    # (b, c) consensus ranks
    methods = ["spearman_abs_rho_rank", "xgb_gain_rank", "permutation_drmse_rank", "mean_abs_shap_rank"]
    mlab = ["Spearman", "XGB gain", "Permutation", "SHAP"]
    for j, t in enumerate(TARGETS):
        ax = fig.add_subplot(gs[j + 1])
        d = imp[imp.target == t].sort_values("mean_rank").set_index("feature")
        mat = d[methods].to_numpy()
        ax.imshow(mat, cmap=style.SEQ_CMAP.reversed(), vmin=1, vmax=10, aspect="auto")
        for (r, c), v in np.ndenumerate(mat):
            ax.text(c, r, f"{int(v)}", ha="center", va="center", fontsize=6, color="white" if v <= 4 else style.TEXT)
        ax.set_xticks(range(4)); ax.set_xticklabels(mlab, rotation=35, ha="right")
        ax.set_yticks(range(len(d))); ax.set_yticklabels([f.replace("log_", "") for f in d.index]); ax.grid(False)
        ax.set_title(f"{t} ranks (W = {agree[t]:.2f})", loc="left"); panel(ax, "bc"[j])
    save(fig, "fig5_drivers")


# ------------------------------------------------------------------------------------------------ Fig. 6
def fig6_scenarios() -> None:
    head = pd.read_csv(R / "headline.csv")
    curve = pd.read_csv(P / "sentinel_curve.csv")
    rows = []
    for t in TARGETS:
        for scen, lab in [("S1 single sample", "S1 single sample"), ("S3 network gap-fill", "S3 network gap-fill")]:
            for s in ["month", "site", "forward"]:
                h = head[(head.target == t) & (head.scenario == scen) & (head.scheme == s)]
                if len(h):
                    rows.append({"target": t, "label": lab, "scheme": s, "r2": h.r2_selected.iloc[0], "lo": h.ci_lo.iloc[0], "hi": h.ci_hi.iloc[0]})
        for k in (1, 4):
            c = curve[(curve.target == t) & (curve.k == k)].iloc[0]
            rows.append({"target": t, "label": f"S2 sentinel, k = {k}", "scheme": "month", "r2": c.r2, "lo": c.ci_lo, "hi": c.ci_hi})
    d = pd.DataFrame(rows)
    labels = ["S1 single sample", "S2 sentinel, k = 1", "S2 sentinel, k = 4", "S3 network gap-fill"]
    scol = {"month": GC["main_stem"], "site": GC["tributary"], "forward": GC["treatment"]}
    fig, axes = plt.subplots(1, 2, figsize=(W2, 64 * MM), sharey=True)
    for ax, t, letter in zip(axes, TARGETS, "ab"):
        for i, lab in enumerate(labels[::-1]):
            for j, s in enumerate(["month", "site", "forward"]):
                e = d[(d.target == t) & (d.label == lab) & (d.scheme == s)]
                if e.empty: continue
                yy = i + (j - 1) * 0.22
                ax.hlines(yy, max(e.lo.iloc[0], -0.4), e.hi.iloc[0], color=scol[s], lw=1.4)
                ax.scatter(e.r2, yy, color=scol[s], s=16, zorder=3, edgecolors=style.SURFACE, linewidths=0.6)
                if e.lo.iloc[0] < -0.4:
                    ax.annotate("", xy=(-0.42, yy), xytext=(-0.3, yy), arrowprops=dict(arrowstyle="-|>", color=scol[s], lw=0.8))
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels[::-1]); ax.set_xlim(-0.45, 0.9)
        ax.axvline(0, color=style.TEXT_2, lw=0.7); ax.set_xlabel("selection-adjusted R² (log$_{10}$), 90 % CI")
        ax.set_title(t, loc="left"); panel(ax, letter)
    fig.legend(handles=[Line2D([], [], color=c, marker="o", lw=1.4, label=s) for s, c in
                        {"new months (leave-months-out)": scol["month"], "new sites (leave-sites-out)": scol["site"],
                         "forward holdout (May–Aug 2018)": scol["forward"]}.items()],
               frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.27, wspace=0.08)
    save(fig, "fig6_scenarios")


# ------------------------------------------------------------------------------------------------ Fig. 7
def fig7_covariation() -> None:
    dist = pd.read_csv(P / "covariation_distance.csv")
    ser = pd.read_csv(P / "covariation_series.csv", parse_dates=["Date"])
    curve = pd.read_csv(P / "sentinel_curve.csv")
    head = pd.read_csv(R / "headline.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 58 * MM))
    ax = axes[0]
    for t in TARGETS:
        d = dist[(dist.target == t) & (dist.separation <= 14)]
        ax.plot(d.separation, d.median_corr, color=TC[t], marker="o", markersize=2.5, label=t)
    ax.set_xlabel("separation along main stem (sites)"); ax.set_ylabel("median correlation of monthly anomalies")
    ax.set_ylim(0, 1); ax.set_xticks([1, 3, 5, 7, 9, 11, 13]); ax.set_title("Co-variation vs distance", loc="left"); ax.legend(frameon=False); panel(ax, "a")
    ax = axes[1]
    for t in TARGETS:
        s = ser[ser.target == t]
        ax.plot(s.Date, 10 ** s.common_signal, color=TC[t], marker="o", markersize=2.5, label=t)
    ax.axhline(1, color=style.TEXT_2, lw=0.7)
    ax.set_ylabel("basin-wide monthly factor (× site normal)"); ax.tick_params(axis="x", rotation=40)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m")); ax.set_title("Shared monthly signal", loc="left"); panel(ax, "b")
    ax = axes[2]
    for t in TARGETS:
        c = curve[curve.target == t].sort_values("k")
        ax.fill_between(c.k, c.ci_lo, c.ci_hi, color=TC[t], alpha=0.10, lw=0)
        ax.plot(c.k, c.r2, color=TC[t], marker="o", markersize=2.8, label=t)
        s3 = head[(head.target == t) & (head.scenario == "S3 network gap-fill") & (head.scheme == "month")].r2_selected.iloc[0]
        ax.axhline(s3, color=TC[t], lw=0.8, ls="--")
    ax.set_xlabel("sentinel sites lab-tested per month (of 44)"); ax.set_ylabel("R², non-sentinel sites, new months")
    ax.set_ylim(0.3, 0.85); ax.set_xticks(sorted(curve.k.unique()))
    ax.set_title("Sentinel design (dashed: S3 gap-fill)", loc="left"); ax.legend(frameon=False, loc="lower right"); panel(ax, "c")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig7_covariation_sentinel")


# ------------------------------------------------------------------------------------------------ Fig. 8
def fig8_extrapolation() -> None:
    p = pd.read_csv(R / "predictions.csv", dtype={"tier": str})
    p["Date"] = pd.to_datetime(p.Date)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 62 * MM), gridspec_kw={"width_ratios": [1.5, 1]})
    mcol = {"ridge": GC["main_stem"], "xgb": GC["tributary"]}
    ax = axes[0]
    for m in ["ridge", "xgb"]:
        d = p[(p.target == "THMFP") & (p.tier == "t3_basin") & (p.model == m) & (p.scheme == "month") & (p.group == "main_stem")]
        g = d.groupby("Date")[["y", "y_pred"]].median()
        if m == "ridge":
            ax.plot(g.index, 10 ** g.y, color=style.TEXT, marker="o", markersize=2.8, lw=1.3, label="observed")
        ax.plot(g.index, 10 ** g.y_pred, color=mcol[m], lw=1.3, label=f"{'ridge' if m == 'ridge' else 'XGBoost'} (held-out month)")
    ax.axvspan(pd.Timestamp("2018-04-16"), pd.Timestamp("2018-08-16"), color=GC["treatment"], alpha=0.07, lw=0)
    ax.text(pd.Timestamp("2018-06-15"), ax.get_ylim()[1] * 0.97, "anomalous\nsummer", ha="center", va="top", fontsize=5.8, color=style.TEXT_2)
    ax.set_ylabel("main-stem median THMFP (mg/L)"); ax.tick_params(axis="x", rotation=40)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m")); ax.legend(frameon=False, loc="upper left")
    ax.set_title("THMFP, S3 gap-fill, leave-months-out", loc="left"); panel(ax, "a")
    ax = axes[1]
    train_max = p[(p.target == "THMFP") & (p.scheme == "month") & (p.tier == "t3_basin") & (p.model == "ridge") & (p.Date < "2018-05-01")].y.max()
    for m in ["ridge", "xgb"]:
        d = p[(p.target == "THMFP") & (p.tier == "t3_basin") & (p.model == m) & (p.scheme == "forward")]
        ax.scatter(10 ** d.y, 10 ** d.y_pred, s=8, color=mcol[m], alpha=0.7, edgecolors=style.SURFACE, linewidths=0.4,
                   label="ridge" if m == "ridge" else "XGBoost")
    lim = [0.015, 0.7]; ax.plot(lim, lim, color=style.TEXT_2, lw=0.7)
    ax.axvline(10 ** train_max, color=style.TEXT_2, lw=0.7, ls=":"); ax.axhline(10 ** train_max, color=style.TEXT_2, lw=0.7, ls=":")
    ax.text(10 ** train_max * 1.05, 0.02, "training max", fontsize=5.6, color=style.TEXT_2, rotation=90, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ticks = [0.02, 0.05, 0.1, 0.2, 0.5]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mpl.ticker.FixedLocator(ticks)); axis.set_minor_formatter(mpl.ticker.NullFormatter())
        axis.set_major_formatter(mpl.ticker.FixedFormatter([str(t) for t in ticks]))
    ax.set_xlabel("observed THMFP (mg/L)"); ax.set_ylabel("predicted THMFP (mg/L)")
    ax.set_title("Forward holdout, May–Aug 2018", loc="left"); ax.legend(frameon=False, loc="upper left"); panel(ax, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig8_extrapolation")


def main() -> None:
    setup()
    df = load_clean(); df = df[df.n_measured > 2]
    fig1_network(); fig2_workflow(); fig3_spatiotemporal(df); fig4_treatment(df)
    fig5_drivers(); fig6_scenarios(); fig7_covariation(); fig8_extrapolation()


if __name__ == "__main__":
    main()
