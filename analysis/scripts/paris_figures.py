"""Presentation figures for the Paris talk (v2).

    uv run python analysis/scripts/paris_figures.py

Reads output/plots/rho_sweep_all/sweep.csv, output/plots/all_sizes_summary/summary_by_size.csv and the
rho-sweep reports (for the mission timeline); writes output/plots/paris_slides/.
"""

import os
from pathlib import Path
import sys
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd


REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis", "scripts"))
os.chdir(REPO)

OUT = os.path.join(REPO, "output/plots/paris_slides")
os.makedirs(OUT, exist_ok=True)

# deck colours
NAVY = "#1E2761"
INK = "#10193A"
MUTED = "#5B6378"
GRID = "#E3E8F0"
ORANGE = "#EB6834"  # LLS-Budget (ours)
BLUE = "#2A78D6"  # LLS
AQUA = "#1BAF7A"  # FCP
VIOLET = "#4A3AA7"  # Lifespan-Aware v1
TINT_O = "#FDEDE6"
TINT_B = "#F2F6FB"

plt.rcParams.update(
    {
        "font.family": "Carlito",
        "font.size": 20,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "axes.axisbelow": True,
        "lines.linewidth": 3,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

sweep = pd.read_csv("output/plots/rho_sweep_all/sweep.csv")
summ = pd.read_csv("output/plots/all_sizes_summary/summary_by_size.csv")
S20 = "mars_relay_earth_scenario_inc_20"


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- F1: rho axis (diagnosis)
def fig_rho_axis():
    fig, ax = plt.subplots(figsize=(16, 4.6))
    ax.set_xscale("log")
    lo, hi = 0.01, 25
    rs, rc = 0.18, 0.89
    ax.axvspan(lo, rs, color=TINT_O, lw=0)
    ax.axvspan(rs, rc, color="#FFD3C0", lw=0)
    ax.axvspan(rc, hi, color=TINT_B, lw=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.set_xticks([0.01, 0.025, 0.1, 0.18, 0.5, 0.89, 2, 12.5])
    ax.set_xticklabels(
        ["0.01", "0.025", "0.1", "0.18", "0.5", "0.89", "2", "12.5"]
    )
    ax.set_xlabel(
        r"$\rho$ = (P$_0$ − P$_{base}$) / P$_{tx}$   (largest duty cycle a node can sustain)"
    )
    kw: dict[str, Any] = dict(ha="center", va="top", color=INK, fontsize=19)
    ax.text(np.sqrt(lo * rs), 0.95, "energy limits data", **kw)
    ax.text(
        np.sqrt(rs * rc),
        0.95,
        "budget saves energy,\nno data lost",
        fontweight="bold",
        **kw,
    )
    ax.text(np.sqrt(rc * hi), 0.95, "energy does not matter", **kw)
    # markers
    y = 0.32
    ax.plot([0.025], [y], marker="D", ms=16, color=NAVY, clip_on=False)
    ax.plot([12.5], [y], marker="D", ms=16, color=NAVY, clip_on=False)
    ax.plot([0.5], [y], marker="o", ms=20, color=ORANGE, mec="white", mew=2.5)
    lk: dict[str, Any] = dict(ha="center", va="bottom", fontsize=17, color=INK)
    ax.text(0.025, y + 0.09, "WiSEE EOL\n0.1 W / 4 W", **lk)
    ax.text(12.5, y + 0.09, "WiSEE BOL\n50 W / 4 W", **lk)
    lk_bold: dict[str, Any] = {**lk, "color": ORANGE, "fontweight": "bold"}
    ax.text(
        0.5,
        y + 0.09,
        "this talk\n20 W / 40 W",
        **lk_bold,
    )
    sk: dict[str, Any] = dict(ha="center", va="top", fontsize=15, color=MUTED)
    ax.text(0.025, y - 0.08, "every source dies\nin ~1.5 days", **sk)
    ax.text(12.5, y - 0.08, "no source dies\nfor ~14 years", **sk)
    ax.text(
        rs * 0.95,
        0.02,
        r"$\rho_{sat}$ (20 sources)",
        ha="right",
        va="bottom",
        fontsize=14,
        color=MUTED,
    )
    ax.text(
        rc * 1.06,
        0.02,
        "Earth visible 89% of the day",
        ha="left",
        va="bottom",
        fontsize=14,
        color=MUTED,
    )
    save(fig, "f1_rho_axis")


# ---------------------------------------------------------------- F2: capacity vs rho (R1)
def fig_capacity_vs_rho():
    s = sweep[sweep.scenario == S20]
    b = s[s.algorithm == "lls_budget"].sort_values("rho")
    b = b[b.rho <= 1.0]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axvspan(0.184, 0.886, color=TINT_O, lw=0)
    ax.plot(
        b.rho,
        b.capacity_tbit_day,
        "-o",
        color=ORANGE,
        ms=9,
        mec="white",
        mew=1.5,
        zorder=4,
    )
    ax.text(0.30, 70, "LLS-Budget", color=INK, fontsize=21, fontweight="bold")
    pts = [
        ("lls", "LLS", BLUE),
        ("fcp", "FCP", AQUA),
        ("lifespan_aware", "Lifespan-Aware (WiSEE)", VIOLET),
    ]
    for alg, label, col in pts:
        r = s[s.algorithm == alg].iloc[0]
        x, yv = r.max_source_duty, r.capacity_tbit_day
        ax.plot([x], [yv], "D", ms=14, color=col, mec="white", mew=2, zorder=5)
        if alg == "lls":
            ax.text(
                x - 0.01,
                yv - 5,
                f"{label}\nneeds ρ ≥ {x:.2f}",
                ha="right",
                va="top",
                fontsize=17,
                color=INK,
            )
        else:
            ax.text(
                x - 0.02,
                yv,
                f"{label}\nneeds ρ ≥ {x:.2f}",
                ha="right",
                va="center",
                fontsize=17,
                color=INK,
            )
    ax.axvline(0.184, color=ORANGE, lw=1.5, ls="--")
    ax.text(0.19, 6, r"$\rho_{sat}$ = 0.18", fontsize=17, color=INK)
    ax.annotate(
        "",
        xy=(0.184, 88),
        xytext=(0.886, 88),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=2),
        annotation_clip=False,
    )
    ax.text(
        0.535,
        89.5,
        "79% less spare power for ≥ 95% of the data",
        ha="center",
        va="bottom",
        fontsize=19,
        color=INK,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 95)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel("ρ = spare RTG power / terminal power")
    ax.set_ylabel("data delivered [Tbit/day]")
    save(fig, "f2_capacity_vs_rho_inc20")


# ---------------------------------------------------------------- F3: ground stations (bridge)
def fig_stations():
    n = summ["size"].values
    cap = summ["lls_capacity"].values
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.step(n, cap, where="mid", color=BLUE, lw=3)
    ax.plot(n, cap, "o", color=BLUE, ms=10, mec="white", mew=1.5)
    for start, gs in [(4, 3), (20, 6), (36, 9), (52, 12)]:
        y = cap[n == start][0]
        ax.text(
            start + 6,
            y + 9,
            f"{gs} station terminals",
            ha="center",
            fontsize=18,
            color=INK,
        )
    ax.text(
        5,
        140,
        "≈ 13.2 Tbit/day per station terminal\nadding satellites adds no data",
        fontsize=19,
        color=INK,
        fontweight="bold",
        va="top",
    )
    ax.set_xlabel("source satellites at Mars")
    ax.set_ylabel("data delivered [Tbit/day]")
    ax.set_ylim(0, 175)
    ax.set_xticks([4, 16, 20, 32, 36, 48, 52, 64])
    save(fig, "f3_ground_stations")


# ---------------------------------------------------------------- F4: design map (R2)
def fig_design_map():
    s = sweep[
        (sweep.scenario == S20) & (sweep.algorithm == "lls_budget")
    ].sort_values("rho")
    cmax = s.capacity_tbit_day.max()
    ptx = np.linspace(10, 140, 400)
    m = np.linspace(0.5, 60, 400)
    P, M = np.meshgrid(ptx, m)
    frac = np.interp(
        M / P, s.rho.values, s.capacity_tbit_day.values / cmax, right=1.0
    )
    loss = 100 * (1 - frac)
    fig, ax = plt.subplots(figsize=(13, 7.2))
    levels = [5, 25, 50, 75, 100]
    cols = ["#FDEDE6", "#F9C7B0", "#F39A74", "#EB6834"]
    cs = ax.contourf(P, M, loss, levels=levels, colors=cols)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label("data lost, 20 sources [%]", color=INK)
    cb.outline.set_visible(False)  # ty: ignore[call-non-callable]
    for size, col, lw in [(8, MUTED, 2), (20, NAVY, 4), (64, MUTED, 2)]:
        rs = float(summ.loc[summ["size"] == size, "rho_sat_95"].iloc[0])
        y = rs * ptx
        ok = y <= 60
        ax.plot(ptx[ok], y[ok], color=col, lw=lw)
        xe = min(138, 59 / rs)
        ax.text(
            xe,
            rs * xe + 1.2,
            f"{size} sources",
            ha="right",
            va="bottom",
            fontsize=16,
            color=INK,
            fontweight="bold" if size == 20 else "normal",
        )
    for x, lab in [
        (61, "DSOC terminal\n61 W"),
        (110, "DOT allocation\n110 W"),
    ]:
        ax.axvline(x, color=MUTED, ls=":", lw=1.8)
        ax.text(x + 1, 58.5, lab, fontsize=14, color=MUTED, va="top")
    ax.plot([40], [20], "o", ms=18, color=ORANGE, mec="white", mew=3, zorder=6)
    ax.annotate(
        "reference\n40 W, 20 W margin",
        xy=(40, 20),
        xytext=(18, 33),
        fontsize=16,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK, lw=1.5),
    )
    ax.plot([109], [20], "s", ms=12, color=NAVY, zorder=6)
    ax.annotate(
        "P$_{tx,max}$ = 20 / 0.184 ≈ 109 W",
        xy=(109, 20),
        xytext=(72, 30),
        fontsize=16,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK, lw=1.5),
    )
    ax.set_xlim(10, 140)
    ax.set_ylim(0, 60)
    ax.set_xlabel(r"terminal electrical power $P_{tx}$ [W]")
    ax.set_ylabel(r"RTG margin $P_0 - P_{base}$ [W]")
    ax.grid(False)
    save(fig, "f4_design_map")


# ---------------------------------------------------------------- F5: mission timeline (R3)
def fig_mission():
    from rho_sweep import load_runs, mission_budget, mission_fixed

    plt.rcParams.update({"font.size": 20, "font.family": "Carlito"})

    ids = sorted(sweep[sweep.scenario == S20].report_id.unique().tolist())
    sw, per_node = load_runs(ids)
    sw = sw[sw.scenario == S20]
    b = sw[sw.algorithm == "lls_budget"]
    prof = per_node[(b.report_id.iloc[0], "lls_budget", S20)]["profile"]
    rho0 = 0.5
    lam = prof["decay_rate_per_s"]
    p0 = prof["source_p_base"] + rho0 * prof["p_tx"]
    t_end = np.log(p0 / prof["source_p_base"]) / lam / 86400
    days = np.arange(0, int(t_end) + 200, 1.0)
    yrs = days / 365.25
    cap_b = mission_budget(sw, S20, rho0, prof, days)
    row = b[np.isclose(b.rho, 0.2)].iloc[0]
    cap_f, _, _ = mission_fixed(
        per_node[(row.report_id, "lls_budget", S20)],
        row.capacity_tbit_day,
        rho0,
        days,
    )
    curves = {}
    for alg in ["lls", "fcp", "lifespan_aware"]:
        r = sw[sw.algorithm == alg].iloc[0]
        curves[alg], _, _ = mission_fixed(
            per_node[(r.report_id, alg, S20)], r.capacity_tbit_day, rho0, days
        )
    vol = lambda c: c.sum() / 1e3  # noqa: E731
    print(
        "volumes",
        vol(cap_b),
        vol(cap_f),
        {k: vol(v) for k, v in curves.items()},
    )
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(yrs, cap_b, color=ORANGE, lw=4, zorder=5)
    ax.plot(yrs, cap_f, color=NAVY, lw=3, ls="--", zorder=4)
    ax.plot(yrs, curves["lls"], color=BLUE, lw=3)
    ax.plot(yrs, curves["fcp"], color=AQUA, lw=3)
    ax.plot(yrs, curves["lifespan_aware"], color=VIOLET, lw=3)
    ax.text(
        4.35,
        62,
        f"adaptive budget\n{vol(cap_b):.0f} Pbit",
        color=INK,
        fontsize=19,
        fontweight="bold",
    )
    ax.text(
        1.0,
        72,
        f"best fixed duty cycle (d = 0.2)\n{vol(cap_f):.0f} Pbit · first loss at 3.8 y",
        color=INK,
        fontsize=17,
        va="top",
    )
    ax.text(
        0.15,
        22,
        f"LLS (unconstrained): {vol(curves['lls']):.0f} Pbit",
        color=INK,
        fontsize=17,
    )
    ax.text(
        0.15,
        47,
        "FCP, Lifespan-Aware (WiSEE):\nbatteries empty within 5 days, < 1 Pbit",
        color=INK,
        fontsize=16,
    )
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 90)
    ax.set_xlabel("mission time [years]")
    ax.set_ylabel("data delivered [Tbit/day]")
    save(fig, "f5_mission_timeline")


# ================================================================ simplified set (NTN Days)
# Same data, fewer numbers: words instead of thresholds, one headline per figure.
OUT_SIMPLE = os.path.join(OUT, "simple")
os.makedirs(OUT_SIMPLE, exist_ok=True)


def save_simple(fig, name):
    fig.savefig(os.path.join(OUT_SIMPLE, f"{name}.png"))
    fig.savefig(os.path.join(OUT_SIMPLE, f"{name}.pdf"))
    plt.close(fig)


def g1_zones():
    fig, ax = plt.subplots(figsize=(16, 4.2))
    ax.set_xscale("log")
    lo, hi = 0.01, 25
    rs, rc = 0.18, 0.89
    ax.axvspan(lo, rs, color=TINT_O, lw=0)
    ax.axvspan(rs, rc, color="#FFD3C0", lw=0)
    ax.axvspan(rc, hi, color=TINT_B, lw=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(
        "spare power ÷ laser power   =   share of the day a satellite can afford to transmit"
    )
    kw: dict[str, Any] = dict(ha="center", va="top", color=INK, fontsize=21)
    ax.text(
        np.sqrt(lo * rs),
        0.93,
        "too little power:\nenergy limits the data",
        **kw,
    )
    ax.text(
        np.sqrt(rs * rc),
        0.93,
        "a budget saves energy,\nno data lost",
        fontweight="bold",
        **kw,
    )
    ax.text(
        np.sqrt(rc * hi),
        0.93,
        "plenty of power:\nenergy does not matter",
        **kw,
    )
    ax.plot(
        [0.42], [0.3], marker="o", ms=22, color=ORANGE, mec="white", mew=2.5
    )
    ax.text(
        0.42,
        0.13,
        "our case",
        ha="center",
        va="top",
        fontsize=18,
        color=ORANGE,
        fontweight="bold",
    )
    sk: dict[str, Any] = dict(va="bottom", fontsize=15, color=MUTED)
    ax.text(rs * 0.95, 0.03, "stations saturated", ha="right", **sk)
    ax.text(
        rc * 1.06, 0.03, "transmits whenever it sees Earth", ha="left", **sk
    )
    ax.xaxis.set_minor_locator(NullLocator())
    save_simple(fig, "g1_zones")


def g2_same_data():
    s = sweep[sweep.scenario == S20]
    b = s[s.algorithm == "lls_budget"].sort_values("rho")
    b = b[b.rho <= 1.0]
    lls = s[s.algorithm == "lls"].iloc[0]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axvspan(0.184, lls.max_source_duty, color=TINT_O, lw=0)
    ax.plot(
        b.rho,
        b.capacity_tbit_day,
        "-o",
        color=ORANGE,
        ms=9,
        mec="white",
        mew=1.5,
        zorder=4,
    )
    ax.text(
        0.30,
        70,
        "LLS-Budget (ours)",
        color=INK,
        fontsize=21,
        fontweight="bold",
    )
    pts = [
        ("lls", "LLS: data only", BLUE),
        ("fcp", "FCP", AQUA),
        ("lifespan_aware", "Lifespan-Aware", VIOLET),
    ]
    for alg, label, col in pts:
        r = s[s.algorithm == alg].iloc[0]
        x, yv = r.max_source_duty, r.capacity_tbit_day
        ax.plot([x], [yv], "D", ms=14, color=col, mec="white", mew=2, zorder=5)
        if alg == "lls":
            ax.text(
                x - 0.01,
                yv - 5,
                label,
                ha="right",
                va="top",
                fontsize=18,
                color=INK,
            )
        else:
            ax.text(
                x - 0.025,
                yv,
                label,
                ha="right",
                va="center",
                fontsize=18,
                color=INK,
            )
    ax.axvline(0.184, color=ORANGE, lw=1.5, ls="--")
    ax.annotate(
        "",
        xy=(0.184, 88),
        xytext=(lls.max_source_duty, 88),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=2),
        annotation_clip=False,
    )
    ax.text(
        0.535,
        89.5,
        "same data with ~80% less spare power",
        ha="center",
        va="bottom",
        fontsize=21,
        color=INK,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 95)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel(
        "spare power ÷ laser power  (share of the day a satellite can transmit)"
    )
    ax.set_ylabel("data delivered [Tbit/day]")
    save_simple(fig, "g2_same_data")


def g3_stations():
    n = summ["size"].values
    cap = summ["lls_capacity"].values
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.step(n, cap, where="mid", color=BLUE, lw=3)
    ax.plot(n, cap, "o", color=BLUE, ms=10, mec="white", mew=1.5)
    for start, gs in [(4, 3), (20, 6), (36, 9), (52, 12)]:
        y = cap[n == start][0]
        ax.text(
            start + 6,
            y + 9,
            f"{gs} station terminals",
            ha="center",
            fontsize=18,
            color=INK,
        )
    ax.text(
        5,
        140,
        "data grows with ground stations,\nnot with satellites",
        fontsize=21,
        color=INK,
        fontweight="bold",
        va="top",
    )
    ax.set_xlabel("satellites at Mars")
    ax.set_ylabel("data delivered [Tbit/day]")
    ax.set_ylim(0, 175)
    ax.set_xticks([4, 16, 32, 48, 64])
    save_simple(fig, "g3_stations")


def g4_design_map(terminals="top"):
    """Old F4 (all data) with the rule written on the plot.

    terminals="top": DSOC/DOT marked as ticks on a top axis (no lines across the map).
    terminals="lines": DSOC/DOT as solid, labelled vertical lines.
    terminals="none": no terminal references (version used on the slide).
    """
    s = sweep[
        (sweep.scenario == S20) & (sweep.algorithm == "lls_budget")
    ].sort_values("rho")
    cmax = s.capacity_tbit_day.max()
    ptx = np.linspace(10, 140, 400)
    m = np.linspace(0.5, 60, 400)
    P, M = np.meshgrid(ptx, m)
    frac = np.interp(
        M / P, s.rho.values, s.capacity_tbit_day.values / cmax, right=1.0
    )
    loss = 100 * (1 - frac)
    fig, ax = plt.subplots(figsize=(13, 7.2))
    levels = [5, 25, 50, 75, 100]
    cols = ["#FDEDE6", "#F9C7B0", "#F39A74", "#EB6834"]
    cs = ax.contourf(P, M, loss, levels=levels, colors=cols)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label("data lost, 20 sources [%]", color=INK)
    cb.outline.set_visible(False)  # ty: ignore[call-non-callable]
    for size, col, lw in [(8, MUTED, 2), (20, NAVY, 4), (64, MUTED, 2)]:
        rs = float(summ.loc[summ["size"] == size, "rho_sat_95"].iloc[0])
        y = rs * ptx
        ok = y <= 60
        ax.plot(ptx[ok], y[ok], color=col, lw=lw)
        xe = min(138, 59 / rs)
        ax.text(
            xe,
            rs * xe + 1.2,
            f"{size} sources",
            ha="right",
            va="bottom",
            fontsize=16,
            color=INK,
            fontweight="bold" if size == 20 else "normal",
        )
    terms = [(61, "DSOC terminal\n61 W"), (110, "DOT allocation\n110 W")]
    if terminals == "top":
        top = ax.secondary_xaxis("top")
        top.set_xticks([x for x, _ in terms])
        top.set_xticklabels([lab for _, lab in terms], fontsize=15, color=INK)
        top.tick_params(length=10, width=2, color=INK)
        top.spines["top"].set_visible(False)
    elif terminals == "lines":
        for x, lab in terms:
            ax.axvline(x, color=INK, lw=2, alpha=0.55, zorder=3)
            ax.text(
                x,
                59,
                lab,
                fontsize=15,
                color=INK,
                ha="center",
                va="top",
                zorder=7,
                bbox=dict(
                    boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9
                ),
            )
    ax.plot([40], [20], "o", ms=18, color=ORANGE, mec="white", mew=3, zorder=6)
    ax.annotate(
        "reference\n40 W, 20 W margin",
        xy=(40, 20),
        xytext=(18, 31),
        fontsize=16,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK, lw=1.5),
    )
    ax.plot([109], [20], "s", ms=12, color=NAVY, zorder=6)
    ax.annotate(
        "P$_{tx,max}$ = 20 / 0.184 ≈ 109 W",
        xy=(109, 20),
        xytext=(70, 30),
        fontsize=16,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK, lw=1.5),
    )
    # the rule itself
    big = (
        terminals != "lines"
    )  # with vertical lines the box must fit left of the DSOC line
    ax.text(
        14,
        54 if big else 51,
        r"$P_{tx,\max} = \dfrac{M}{\rho_{sat}}$"
        + "\n"
        + r"$\rho_{sat} \approx 0.635\,\cdot\,N_{GS}\,/\,N_{src}$",
        fontsize=21 if big else 17,
        color=INK,
        va="top",
        linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=NAVY, lw=1.8),
        zorder=8,
    )
    ax.set_xlim(10, 140)
    ax.set_ylim(0, 60)
    ax.set_xlabel(r"terminal electrical power $P_{tx}$ [W]")
    ax.set_ylabel(r"RTG margin $M = P_0 - P_{base}$ [W]")
    ax.grid(False)
    save_simple(
        fig,
        "g4_design_map"
        if terminals == "none"
        else f"g4_design_map_{terminals}",
    )


def g4_sizing():
    s = sweep[
        (sweep.scenario == S20) & (sweep.algorithm == "lls_budget")
    ].sort_values("rho")
    cmax = s.capacity_tbit_day.max()
    ptx = np.linspace(10, 140, 400)
    m = np.linspace(0.5, 60, 400)
    P, M = np.meshgrid(ptx, m)
    frac = np.interp(
        M / P, s.rho.values, s.capacity_tbit_day.values / cmax, right=1.0
    )
    loss = 100 * (1 - frac)
    fig, ax = plt.subplots(figsize=(13, 7.2))
    levels = [5, 25, 50, 75, 100]
    cols = ["#FDEDE6", "#F9C7B0", "#F39A74", "#EB6834"]
    ax.contourf(P, M, loss, levels=levels, colors=cols)
    rs = float(summ.loc[summ["size"] == 20, "rho_sat_95"].iloc[0])
    ax.plot(ptx, rs * ptx, color=NAVY, lw=4)
    ax.text(
        136,
        rs * 136 + 1.5,
        "no-loss limit",
        ha="right",
        va="bottom",
        fontsize=19,
        color=INK,
        fontweight="bold",
    )
    # how to read the rule: from a given spare power, across to the limit, down to the laser
    m0 = 20.0
    pmax = m0 / rs
    ax.annotate(
        "",
        xy=(pmax, m0),
        xytext=(10, m0),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.2, ls="--"),
    )
    ax.annotate(
        "",
        xy=(pmax, 0.3),
        xytext=(pmax, m0),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.2, ls="--"),
    )
    ax.plot([pmax], [m0], "o", ms=13, color=NAVY, zorder=6)
    ax.text(
        12, m0 + 1.2, "your spare power", fontsize=17, color=INK, va="bottom"
    )
    ax.text(
        pmax - 2,
        30,
        "largest laser\nyou can afford",
        fontsize=17,
        color=INK,
        ha="right",
        va="bottom",
    )
    ax.annotate(
        "",
        xy=(pmax - 1, 22),
        xytext=(pmax - 12, 29.5),
        arrowprops=dict(arrowstyle="-", color=INK, lw=1.2),
    )
    ax.text(18, 50, "no data lost", fontsize=24, color=INK, fontweight="bold")
    ax.text(
        132,
        5,
        "data lost\n(darker = more)",
        fontsize=19,
        color="white",
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    ax.set_xlim(10, 140)
    ax.set_ylim(0, 60)
    ax.set_xlabel("laser terminal power [W]")
    ax.set_ylabel("spare generator power [W]")
    ax.grid(False)
    save_simple(fig, "g4_sizing")


def g5_mission():
    from rho_sweep import load_runs, mission_budget, mission_fixed

    plt.rcParams.update({"font.size": 20, "font.family": "Carlito"})
    ids = sorted(sweep[sweep.scenario == S20].report_id.unique().tolist())
    sw, per_node = load_runs(ids)
    sw = sw[sw.scenario == S20]
    b = sw[sw.algorithm == "lls_budget"]
    prof = per_node[(b.report_id.iloc[0], "lls_budget", S20)]["profile"]
    rho0 = 0.5
    lam = prof["decay_rate_per_s"]
    p0 = prof["source_p_base"] + rho0 * prof["p_tx"]
    t_end = np.log(p0 / prof["source_p_base"]) / lam / 86400
    days = np.arange(0, int(t_end) + 200, 1.0)
    yrs = days / 365.25
    cap_b = mission_budget(sw, S20, rho0, prof, days)
    row = b[np.isclose(b.rho, 0.2)].iloc[0]
    cap_f, _, _ = mission_fixed(
        per_node[(row.report_id, "lls_budget", S20)],
        row.capacity_tbit_day,
        rho0,
        days,
    )
    curves = {}
    for alg in ["lls", "fcp", "lifespan_aware"]:
        r = sw[sw.algorithm == alg].iloc[0]
        curves[alg], _, _ = mission_fixed(
            per_node[(r.report_id, alg, S20)], r.capacity_tbit_day, rho0, days
        )
    gain = cap_b.sum() / cap_f.sum() - 1
    print("g5 gain over best fixed", gain)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.fill_between(yrs, cap_f, cap_b, where=cap_b > cap_f, color=TINT_O, lw=0)
    ax.plot(yrs, cap_b, color=ORANGE, lw=4, zorder=5)
    ax.plot(yrs, cap_f, color=NAVY, lw=3, ls="--", zorder=4)
    ax.plot(yrs, curves["lls"], color=BLUE, lw=3)
    ax.plot(yrs, curves["fcp"], color=AQUA, lw=3)
    ax.plot(yrs, curves["lifespan_aware"], color=VIOLET, lw=3)
    ax.text(
        4.45,
        62,
        "budget that follows\nthe generator",
        color=ORANGE,
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        4.55,
        22,
        f"+{100 * gain:.0f}% data",
        color=ORANGE,
        fontsize=24,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        1.0,
        72,
        "best fixed budget: duty cycle (d = 0.2)",
        color=INK,
        fontsize=18,
        va="top",
    )
    ax.text(
        0.15, 22, "LLS: first satellite lost in days", color=BLUE, fontsize=17
    )
    ax.text(
        0.15,
        47,
        "FCP, Lifespan-Aware:\nbatteries empty within days",
        color=INK,
        fontsize=17,
    )
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 90)
    ax.set_xlabel("mission time [years]")
    ax.set_ylabel("data delivered [Tbit/day]")
    save_simple(fig, "g5_mission")


if __name__ == "__main__":
    if "--simple" in sys.argv:
        g1_zones()
        g2_same_data()
        g3_stations()
        g4_sizing()
        g5_mission()
        print("ok", os.listdir(OUT_SIMPLE))
    else:
        fig_rho_axis()
        fig_capacity_vs_rho()
        fig_stations()
        fig_design_map()
        fig_mission()
        print("ok", os.listdir(OUT))
