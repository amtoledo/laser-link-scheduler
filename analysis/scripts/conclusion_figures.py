"""
Two summary figures for the thesis/talk conclusions.

1. design_map: which (terminal power, RTG margin) combinations lose data, for three network sizes.
   Shading = LLS-Budget daily capacity as % of the unconstrained LLS capacity (20 sources).
   Lines = no-loss boundary P_tx = margin / rho_sat for 4, 8 and 20 sources.
2. ground_segment: unconstrained LLS capacity vs number of sources, with the ground-station count.

    uv run python analysis/scripts/conclusion_figures.py --rho-sweep output/plots/rho_sweep_all/sweep.csv \
        --summary output/plots/all_sizes_summary/summary_by_size.csv
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.constants import PLOTS_ROOT  # noqa: E402

plt.rcParams.update(
    {"font.size": 13, "axes.spines.top": False, "axes.spines.right": False}
)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference categorical slots 1-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho-sweep", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out-id", default="conclusions")
    args = ap.parse_args()
    out = os.path.join(PLOTS_ROOT, args.out_id)
    for sub in ("pdf", "png"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(out, "pdf", f"{name}.pdf"))
        fig.savefig(os.path.join(out, "png", f"{name}.png"), dpi=160)
        plt.close(fig)

    sw = pd.read_csv(args.rho_sweep)
    summ = pd.read_csv(args.summary)

    # ---- 1. design map -------------------------------------------------------------------
    scen = "mars_relay_earth_scenario_inc_20"
    b = sw[(sw.scenario == scen) & (sw.algorithm == "lls_budget")].sort_values(
        "rho"
    )
    lls = sw[
        (sw.scenario == scen) & (sw.algorithm == "lls")
    ].capacity_tbit_day.iloc[0]
    ptx = np.linspace(10, 150, 281)
    margin = np.linspace(1, 60, 237)
    P, M = np.meshgrid(ptx, margin)
    frac = (
        100 * np.interp(M / P, b.rho.values, b.capacity_tbit_day.values) / lls
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    levels = [0, 25, 50, 75, 90, 95, 100]
    cf = ax.contourf(P, M, frac, levels=levels, cmap="Blues", alpha=0.9)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(
        "daily data delivered, % of maximum (20 sources)", color=INK2
    )
    for color, n, label_y in zip(SERIES[::-1], (4, 8, 20), (0.58, 0.58, 0.58)):
        rs = float(summ.loc[summ["size"] == n, "rho_sat_95"].iloc[0])
        ax.plot(ptx, rs * ptx, color=color, lw=2)
        x_lab = min(150, 58 / rs) * 0.97
        ax.text(
            x_lab,
            rs * x_lab,
            f" {n} sources",
            color=INK,
            fontsize=11,
            ha="right",
            va="bottom",
        )
    ax.text(
        18,
        52,
        "above the line:\nenergy is not the limit",
        color=INK,
        fontsize=12,
        va="top",
    )
    ax.text(
        147,
        4,
        "below the line:\nenergy-limited, data ∝ margin / P$_{tx}$",
        color=INK,
        fontsize=12,
        ha="right",
        va="bottom",
    )
    ax.set_xlim(10, 150)
    ax.set_ylim(1, 60)
    ax.set_xlabel("terminal electrical power while transmitting, P$_{tx}$ [W]")
    ax.set_ylabel("RTG margin, P$_0$ − P$_{base}$ [W]")
    ax.set_title(
        "No-loss line: margin = ρ$_{sat}$ × P$_{tx}$ (ρ$_{sat}$ ≈ 0.64 × stations / sources)",
        fontsize=12,
        color=INK,
    )
    save(fig, "design_map")

    # ---- 2. ground segment is the bottleneck ----------------------------------------------
    summ = summ.sort_values("size")
    gs = 3 * np.ceil(summ["size"] / 16)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.step(
        summ["size"], summ.lls_capacity, where="mid", color=SERIES[0], lw=2
    )
    ax.plot(summ["size"], summ.lls_capacity, "o", color=SERIES[0], ms=6)
    for n_gs in sorted(gs.unique()):
        sel = gs == n_gs
        x = summ["size"][sel].mean()
        y = summ.lls_capacity[sel].max()
        ax.text(
            x,
            y + 6,
            f"{int(n_gs)} ground stations",
            ha="center",
            color=INK2,
            fontsize=11,
        )
    ax.set_ylim(0, summ.lls_capacity.max() * 1.15)
    ax.set_xlabel("number of source satellites at Mars")
    ax.set_ylabel("data delivered [Tbit/day]")
    ax.set_title(
        "Capacity grows only when ground stations are added",
        fontsize=12,
        color=INK,
    )
    ax.grid(axis="y", color=GRID, lw=0.8)
    save(fig, "ground_segment_bottleneck")
    print("written to", out)


if __name__ == "__main__":
    main()
