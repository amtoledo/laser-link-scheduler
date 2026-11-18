"""
All ground stations (3 per 16 sources, co-located at the 3 DSN sites) vs exactly 3 stations
(one per site), same constellations, day 0, plans from tools/contact_plan_generator.py.

    uv run python analysis/scripts/gs_count.py --full-ids $(cat output/gs_count_full_ids.txt) \
        --gs3-ids $(cat output/gs_count_gs3_ids.txt)
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
from analysis.scripts.rho_sweep import load_runs  # noqa: E402
from src.constants import PLOTS_ROOT  # noqa: E402

plt.rcParams.update(
    {"font.size": 13, "axes.spines.top": False, "axes.spines.right": False}
)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
INK2, GRID = "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VARIANT_COLORS = (
    ("all stations", BLUE),
    ("3 stations", ORANGE),
    ("12 stations", AQUA),
)


def rho_sat(b, frac=0.95):
    b = b.sort_values("rho")
    rho = np.concatenate([[0.0], b.rho.values])
    cap = np.concatenate([[0.0], b.capacity_tbit_day.values])
    return float(np.interp(frac * cap.max(), cap, rho))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ids", nargs="+", type=int, required=True)
    ap.add_argument("--gs3-ids", nargs="+", type=int, required=True)
    ap.add_argument("--gs12-ids", nargs="*", type=int, default=[])
    ap.add_argument("--out-id", default="gs_count")
    args = ap.parse_args()
    out = os.path.join(PLOTS_ROOT, args.out_id)
    for sub in ("pdf", "png"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    frames = []
    variants = [("all stations", args.full_ids), ("3 stations", args.gs3_ids)]
    if args.gs12_ids:
        variants.append(("12 stations", args.gs12_ids))
    for variant, ids in variants:
        sw, _ = load_runs(ids)
        sw["variant"] = variant
        sw["size"] = sw.scenario.str.extract(r"_inc_(\d+)_")[0].astype(int)
        frames.append(sw)
    sw = pd.concat(frames)

    rows = []
    for (variant, size), g in sw.groupby(["variant", "size"]):
        b = g[g.algorithm == "lls_budget"]
        n_gs = {"3 stations": 3, "12 stations": 12}.get(
            variant, int(3 * np.ceil(size / 16))
        )
        rows.append(
            {
                "variant": variant,
                "size": size,
                "ground_stations": n_gs,
                "lls": g[g.algorithm == "lls"].capacity_tbit_day.iloc[0],
                "fcp": g[g.algorithm == "fcp"].capacity_tbit_day.iloc[0],
                "budget_rho_0.1": b[
                    np.isclose(b.rho, 0.1)
                ].capacity_tbit_day.iloc[0],
                "budget_rho_0.2": b[
                    np.isclose(b.rho, 0.2)
                ].capacity_tbit_day.iloc[0],
                "rho_sat_95": rho_sat(b),
                "rho_sat_rule": 0.635 * n_gs / size,
                "lls_busiest_source_duty": g[
                    g.algorithm == "lls"
                ].max_source_duty.iloc[0],
            }
        )
    res = pd.DataFrame(rows).sort_values(["variant", "size"])
    res.round(3).to_csv(os.path.join(out, "gs_count.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for variant, color in VARIANT_COLORS:
        r = res[res.variant == variant]
        if r.empty:
            continue
        ax.plot(
            r["size"],
            r.lls,
            "o-",
            color=color,
            lw=2,
            ms=6,
            label=f"LLS, {variant}",
        )
    ax.set_ylim(0, res.lls.max() * 1.12)
    ax.set_xlabel("number of source satellites at Mars")
    ax.set_ylabel("data delivered [Tbit/day]")
    ax.set_title(
        "Same constellations, different numbers of ground stations",
        fontsize=12,
    )
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pdf", "capacity_by_station_count.pdf"))
    fig.savefig(
        os.path.join(out, "png", "capacity_by_station_count.png"), dpi=160
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for variant, color in VARIANT_COLORS:
        r = res[res.variant == variant]
        if r.empty:
            continue
        ax.plot(
            r["size"],
            r.rho_sat_95,
            "o-",
            color=color,
            lw=2,
            ms=6,
            label=f"measured, {variant}",
        )
        ax.plot(
            r["size"],
            r.rho_sat_rule.clip(upper=1.0),
            ":",
            color=color,
            lw=1.5,
            label=f"0.64 × stations / sources, {variant}",
        )
    lowest = sw[sw.algorithm == "lls_budget"].rho.min()
    ax.axhline(lowest, color=INK2, lw=0.8, ls="--")
    ax.text(
        res["size"].min(),
        lowest + 0.01,
        f"lowest ρ simulated ({lowest:g}): points on this line mean ≤ {lowest:g}",
        ha="left",
        va="bottom",
        fontsize=9,
        color=INK2,
    )
    ceiling = res.lls_busiest_source_duty.max()
    ax.axhline(ceiling, color=INK2, lw=0.8, ls="--")
    ax.text(
        res["size"].max(),
        ceiling + 0.01,
        f"a satellite cannot transmit more than ~{ceiling:.2f} of the day (its view of Earth)",
        ha="right",
        va="bottom",
        fontsize=9,
        color=INK2,
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("number of source satellites at Mars")
    ax.set_ylabel("ρ needed for 95% of the data")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pdf", "rho_needed_by_station_count.pdf"))
    fig.savefig(
        os.path.join(out, "png", "rho_needed_by_station_count.png"), dpi=160
    )
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print("written to", out)


if __name__ == "__main__":
    main()
