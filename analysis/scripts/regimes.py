"""
Option C: one figure, three ground segments. For every constellation size it shows how much spare
energy each scheduler needs:
  - LLS-Budget: margin ratio rho at which it delivers 95% of its maximum data (rho_sat);
  - LLS and FCP: the duty cycle of their busiest source, i.e. the rho they need to avoid
    draining any battery (they ignore energy, so this is fixed by their plan).
The gap between the lines is the energy LLS-Budget saves. It closes when there are fewer
satellites than ground-station terminals: every satellite must then transmit whenever it sees
Earth, whatever the scheduler.

    uv run python analysis/scripts/regimes.py --full-ids $(cat output/gs_count_full_ids.txt) \
        --gs3-ids $(cat output/gs_count_gs3_ids.txt) --gs12-ids $(cat output/gs_count_gs12_ids.txt)
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
    {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}
)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def rho_sat(b, frac=0.95):
    b = b.sort_values("rho")
    rho = np.concatenate([[0.0], b.rho.values])
    cap = np.concatenate([[0.0], b.capacity_tbit_day.values])
    cap = np.maximum.accumulate(cap)  # guard against tiny non-monotonic noise
    return float(np.interp(frac * cap.max(), cap, rho))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ids", nargs="+", type=int, required=True)
    ap.add_argument("--gs3-ids", nargs="+", type=int, required=True)
    ap.add_argument("--gs12-ids", nargs="+", type=int, required=True)
    ap.add_argument("--out-id", default="regimes")
    args = ap.parse_args()
    out = os.path.join(PLOTS_ROOT, args.out_id)
    for sub in ("pdf", "png"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    rows = []
    variants = [
        ("3 stations (one per DSN site)", args.gs3_ids, lambda n: 3),
        (
            "Original: 3 per 16 satellites",
            args.full_ids,
            lambda n: int(3 * np.ceil(n / 16)),
        ),
        ("12 stations", args.gs12_ids, lambda n: 12),
    ]
    for label, ids, n_gs in variants:
        sw, _ = load_runs(ids)
        sw["size"] = sw.scenario.str.extract(r"_inc_(\d+)_")[0].astype(int)
        for size, g in sw.groupby("size"):
            b = g[g.algorithm == "lls_budget"]
            rows.append(
                {
                    "ground_segment": label,
                    "size": size,
                    "stations": n_gs(size),
                    "satellites_per_station": size / n_gs(size),
                    "budget_rho_needed": rho_sat(b),
                    "lls_rho_needed": g[
                        g.algorithm == "lls"
                    ].max_source_duty.iloc[0],
                    "fcp_rho_needed": g[
                        g.algorithm == "fcp"
                    ].max_source_duty.iloc[0],
                    "lls_capacity": g[
                        g.algorithm == "lls"
                    ].capacity_tbit_day.iloc[0],
                    "budget_capacity_max": b.capacity_tbit_day.max(),
                }
            )
    res = pd.DataFrame(rows).sort_values(["ground_segment", "size"])
    res["energy_saved_vs_lls_pct"] = 100 * (
        1 - res.budget_rho_needed / res.lls_rho_needed
    )
    res.round(3).to_csv(os.path.join(out, "regimes.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (label, _, _) in zip(axes, variants):
        r = res[res.ground_segment == label]
        ax.fill_between(
            r["size"],
            r.budget_rho_needed,
            r.lls_rho_needed,
            where=r.lls_rho_needed >= r.budget_rho_needed,
            color=BLUE,
            alpha=0.12,
            lw=0,
            label="energy LLS-Budget saves vs LLS",
        )
        ax.plot(
            r["size"],
            r.lls_rho_needed,
            "^-",
            color=ORANGE,
            lw=2,
            ms=6,
            label="LLS",
        )
        ax.plot(
            r["size"],
            r.fcp_rho_needed,
            "v-",
            color=AQUA,
            lw=2,
            ms=6,
            label="FCP",
        )
        ax.plot(
            r["size"],
            r.budget_rho_needed,
            "o-",
            color=BLUE,
            lw=2,
            ms=6,
            label="LLS-Budget",
        )
        tight = r[r.satellites_per_station < 1]
        if not tight.empty:
            ax.axvspan(
                r["size"].min() - 2,
                tight["size"].max() + 2,
                color=INK2,
                alpha=0.08,
                lw=0,
            )
            ax.text(
                tight["size"].max() + 3,
                0.05,
                "\u2190 fewer satellites\n    than stations",
                ha="left",
                fontsize=9,
                color=INK2,
            )
        ax.set_title(label, fontsize=12, color=INK)
        ax.set_xlabel("satellites at Mars")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("spare-power ratio ρ needed")
    axes[0].text(
        64,
        0.005,
        "lowest ρ simulated: 0.05",
        ha="right",
        fontsize=8,
        color=INK2,
    )
    axes[-1].legend(fontsize=9, loc="center right")
    fig.suptitle(
        "How much spare energy each scheduler needs (LLS-Budget: 95% of max data; "
        "LLS/FCP: no battery drained)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "pdf", "energy_needed_by_ground_segment.pdf")
    )
    fig.savefig(
        os.path.join(out, "png", "energy_needed_by_ground_segment.png"),
        dpi=160,
    )
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print("written to", out)


if __name__ == "__main__":
    main()
