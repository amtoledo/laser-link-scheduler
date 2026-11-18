"""
Cross-size summary of the rho and P_tx sweeps (one row / one curve per constellation size).

    uv run python analysis/scripts/all_sizes_summary.py --rho-dir output/plots/rho_sweep_all \
        --ptx-dir output/plots/ptx_sweep_all --out-id all_sizes_summary
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

plt.rcParams.update({"font.size": 13})
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def size_of(s: pd.Series) -> pd.Series:
    return s.str.extract(r"_inc_(\d+)$")[0].astype(int)


def rho_sat(b: pd.DataFrame, frac=0.95) -> float:
    b = b.sort_values("rho")
    return float(
        np.interp(
            frac * b.capacity_tbit_day.max(),
            b.capacity_tbit_day.values,
            b.rho.values,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho-dir", required=True)
    ap.add_argument("--ptx-dir", required=True)
    ap.add_argument("--out-id", default="all_sizes_summary")
    args = ap.parse_args()
    out_dir = os.path.join(PLOTS_ROOT, args.out_id)
    for sub in ("pdf", "png"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    sw = pd.read_csv(os.path.join(args.rho_dir, "sweep.csv"))
    sw["size"] = size_of(sw.scenario)
    px = pd.read_csv(os.path.join(args.ptx_dir, "ptx_sweep.csv"))
    px["size"] = size_of(px.scenario)

    rows = []
    for n, s in sw.groupby("size"):
        b = s[s.algorithm == "lls_budget"]
        lls = s[s.algorithm == "lls"].iloc[0]
        fcp = s[s.algorithm == "fcp"].iloc[0]
        v1 = s[s.algorithm == "lifespan_aware"]
        row = {
            "size": n,
            "lls_capacity": lls.capacity_tbit_day,
            "budget_max_capacity": b.capacity_tbit_day.max(),
            "budget_at_rho_0.2_pct_of_lls": 100
            * np.interp(0.2, b.rho, b.capacity_tbit_day)
            / lls.capacity_tbit_day,
            "rho_sat_95": rho_sat(b),
            "lls_neutral_rho": lls.max_source_duty,
            "fcp_capacity": fcp.capacity_tbit_day,
            "fcp_neutral_rho": fcp.max_source_duty,
            "v1_neutral_rho(rho=1 run)": v1.max_source_duty.iloc[0]
            if not v1.empty
            else np.nan,
        }
        for m in sorted(px.margin_w.unique()):
            a = px[
                (px["size"] == n)
                & (px.margin_w == m)
                & (px.algorithm == "lls_budget_adaptive")
            ].set_index("p_tx")
            row[f"ptx_max_M{m:g}"] = m / row["rho_sat_95"]
            row[f"vol_loss_40to120_M{m:g}_pct"] = 100 * (
                1 - a.mission_volume_pbit[120] / a.mission_volume_pbit[40]
            )
        rows.append(row)
    summ = pd.DataFrame(rows).sort_values("size")
    summ.round(3).to_csv(
        os.path.join(out_dir, "summary_by_size.csv"), index=False
    )

    # (1) rho_sat and energy-neutral thresholds vs size
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        summ["size"],
        summ.rho_sat_95,
        "o-",
        label=r"LLS-Budget $\rho_{sat}$ (95% capacity)",
    )
    ax.plot(
        summ["size"],
        summ.lls_neutral_rho,
        "^--",
        label="LLS energy-neutral above",
    )
    ax.plot(
        summ["size"],
        summ.fcp_neutral_rho,
        "v--",
        label="FCP energy-neutral above",
    )
    ax.plot(
        summ["size"],
        summ["v1_neutral_rho(rho=1 run)"],
        "x--",
        label="Lifespan-Aware v1 energy-neutral above",
    )
    ax.set_xlabel("number of source nodes")
    ax.set_ylabel(r"margin ratio $\rho$")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pdf", "rho_thresholds_vs_size.pdf"))
    fig.savefig(
        os.path.join(out_dir, "png", "rho_thresholds_vs_size.png"), dpi=150
    )
    plt.close(fig)

    # (2) mission-volume loss 40 -> 120 W vs size, per margin
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, m in enumerate(sorted(px.margin_w.unique())):
        ax.plot(
            summ["size"],
            summ[f"vol_loss_40to120_M{m:g}_pct"],
            "o-",
            color=f"C{i}",
            label=f"margin {m:g} W",
        )
    ax.set_xlabel("number of source nodes")
    ax.set_ylabel("mission volume lost, 40 W -> 120 W [%]")
    ax.set_title(
        "LLS-Budget (adaptive): sensitivity to terminal power", fontsize=11
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pdf", "ptx_sensitivity_vs_size.pdf"))
    fig.savefig(
        os.path.join(out_dir, "png", "ptx_sensitivity_vs_size.png"), dpi=150
    )
    plt.close(fig)

    # (3) normalised daily capacity vs rho, one curve per size
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap("viridis")
    sizes = sorted(sw["size"].unique())
    for i, n in enumerate(sizes):
        b = sw[(sw["size"] == n) & (sw.algorithm == "lls_budget")].sort_values(
            "rho"
        )
        lls = sw[
            (sw["size"] == n) & (sw.algorithm == "lls")
        ].capacity_tbit_day.iloc[0]
        ax.plot(
            b.rho,
            b.capacity_tbit_day / lls,
            "-",
            color=cmap(i / (len(sizes) - 1)),
            label=f"{n}" if n in (4, 16, 32, 48, 64) else None,
        )
    ax.set_xlim(0, 0.6)
    ax.set_xlabel(r"margin ratio $\rho$")
    ax.set_ylabel("LLS-Budget capacity / LLS capacity")
    ax.legend(title="sources", fontsize=9)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            out_dir, "pdf", "normalised_capacity_vs_rho_all_sizes.pdf"
        )
    )
    fig.savefig(
        os.path.join(
            out_dir, "png", "normalised_capacity_vs_rho_all_sizes.png"
        ),
        dpi=150,
    )
    plt.close(fig)

    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
