"""
Epoch sweep analysis: how much does the 24 h schedule change from one day to another?

Reads reports produced by scripts/run_epoch_sweep.sh (scenarios named ..._inc_<N>_d<day>) and
tabulates daily capacity per scheduler (and per rho for lls_budget) against the start day.

Usage:
    uv run python analysis/scripts/epoch_sweep.py --report-ids $(cat output/epoch_ids_20_short.txt) --out-id epoch_short_20
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
from analysis.scripts.rho_sweep import load_runs  # noqa: E402

plt.rcParams.update({"font.size": 13})
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-ids", nargs="+", type=int, required=True)
    ap.add_argument("--out-id", default="epoch_sweep")
    ap.add_argument(
        "--reference-ids",
        nargs="*",
        type=int,
        default=[],
        help="optional reports on the original IPN-D plan, drawn as day-0 markers",
    )
    args = ap.parse_args()

    sweep, per_node = load_runs(args.report_ids)
    m = sweep.scenario.str.extract(r"_inc_(\d+)_d([\d.]+)")
    sweep["size"] = m[0].astype(int)
    sweep["day"] = m[1].astype(float)
    sweep["label"] = np.where(
        sweep.algorithm == "lls_budget",
        "lls_budget rho=" + sweep.rho.round(3).astype(str),
        sweep.algorithm,
    )
    out_dir = os.path.join(PLOTS_ROOT, args.out_id)
    for sub in ("pdf", "png"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    sweep.sort_values(["size", "label", "day"]).to_csv(
        os.path.join(out_dir, "epoch_sweep.csv"), index=False
    )

    stats = (
        sweep.groupby(["size", "label"])
        .capacity_tbit_day.agg(["mean", "std", "min", "max", "count"])
        .assign(
            cv_pct=lambda d: 100 * d["std"] / d["mean"],
            range_pct=lambda d: 100 * (d["max"] - d["min"]) / d["mean"],
        )
    )
    day0 = sweep[sweep.day == 0].set_index(["size", "label"]).capacity_tbit_day
    stats["day0"] = day0
    stats["day0_vs_mean_pct"] = 100 * (stats["day0"] / stats["mean"] - 1)
    stats.round(3).to_csv(os.path.join(out_dir, "epoch_stats.csv"))

    duty = sweep.groupby(["size", "label"])[
        ["mean_source_duty", "max_source_duty"]
    ].agg(["mean", "min", "max"])
    duty.round(3).to_csv(os.path.join(out_dir, "epoch_duty.csv"))

    for size in sorted(sweep["size"].unique()):
        s = sweep[sweep["size"] == size]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for i, lab in enumerate(
            sorted(
                s.label.unique(), key=lambda x: (not x.startswith("lls_b"), x)
            )
        ):
            r = s[s.label == lab].sort_values("day")
            ax.plot(
                r.day,
                r.capacity_tbit_day,
                "o-" if len(r) < 30 else "-",
                ms=3,
                label=lab,
                color=f"C{i}",
            )
        ax.set_xlabel("start day after the IPN-D epoch")
        ax.set_ylabel("capacity [Tbit/day]")
        ax.set_title(
            f"mars_relay_earth_scenario_inc_{size}: one 24 h plan per start day",
            fontsize=11,
        )
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, "pdf", f"capacity_vs_day_{size}.pdf")
        )
        fig.savefig(
            os.path.join(out_dir, "png", f"capacity_vs_day_{size}.png"),
            dpi=150,
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.2))
        for i, lab in enumerate(["lls", "fcp"]):
            r = s[s.label == lab].sort_values("day")
            if not r.empty:
                ax.plot(
                    r.day,
                    r.max_source_duty,
                    "-",
                    label=f"{lab}: busiest source",
                    color=f"C{i}",
                )
                ax.plot(
                    r.day,
                    r.mean_source_duty,
                    "--",
                    label=f"{lab}: mean source",
                    color=f"C{i}",
                )
        ax.set_xlabel("start day after the IPN-D epoch")
        ax.set_ylabel("transmit duty cycle")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pdf", f"duty_vs_day_{size}.pdf"))
        fig.savefig(
            os.path.join(out_dir, "png", f"duty_vs_day_{size}.png"), dpi=150
        )
        plt.close(fig)

    pd.set_option("display.width", 200)
    print(stats.round(2).to_string())
    print()
    print(duty.round(3).to_string())
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
