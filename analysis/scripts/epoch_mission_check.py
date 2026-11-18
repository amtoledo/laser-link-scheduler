"""
Does the mission-time extrapolation change when each mission day uses its own geometry?

Compares the adaptive LLS-Budget mission data volume computed two ways:
  (a) day-0 geometry repeated every day (what ptx_sweep.py / rho_sweep.py do), and
  (b) epoch-resolved geometry: capacity C(rho, day) measured on 24 h plans generated at many start
      days (scripts/run_epoch_sweep.sh), interpolated in rho and in day, and extended periodically
      beyond the sampled window.
Both use the same rho grid, so the difference isolates the geometry.

    uv run python analysis/scripts/epoch_mission_check.py --report-ids $(cat output/epoch_ids_20_long.txt)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import MLConfig, PLOTS_ROOT  # noqa: E402
from analysis.scripts.rho_sweep import DAY, load_runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-ids", nargs="+", type=int, required=True)
    ap.add_argument("--p-tx", nargs="+", type=float, default=[20, 40, 80, 120])
    ap.add_argument(
        "--margins", nargs="+", type=float, default=[5, 10, 25, 50]
    )
    ap.add_argument("--p-base", type=float, default=200.0)
    ap.add_argument("--out-id", default="epoch_mission_check")
    args = ap.parse_args()

    sweep, _ = load_runs(args.report_ids)
    b = sweep[sweep.algorithm == "lls_budget"].copy()
    m = b.scenario.str.extract(r"_inc_(\d+)_d([\d.]+)")
    b["size"], b["day"] = m[0].astype(int), m[1].astype(float)
    lam = MLConfig.DECAY_RATE
    rows = []
    for size, s in b.groupby("size"):
        rhos = np.array(sorted(s.rho.unique()))
        days = np.array(sorted(s.day.unique()))
        grid = (
            s.pivot_table(
                index="day", columns="rho", values="capacity_tbit_day"
            )
            .loc[days, rhos]
            .values
        )
        # rho = 0 means no budget, no capacity
        rgrid = np.concatenate([[0.0], rhos])
        grid = np.hstack([np.zeros((len(days), 1)), grid])
        period = days[-1] + (
            days[1] - days[0]
        )  # sampled window, extended periodically
        day0 = grid[0]

        for mgn in args.margins:
            p0 = args.p_base + mgn
            t_hw = np.log(p0 / args.p_base) / lam / DAY
            mission_days = np.arange(0, int(t_hw) + 1, 1.0)
            for ptx in args.p_tx:
                rho_t = (
                    p0 * np.exp(-lam * mission_days * DAY) - args.p_base
                ) / ptx
                rho_t = np.clip(rho_t, 0.0, None)
                c_fixed = np.interp(rho_t, rgrid, day0)
                # epoch-resolved: interpolate over rho on every sampled day, then over day
                phase = np.mod(mission_days, period)
                per_day = np.array(
                    [np.interp(rho_t, rgrid, g) for g in grid]
                )  # (n_days, n_mission)
                ext_days = np.concatenate([days, [period]])
                per_day = np.vstack([per_day, per_day[:1]])
                idx = np.searchsorted(ext_days, phase, side="right") - 1
                w = (phase - ext_days[idx]) / (
                    ext_days[idx + 1] - ext_days[idx]
                )
                cols = np.arange(len(mission_days))
                c_epoch = (1 - w) * per_day[idx, cols] + w * per_day[
                    idx + 1, cols
                ]
                v_fixed, v_epoch = c_fixed.sum() / 1e3, c_epoch.sum() / 1e3
                rows.append(
                    {
                        "size": size,
                        "margin_w": mgn,
                        "p_tx": ptx,
                        "rho0": mgn / ptx,
                        "mission_years": t_hw / 365.25,
                        "volume_day0_pbit": v_fixed,
                        "volume_epoch_pbit": v_epoch,
                        "diff_pct": 100 * (v_epoch / v_fixed - 1),
                    }
                )
    out = pd.DataFrame(rows)
    out_dir = os.path.join(PLOTS_ROOT, args.out_id)
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "epoch_mission_check.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    print(
        f"\nmax |diff| = {out.diff_pct.abs().max():.2f} %; written to {out_dir}"
    )


if __name__ == "__main__":
    main()
