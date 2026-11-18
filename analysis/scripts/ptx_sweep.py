"""
P_tx sweep (terminal efficiency) on top of a rho sweep.

The data rate is fixed (DSOC reference); P_tx is the electrical power the optical terminal draws
while transmitting, so sweeping P_tx sweeps the terminal's efficiency (energy per bit = P_tx / R_b).
With RTG margin M = P0 - P_base, the margin ratio is rho = M / P_tx.

Because the rate is fixed, the lls_budget schedule depends on rho only, and every schedule's duty
cycles are dimensionless. One rho sweep (scripts/run_rho_sweep.sh) therefore covers every
(P_tx, M) pair: this script re-evaluates the measured schedules under each pair.
Lifespan-Aware v1 is the exception (its weights depend on the power profile); it is evaluated from
its own runs when they exist for that exact pair (scripts/run_ptx_sweep.sh), otherwise skipped.

Usage:
    uv run python analysis/scripts/ptx_sweep.py --report-ids $(cat output/rho_sweep_ids.txt output/ptx_v1_ids.txt)
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
from analysis.scripts.rho_sweep import (  # noqa: E402
    BUDGET_ALGORITHMS,
    DAY,
    load_runs,
    mission_budget,
    mission_fixed,
)

plt.rcParams.update({"font.size": 13})
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

UNCONSTRAINED = ["lls", "fcp"]


def saturation_rho(b: pd.DataFrame, frac: float = 0.95) -> float:
    """Smallest rho at which lls_budget reaches `frac` of its maximum capacity (interpolated)."""
    b = b.sort_values("rho")
    target = frac * b.capacity_tbit_day.max()
    return float(np.interp(target, b.capacity_tbit_day.values, b.rho.values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-ids", nargs="+", type=int, required=True)
    ap.add_argument(
        "--p-tx", nargs="+", type=float, default=[20, 40, 60, 80, 100, 120]
    )
    ap.add_argument(
        "--margins", nargs="+", type=float, default=[5, 10, 25, 50]
    )
    ap.add_argument("--p-base", type=float, default=200.0)
    ap.add_argument("--out-id", default="ptx_sweep")
    args = ap.parse_args()

    sweep, per_node = load_runs(args.report_ids)
    out_dir = os.path.join(PLOTS_ROOT, args.out_id)
    for sub in ("pdf", "png"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pdf", f"{name}.pdf"))
        fig.savefig(os.path.join(out_dir, "png", f"{name}.png"), dpi=150)
        plt.close(fig)

    rows = []
    for scen in sorted(sweep.scenario.unique()):
        s = sweep[sweep.scenario == scen]
        b = s[s.algorithm.isin(BUDGET_ALGORITHMS)].sort_values("rho")
        if b.empty:
            continue
        rho_sat = saturation_rho(b)
        prof = per_node[(b.report_id.iloc[0], b.algorithm.iloc[0], scen)][
            "profile"
        ]
        lam = prof["decay_rate_per_s"]

        for m in args.margins:
            p0 = args.p_base + m
            t_hw_days = np.log(p0 / args.p_base) / lam / DAY
            days = np.arange(0, int(t_hw_days) + 400, 1.0)
            for ptx in args.p_tx:
                rho0 = m / ptx
                base = {
                    "scenario": scen,
                    "p_tx": ptx,
                    "margin_w": m,
                    "rho0": rho0,
                }

                # Adaptive budget
                cap = mission_budget(
                    sweep, scen, rho0, prof, days, p_tx=ptx, p_base=args.p_base
                )
                rows.append(
                    base
                    | {
                        "algorithm": "lls_budget_adaptive",
                        "day1_capacity_tbit": cap[0],
                        "energy_neutral": True,
                        "mission_volume_pbit": cap.sum() / 1e3,
                        "first_source_death_years": np.nan,
                    }
                )

                # Best fixed duty cycle d (lls_budget schedule measured at rho = d)
                best = None
                for _, r in b[b.rho > 0].iterrows():
                    node = per_node[(r.report_id, r.algorithm, scen)]
                    c, fsd, _ = mission_fixed(
                        node,
                        r.capacity_tbit_day,
                        rho0,
                        days,
                        p_tx=ptx,
                        p_base=args.p_base,
                    )
                    if best is None or c.sum() > best[1].sum():
                        best = (r.rho, c, fsd)
                if best is not None:
                    rows.append(
                        base
                        | {
                            "algorithm": "lls_budget_fixed_d",
                            "fixed_d": best[0],
                            "day1_capacity_tbit": best[1][0],
                            "energy_neutral": best[0] <= rho0 + 1e-9,
                            "mission_volume_pbit": best[1].sum() / 1e3,
                            "first_source_death_years": np.nan
                            if best[2] is None
                            else best[2] / 365.25,
                        }
                    )

                # Unconstrained schedules flown unchanged (power-independent schedules)
                for alg in UNCONSTRAINED:
                    u = s[s.algorithm == alg]
                    if u.empty:
                        continue
                    r = u.iloc[0]
                    node = per_node[(r.report_id, alg, scen)]
                    c, fsd, _ = mission_fixed(
                        node,
                        r.capacity_tbit_day,
                        rho0,
                        days,
                        p_tx=ptx,
                        p_base=args.p_base,
                    )
                    rows.append(
                        base
                        | {
                            "algorithm": alg,
                            "day1_capacity_tbit": r.capacity_tbit_day,
                            "energy_neutral": r.max_source_duty <= rho0 + 1e-9,
                            "mission_volume_pbit": c.sum() / 1e3,
                            "first_source_death_years": np.nan
                            if fsd is None
                            else fsd / 365.25,
                        }
                    )

                # Lifespan-Aware v1: only from a run made with this exact profile
                v1 = s[
                    (s.algorithm == "lifespan_aware")
                    & np.isclose(s.p_tx, ptx)
                    & np.isclose(s.p_base, args.p_base)
                    & np.isclose(s.rho, rho0)
                ]
                if not v1.empty:
                    r = v1.iloc[0]
                    node = per_node[(r.report_id, "lifespan_aware", scen)]
                    c, fsd, _ = mission_fixed(
                        node,
                        r.capacity_tbit_day,
                        rho0,
                        days,
                        p_tx=ptx,
                        p_base=args.p_base,
                    )
                    rows.append(
                        base
                        | {
                            "algorithm": "lifespan_aware",
                            "day1_capacity_tbit": r.capacity_tbit_day,
                            "energy_neutral": r.max_source_duty <= rho0 + 1e-9,
                            "mission_volume_pbit": c.sum() / 1e3,
                            "first_source_death_years": np.nan
                            if fsd is None
                            else fsd / 365.25,
                        }
                    )

        res = pd.DataFrame([r for r in rows if r["scenario"] == scen])

        # --- (A) daily capacity vs P_tx, one curve per margin, with P_tx,max = M / rho_sat ---
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        lls_cap = s[s.algorithm == "lls"].capacity_tbit_day.mean()
        for i, m in enumerate(args.margins):
            r = res[
                (res.algorithm == "lls_budget_adaptive") & (res.margin_w == m)
            ].sort_values("p_tx")
            ax.plot(
                r.p_tx,
                r.day1_capacity_tbit,
                "o-",
                color=f"C{i}",
                label=f"margin {m:g} W",
            )
            ptx_max = m / rho_sat
            if args.p_tx[0] <= ptx_max <= args.p_tx[-1]:
                ax.axvline(ptx_max, color=f"C{i}", ls=":", lw=1)
        if not np.isnan(lls_cap):
            ax.axhline(
                lls_cap, color="k", ls="--", lw=1, label="LLS (unconstrained)"
            )
        ax.set_xlabel(r"OCT electrical power while transmitting $P_{tx}$ [W]")
        ax.set_ylabel("LLS-Budget capacity, day 1 [Tbit/day]")
        ax.set_title(
            rf"{scen}  (dotted: $P_{{tx}}^{{max}} = M/\rho_{{sat}}$, $\rho_{{sat}}$={rho_sat:.2f})",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        save(fig, f"capacity_vs_ptx_{scen}")

        # --- (B) mission data volume vs P_tx per margin: adaptive vs fixed d vs LLS unchanged ---
        fig, axes = plt.subplots(
            1,
            len(args.margins),
            figsize=(4 * len(args.margins), 4.2),
            sharey=False,
        )
        for ax, m in zip(np.atleast_1d(axes), args.margins):
            for alg, style in [
                ("lls_budget_adaptive", "o-"),
                ("lls_budget_fixed_d", "s--"),
                ("lls", "^:"),
                ("fcp", "v:"),
                ("lifespan_aware", "x:"),
            ]:
                r = res[
                    (res.algorithm == alg) & (res.margin_w == m)
                ].sort_values("p_tx")
                if not r.empty:
                    ax.plot(r.p_tx, r.mission_volume_pbit, style, label=alg)
            ax.set_yscale("log")
            ax.set_title(f"margin {m:g} W")
            ax.set_xlabel(r"$P_{tx}$ [W]")
        np.atleast_1d(axes)[0].set_ylabel("mission data volume [Pbit]")
        np.atleast_1d(axes)[-1].legend(fontsize=8)
        save(fig, f"mission_volume_vs_ptx_{scen}")

        # --- (C) heatmap: adaptive mission volume over (P_tx, margin) ---
        piv = res[res.algorithm == "lls_budget_adaptive"].pivot(
            index="margin_w", columns="p_tx", values="mission_volume_pbit"
        )
        fig, ax = plt.subplots(figsize=(7, 3.8))
        im = ax.imshow(
            piv.values, aspect="auto", origin="lower", cmap="viridis"
        )
        ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"{i:g}" for i in piv.index])
        for (yi, xi), v in np.ndenumerate(piv.values):
            ax.text(
                xi,
                yi,
                f"{v:.0f}",
                ha="center",
                va="center",
                color="w",
                fontsize=9,
            )
        ax.set_xlabel(r"$P_{tx}$ [W]")
        ax.set_ylabel(r"RTG margin $P_0-P_{base}$ [W]")
        ax.set_title(
            "LLS-Budget (adaptive) mission data volume [Pbit]", fontsize=11
        )
        fig.colorbar(im, ax=ax)
        save(fig, f"mission_volume_heatmap_{scen}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "ptx_sweep.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"rho_sat (95% of max LLS-Budget capacity): {rho_sat:.3f}")
    cols = [
        "p_tx",
        "margin_w",
        "rho0",
        "algorithm",
        "day1_capacity_tbit",
        "energy_neutral",
        "mission_volume_pbit",
        "first_source_death_years",
    ]
    print(out[cols].round(3).to_string(index=False))
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
