"""
rho sweep analysis (v2 model).

Reads a set of reports produced with different power profiles (see scripts/run_rho_sweep.sh), replays
every scheduled TEG with the electrical transmit power recorded in each report's power_profile.json,
and produces:

1. Daily capacity vs. margin ratio rho = (P0 - P_base) / P_tx.
2. Mission data volume vs. initial margin rho0, i.e. the data delivered over the whole mission when the
   24 h scenario is repeated every day while the RTG decays:
     - lls_budget (adaptive): each day is scheduled with that day's rho(t); daily capacity is
       interpolated from the lls_budget runs of the sweep.
     - unconstrained schedulers (lls, fcp, ...): the same schedule repeats every day, so each node keeps
       its measured duty cycle d_n. A daily battery simulation decides when each node dies
       (RTG surplus below P_base + d_n P_tx and battery empty). Daily capacity is scaled by the share of
       source data carried by the sources still alive. This ignores re-planning after a node dies,
       so it underestimates the unconstrained schedulers once nodes start dying.
3. Daily capacity over mission time for each rho0.

Usage:
    uv run python analysis/scripts/rho_sweep.py --report-ids <id> <id> ... [--rho0 0.25 0.5 1.25]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import PLOTS_ROOT, REPORTS_ROOT  # noqa: E402
from src.topology.weights import (  # noqa: E402
    RELAY_IDS,
    SOURCE_IDS,
)
from analysis.scripts.utils import load_report_tegs  # noqa: E402
from src.reporting.run_summary import load_summary, summarize_teg  # noqa: E402

plt.rcParams.update({"font.size": 14})
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

DAY = 86_400.0
BUDGET_ALGORITHMS = {"lls_budget"}


def replay(teg) -> dict:
    """Per-node transmit time [s] and per-source delivered bits over the scenario."""
    return summarize_teg(teg)


def iter_runs(report_id: int):
    """Yield (algorithm, scenario, summary) for every run in a report.

    Reads `<alg>_<scenario>.summary.npz` (main.py --store summary) when present, otherwise
    replays the scheduled TEG pickle.
    """
    rdir = os.path.join(REPORTS_ROOT, str(report_id))
    seen = set()
    for file_name in sorted(os.listdir(rdir)):
        if file_name.endswith(".summary.npz"):
            stem = file_name[: -len(".summary.npz")]
            alg, _, scen = stem.partition("_")
            seen.add(stem)
            yield alg, scen, load_summary(os.path.join(rdir, file_name))
    for alg, scen, teg in load_report_tegs(report_id):
        if f"{alg}_{scen}" in seen:
            continue
        yield alg, scen, replay(teg)


def load_runs(report_ids: list[int]) -> tuple[pd.DataFrame, dict]:
    rows, per_node = [], {}
    for rid in report_ids:
        rdir = os.path.join(REPORTS_ROOT, str(rid))
        with open(os.path.join(rdir, "power_profile.json")) as f:
            prof = json.load(f)
        csv = pd.read_csv(os.path.join(rdir, f"{rid}_report.csv"))
        for alg, scen, r in iter_runs(rid):
            alg = alg.replace("-", "_")
            cap_row = csv[
                (csv["Algorithm"] == alg) & (csv["Scenario"] == scen)
            ]
            if cap_row.empty:
                continue
            ids = r["node_ids"]
            is_src = np.array([i in SOURCE_IDS for i in ids])
            is_rel = np.array([i in RELAY_IDS for i in ids])
            duty = r["tx_time"] / r["horizon"]
            rows.append(
                {
                    "report_id": rid,
                    "algorithm": alg,
                    "scenario": scen,
                    "rho": prof["source_rho"],
                    "relay_rho": prof["relay_rho"],
                    "p_tx": prof["p_tx"],
                    "p_base": prof["source_p_base"],
                    # internal capacity unit -> Tbit (same conversion as visualize_cap_metrics.py)
                    "capacity_tbit_day": float(cap_row["Capacity"].iloc[0])
                    * 267_000
                    / 1e12
                    * DAY
                    / r["horizon"],
                    "mean_source_duty": float(duty[is_src].mean()),
                    "max_source_duty": float(duty[is_src].max()),
                    "mean_relay_duty": float(duty[is_rel].mean())
                    if is_rel.any()
                    else 0.0,
                    "max_relay_duty": float(duty[is_rel].max())
                    if is_rel.any()
                    else 0.0,
                    "sources_over_margin": int(
                        np.sum(duty[is_src] > prof["source_rho"] + 1e-9)
                    ),
                }
            )
            per_node[(rid, alg, scen)] = {
                "duty": duty,
                "is_src": is_src,
                "is_rel": is_rel,
                "source_bits": r["source_bits"],
                "profile": prof,
            }
    return pd.DataFrame(rows), per_node


def rho_t(
    t_s: np.ndarray, rho0: float, p_base: float, p_tx: float, lam: float
):
    p0 = p_base + rho0 * p_tx
    return (p0 * np.exp(-lam * t_s) - p_base) / p_tx


def mission_budget(
    sweep: pd.DataFrame,
    scen: str,
    rho0: float,
    prof: dict,
    days: np.ndarray,
    p_tx: float | None = None,
    p_base: float | None = None,
):
    """Adaptive budget: each day is scheduled at that day's rho(t). With a fixed data rate the
    lls_budget capacity depends on rho only, so one rho sweep serves every (P_tx, P_base)."""
    s = sweep[
        (sweep.algorithm.isin(BUDGET_ALGORITHMS)) & (sweep.scenario == scen)
    ]
    s = s.sort_values("rho")
    rho = rho_t(
        days * DAY,
        rho0,
        prof["source_p_base"] if p_base is None else p_base,
        prof["p_tx"] if p_tx is None else p_tx,
        prof["decay_rate_per_s"],
    )
    cap = np.interp(rho, s.rho.values, s.capacity_tbit_day.values, left=0.0)
    cap[rho <= 0] = 0.0
    return cap


def mission_fixed(
    node: dict,
    cap_day: float,
    rho0: float,
    days: np.ndarray,
    p_tx: float | None = None,
    p_base: float | None = None,
):
    """Repeat one fixed daily schedule while the RTG decays; nodes die when their battery empties.

    Duty cycles are dimensionless and the data rate is fixed, so a schedule measured under one
    power profile can be replayed under another by passing `p_tx` / `p_base`.
    """
    prof = node["profile"]
    p_tx = prof["p_tx"] if p_tx is None else p_tx
    p_base = prof["source_p_base"] if p_base is None else p_base
    lam = prof["decay_rate_per_s"]
    cap_j = prof["battery_wh"] * 3600.0
    duty = node["duty"]
    sc = node["is_src"] | node["is_rel"]
    load = p_base + duty[sc] * p_tx
    battery = np.full(sc.sum(), cap_j)
    alive = np.ones(sc.sum(), dtype=bool)
    p0 = p_base + rho0 * p_tx
    bits = node["source_bits"][sc]
    share = bits / bits.sum() if bits.sum() > 0 else np.zeros_like(bits)
    is_src = node["is_src"][sc]
    is_rel = node["is_rel"][sc]
    out = np.zeros(len(days))
    first_src_death = first_rel_death = None
    for i, d in enumerate(days):
        p = p0 * np.exp(-lam * d * DAY)
        battery = np.clip(battery + (p - load) * DAY, 0.0, cap_j)
        died = alive & (battery <= 0.0)
        if died.any():
            if first_src_death is None and (died & is_src).any():
                first_src_death = d
            if first_rel_death is None and (died & is_rel).any():
                first_rel_death = d
        alive &= battery > 0.0
        out[i] = cap_day * share[alive].sum()
        if not alive[is_src].any():
            break
    return out, first_src_death, first_rel_death


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-ids", nargs="+", type=int, required=True)
    ap.add_argument("--rho0", nargs="+", type=float, default=[0.25, 0.5, 1.25])
    ap.add_argument("--out-id", default=None)
    args = ap.parse_args()

    sweep, per_node = load_runs(args.report_ids)
    out_id = args.out_id or f"rho_sweep_{int(time.time())}"
    out_dir = os.path.join(PLOTS_ROOT, out_id)
    os.makedirs(os.path.join(out_dir, "pdf"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "png"), exist_ok=True)
    sweep.sort_values(["scenario", "algorithm", "rho"]).to_csv(
        os.path.join(out_dir, "sweep.csv"), index=False
    )

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pdf", f"{name}.pdf"))
        fig.savefig(os.path.join(out_dir, "png", f"{name}.png"), dpi=150)
        plt.close(fig)

    mission_rows = []
    for scen in sorted(sweep.scenario.unique()):
        s = sweep[sweep.scenario == scen]
        # --- (1) daily capacity vs rho ---
        fig, ax = plt.subplots(figsize=(7, 4.5))
        b = s[s.algorithm.isin(BUDGET_ALGORITHMS)].sort_values("rho")
        ax.plot(
            b.rho, b.capacity_tbit_day, "o-", color="C0", label="LLS-Budget"
        )
        for i, alg in enumerate(sorted(set(s.algorithm) - BUDGET_ALGORITHMS)):
            rows = s[s.algorithm == alg]
            c = rows.capacity_tbit_day.mean()
            # An unconstrained schedule is energy-neutral only if rho >= its busiest source's duty cycle.
            d_max = rows.max_source_duty.max()
            ax.axhline(
                c,
                ls="--",
                lw=1,
                color=f"C{i + 1}",
                label=f"{alg} (energy-neutral for rho >= {d_max:.2f})",
            )
        ax.set_xlabel(r"margin ratio $\rho=(P_0-P_{base})/P_{tx}$")
        ax.set_ylabel("capacity [Tbit/day]")
        ax.set_title(scen)
        ax.legend(fontsize=10)
        save(fig, f"capacity_vs_rho_{scen}")

        # --- (2)/(3) mission integration ---
        if b.empty:
            continue
        prof = per_node[(b.report_id.iloc[0], b.algorithm.iloc[0], scen)][
            "profile"
        ]
        lam = prof["decay_rate_per_s"]
        for rho0 in args.rho0:
            p0 = prof["source_p_base"] + rho0 * prof["p_tx"]
            t_hw_days = np.log(p0 / prof["source_p_base"]) / lam / DAY
            days = np.arange(0, int(t_hw_days) + 400, 1.0)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            cap_b = mission_budget(sweep, scen, rho0, prof, days)
            ax.plot(days / 365.25, cap_b, label="LLS-Budget")
            mission_rows.append(
                {
                    "scenario": scen,
                    "rho0": rho0,
                    "algorithm": "lls_budget",
                    "mission_volume_pbit": cap_b.sum() / 1e3,
                    "first_source_death_days": None,
                    "first_relay_death_days": None,
                    "day1_capacity_tbit": cap_b[0],
                }
            )
            # Constant-duty policy (director's d* proposal): pick one daily budget d_c * P_tx * 24 h for the
            # whole mission (the lls_budget run at rho = d_c) and keep it while the RTG decays.
            best = None
            for _, brow in b[b.rho > 0].iterrows():
                node = per_node[(brow.report_id, brow.algorithm, scen)]
                cap_c, fsd_c, _ = mission_fixed(
                    node, brow.capacity_tbit_day, rho0, days
                )
                if best is None or cap_c.sum() > best[1].sum():
                    best = (brow.rho, cap_c, fsd_c)
            if best is not None:
                ax.plot(
                    days / 365.25,
                    best[1],
                    label=f"LLS-Budget, fixed d={best[0]}",
                )
                mission_rows.append(
                    {
                        "scenario": scen,
                        "rho0": rho0,
                        "algorithm": f"lls_budget_fixed_d{best[0]}",
                        "mission_volume_pbit": best[1].sum() / 1e3,
                        "first_source_death_days": best[2],
                        "first_relay_death_days": None,
                        "day1_capacity_tbit": best[1][0],
                    }
                )
            for alg in sorted(set(s.algorithm) - BUDGET_ALGORITHMS):
                row = s[s.algorithm == alg].iloc[0]
                node = per_node[(row.report_id, alg, scen)]
                cap_f, fsd, frd = mission_fixed(
                    node, row.capacity_tbit_day, rho0, days
                )
                ax.plot(days / 365.25, cap_f, label=alg)
                mission_rows.append(
                    {
                        "scenario": scen,
                        "rho0": rho0,
                        "algorithm": alg,
                        "mission_volume_pbit": cap_f.sum() / 1e3,
                        "first_source_death_days": fsd,
                        "first_relay_death_days": frd,
                        "day1_capacity_tbit": cap_f[0],
                    }
                )
            ax.set_xlabel("mission time [years]")
            ax.set_ylabel("capacity [Tbit/day]")
            ax.set_title(rf"{scen}, $\rho_0$={rho0}")
            ax.legend(fontsize=10)
            save(fig, f"mission_timeline_{scen}_rho0_{rho0}")

    mission = pd.DataFrame(mission_rows)
    mission.to_csv(os.path.join(out_dir, "mission_volume.csv"), index=False)
    for scen in mission.scenario.unique() if not mission.empty else []:
        m = mission[mission.scenario == scen]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        m = m.assign(
            algorithm=m.algorithm.where(
                ~m.algorithm.str.startswith("lls_budget_fixed"),
                "lls_budget (best fixed d)",
            )
        )
        for alg in m.algorithm.unique():
            mm = m[m.algorithm == alg].sort_values("rho0")
            ax.plot(mm.rho0, mm.mission_volume_pbit, "o-", label=alg)
        ax.set_xlabel(r"initial margin ratio $\rho_0$")
        ax.set_ylabel("mission data volume [Pbit]")
        ax.set_yscale("log")
        ax.set_title(scen)
        ax.legend(fontsize=10)
        save(fig, f"mission_volume_{scen}")

    pd.set_option("display.width", 200)
    print(
        sweep.sort_values(["scenario", "algorithm", "rho"]).to_string(
            index=False
        )
    )
    print()
    print(mission.to_string(index=False))
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
