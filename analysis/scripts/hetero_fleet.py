"""
Heterogeneous fleets: how LLS-Budget redistributes traffic when nodes have different RTG margins.

Set A (mixed-age sources): a fixed ~50% subset of sources is "old" (margin ratio rho_old), the rest
"new" (rho_new). Set B (repurposed relays): relays at rho_relay, sources at rho_new.
For every run: capacity, per-class duty, per-class share of delivered source data, and whether each
class stays energy-neutral (duty <= its own rho). The unconstrained LLS/FCP plans (power-independent)
are checked against each fleet the same way. A homogeneous fleet with the same *mean* source margin
is interpolated from the rho sweep for comparison.

    uv run python analysis/scripts/hetero_fleet.py --report-ids $(cat output/hetero_ids.txt) \
        --rho-sweep output/plots/rho_sweep_all/sweep.csv
"""

import argparse
import json
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

from analysis.scripts.rho_sweep import iter_runs  # noqa: E402
from src.constants import PLOTS_ROOT, REPORTS_ROOT  # noqa: E402
from src.topology.weights import RELAY_IDS, SOURCE_IDS  # noqa: E402

plt.rcParams.update({"font.size": 13})
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

TBIT = 267_000 / 1e12  # internal capacity unit -> Tbit


def node_rho(ids, prof):
    out = {}
    for n in ids:
        if n in prof.get("p0_overrides", {}):
            out[n] = (prof["p0_overrides"][n] - prof["source_p_base"]) / prof[
                "p_tx"
            ]
        elif n in SOURCE_IDS:
            out[n] = prof["source_rho"]
        elif n in RELAY_IDS:
            out[n] = prof["relay_rho"]
    return out


def classify(n, prof):
    if n in RELAY_IDS:
        return "relay"
    if n in SOURCE_IDS:
        return (
            "old source" if n in prof.get("p0_overrides", {}) else "new source"
        )
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-ids", nargs="+", type=int, required=True)
    ap.add_argument(
        "--rho-sweep",
        default=None,
        help="sweep.csv of the homogeneous rho sweep",
    )
    ap.add_argument("--out-id", default="hetero_fleet")
    args = ap.parse_args()

    runs = []  # (report_id, profile, alg, scen, summary, capacity)
    for rid in args.report_ids:
        rdir = os.path.join(REPORTS_ROOT, str(rid))
        prof = json.load(open(os.path.join(rdir, "power_profile.json")))
        csv = pd.read_csv(os.path.join(rdir, f"{rid}_report.csv"))
        for alg, scen, summ in iter_runs(rid):
            alg = alg.replace("-", "_")
            row = csv[(csv.Algorithm == alg) & (csv.Scenario == scen)]
            cap = float(row.Capacity.iloc[0]) * TBIT * 86400 / summ["horizon"]
            runs.append((rid, prof, alg, scen, summ, cap))

    budget = [r for r in runs if r[2] == "lls_budget"]
    unconstrained = [r for r in runs if r[2] in ("lls", "fcp")]
    fleets = [(r[0], r[1], r[3]) for r in budget]

    def evaluate(prof, alg, scen, summ, cap, fleet_rid):
        ids = summ["node_ids"]
        duty = summ["tx_time"] / summ["horizon"]
        rho = node_rho(ids, prof)
        bits = summ["source_bits"]
        total_bits = bits.sum() or 1.0
        n_src = sum(1 for n in ids if n in SOURCE_IDS)
        n_old = len([n for n in ids if n in prof.get("p0_overrides", {})])
        mean_src_rho = np.mean([rho[n] for n in ids if n in SOURCE_IDS])
        setting = (
            "mixed sources"
            if prof.get("p0_overrides")
            else (
                "low-rho relays"
                if prof["relay_rho"] < prof["source_rho"]
                else "homogeneous"
            )
        )
        row = {
            "fleet_report": fleet_rid,
            "scenario": scen,
            "size": int(scen.rsplit("_", 1)[1]),
            "setting": setting,
            "algorithm": alg,
            "rho_new": prof["source_rho"],
            "rho_old": (
                min(rho[n] for n in prof["p0_overrides"] if n in rho)
                if n_old
                else np.nan
            ),
            "rho_relay": prof["relay_rho"],
            "n_sources": n_src,
            "n_old": n_old,
            "mean_source_rho": mean_src_rho,
            "capacity_tbit_day": cap,
        }
        for cls in ("old source", "new source", "relay"):
            members = [
                i for i, n in enumerate(ids) if classify(n, prof) == cls
            ]
            if not members:
                continue
            key = cls.replace(" ", "_")
            row[f"{key}_mean_duty"] = float(np.mean(duty[members]))
            row[f"{key}_max_duty"] = float(np.max(duty[members]))
            over = [i for i in members if duty[i] > rho[ids[i]] + 1e-9]
            row[f"{key}_over_margin"] = len(over)
            if cls != "relay":
                row[f"{key}_data_share"] = float(
                    bits[members].sum() / total_bits
                )
                row[f"{key}_pop_share"] = len(members) / n_src
        return row

    rows = [
        evaluate(prof, alg, scen, summ, cap, rid)
        for rid, prof, alg, scen, summ, cap in budget
    ]
    # unconstrained plans judged against every fleet definition of the same scenario
    for rid, prof, scen in fleets:
        for _, _, alg, uscen, summ, cap in unconstrained:
            if uscen == scen:
                rows.append(evaluate(prof, alg, scen, summ, cap, rid))
    df = pd.DataFrame(rows)

    if args.rho_sweep:
        sw = pd.read_csv(args.rho_sweep)
        sw = sw[sw.algorithm == "lls_budget"]
        hom = []
        for _, r in df[
            (df.algorithm == "lls_budget") & (df.setting == "mixed sources")
        ].iterrows():
            b = sw[sw.scenario == r.scenario].sort_values("rho")
            hom.append(
                np.interp(r.mean_source_rho, b.rho, b.capacity_tbit_day)
                if not b.empty
                else np.nan
            )
        df.loc[
            (df.algorithm == "lls_budget") & (df.setting == "mixed sources"),
            "homogeneous_same_mean_capacity",
        ] = hom

    out_dir = os.path.join(PLOTS_ROOT, args.out_id)
    for sub in ("pdf", "png"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    df.sort_values(
        ["setting", "size", "algorithm", "rho_old", "rho_relay"]
    ).to_csv(os.path.join(out_dir, "hetero_fleet.csv"), index=False)

    # Plot A: capacity and old-source data share vs rho_old
    a = df[(df.setting == "mixed sources") & (df.algorithm == "lls_budget")]
    if not a.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for i, (size, g) in enumerate(a.groupby("size")):
            g = g.sort_values("rho_old")
            ref = df[
                (df["size"] == size) & (df.algorithm == "lls")
            ].capacity_tbit_day.iloc[0]
            axes[0].plot(
                g.rho_old,
                g.capacity_tbit_day / ref,
                "o-",
                color=f"C{i}",
                label=f"{size} sources: mixed",
            )
            if "homogeneous_same_mean_capacity" in g:
                axes[0].plot(
                    g.rho_old,
                    g.homogeneous_same_mean_capacity / ref,
                    "x:",
                    color=f"C{i}",
                    label=f"{size}: homogeneous, same mean margin",
                )
            axes[1].plot(
                g.rho_old,
                g.old_source_data_share / g.old_source_pop_share,
                "o-",
                color=f"C{i}",
                label=f"{size} sources",
            )
        axes[0].set_xlabel(
            r"$\rho$ of the old sources (new sources: $\rho$=%g)"
            % a.rho_new.iloc[0]
        )
        axes[0].set_ylabel("capacity / unconstrained LLS")
        axes[0].legend(fontsize=8)
        axes[1].axhline(1, color="k", lw=0.8, ls="--")
        axes[1].set_xlabel(r"$\rho$ of the old sources")
        axes[1].set_ylabel("old sources' data share / population share")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pdf", "mixed_sources.pdf"))
        fig.savefig(os.path.join(out_dir, "png", "mixed_sources.png"), dpi=150)
        plt.close(fig)

    # Plot B: capacity vs relay rho
    b = df[
        (df.setting.isin(["low-rho relays", "homogeneous"]))
        & (df.algorithm == "lls_budget")
    ]
    if not b.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, (size, g) in enumerate(b.groupby("size")):
            g = g.sort_values("rho_relay")
            ref = df[
                (df["size"] == size) & (df.algorithm == "lls")
            ].capacity_tbit_day.iloc[0]
            ax.plot(
                g.rho_relay,
                g.capacity_tbit_day / ref,
                "o-",
                color=f"C{i}",
                label=f"{size} sources",
            )
        ax.set_xlabel(r"relay $\rho$ (sources: $\rho$=%g)" % b.rho_new.iloc[0])
        ax.set_ylabel("capacity / unconstrained LLS")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pdf", "low_rho_relays.pdf"))
        fig.savefig(
            os.path.join(out_dir, "png", "low_rho_relays.png"), dpi=150
        )
        plt.close(fig)

    pd.set_option("display.width", 250)
    cols = [
        "setting",
        "size",
        "algorithm",
        "rho_old",
        "rho_relay",
        "n_old",
        "capacity_tbit_day",
        "homogeneous_same_mean_capacity",
        "old_source_mean_duty",
        "new_source_mean_duty",
        "old_source_data_share",
        "old_source_pop_share",
        "old_source_over_margin",
        "new_source_over_margin",
        "relay_mean_duty",
        "relay_over_margin",
    ]
    cols = [c for c in cols if c in df]
    print(
        df.sort_values(
            ["setting", "size", "algorithm", "rho_old", "rho_relay"]
        )[cols]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
