from timeit import default_timer as timer
import copy
import traceback
import typer

import numpy as np

from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
    convert_time_expanded_graph_to_contact_plan,
    write_time_expanded_graph,
    get_time_expanded_graph,
)
from src.models.pointing_delay import RETARGETING_DELAY_CACHE
from src.reporting.report_generator import Reporter
from src.schedulers import SCHEDULER_REGISTER
from src.topology.contact_plan import (
    IONContactPlanParser,
    IPNDContactPlanParser,
)
from src.utils import FileType, RunTablePrinter, make_fanout_progress_callback
from src import constants
from src.power_profile import (
    apply_power_profile,
    current_power_profile,
    write_power_profile,
)
import os


def experiment_driver(
    experiment_name: str,
    scheduler_name: str,
    reporter: Reporter,
    time_expanded_graph: TimeExpandedGraph,
    run_table: RunTablePrinter,
) -> dict[str, str | float | int]:
    start = timer()
    contact_plan_parser = IONContactPlanParser()

    try:
        if scheduler_name not in SCHEDULER_REGISTER:
            raise ValueError(f"Unknown scheduler name: {scheduler_name}")

        progress_callback = run_table.make_progress_callback(
            experiment_name, scheduler_name
        )
        scheduler_input_teg = copy.deepcopy(time_expanded_graph)

        teg = (
            scheduler_input_teg.dag_reduction(progress_callback)
            if scheduler_name in ["lls_lp", "lls_mip"]
            else scheduler_input_teg
        )
        if scheduler_name not in ["lls_lp", "lls_mip"]:
            progress_callback("schedule", 0, 1)

        scheduled_time_expanded_graph = SCHEDULER_REGISTER[
            scheduler_name
        ].schedule(teg, progress_callback)

        # We deactivate this for now, I don't have interest in visualize the CT. With the reports is enough.
        if False:
            # Convert the TEG back to a contact plan
            scheduled_contact_plan = (
                convert_time_expanded_graph_to_contact_plan(
                    scheduled_time_expanded_graph,
                    progress_callback,
                )
            )
            contact_plan_parser.write(
                experiment_name,
                scheduled_contact_plan,
                FileType.CONTACT_PLAN_SCHEDULED,
            )

            # Write contact plan to disk as IPN-D contact plan, so we can visualize the output
            ipnd_contact_plan_parser = IPNDContactPlanParser()
            ipnd_contact_plan_parser.write(
                experiment_name, scheduled_contact_plan
            )

        progress_callback("report", 1, 1)
        run_data = reporter.generate_report(
            experiment_name,
            scheduler_name,
            timer() - start,
            scheduled_time_expanded_graph,
        )
        return run_data
    except Exception as e:
        traceback.print_exc()
        if scheduler_name == "lls_mip":
            raise e
        return {
            "progress": "failed",
            "duration": timer() - start,
            "network_capacity": 0,
            "network_wasted_capacity": 0,
            "wasted_buffer_capacity": 0,
            "jains_fairness_index": 0.0,
            "scheduled_delay": 0.0,
        }


def multi_experiment_driver(
    experiment_names: list[str],
    scheduler_names: list[str],
    run_table: RunTablePrinter,
    store: str = "teg",
):
    reporter = Reporter(
        write_pkl=store in ("teg", "both"),
        write_summary=store in ("summary", "both"),
    )

    for experiment_name in experiment_names:
        # Run table progress setup.
        build_callbacks = []
        for scheduler_name in scheduler_names:
            run_table.update_progress(experiment_name, scheduler_name, "0%")
            build_callbacks.append(
                run_table.make_progress_callback(
                    experiment_name, scheduler_name
                )
            )
        build_progress_callback = make_fanout_progress_callback(
            build_callbacks
        )

        # Build TEG.

        # Before build check if the TEG was already built and cached, to avoid rebuilding the TEG every time.
        try:
            time_expanded_graph = get_time_expanded_graph(
                experiment_name, FileType.TEG
            )
        except FileNotFoundError:
            contact_plan_parser = IONContactPlanParser()
            contact_plan = contact_plan_parser.read(experiment_name)
            time_expanded_graph = TimeExpandedGraph.from_contact_plan(
                contact_plan=contact_plan,
                should_fractionate=True,
                progress_callback=build_progress_callback,
            )
            write_time_expanded_graph(
                experiment_name, time_expanded_graph, FileType.TEG
            )
        else:
            _ = [
                run_table.update_progress(
                    experiment_name, scheduler_name, "40%"
                )
                for scheduler_name in scheduler_names
            ]

        for scheduler_name in scheduler_names:
            # Reset caches before each scheduler run since scheduling/reporting uses global caches.
            RETARGETING_DELAY_CACHE.clear()
            run_data = experiment_driver(
                experiment_name,
                scheduler_name,
                reporter,
                time_expanded_graph,
                run_table,
            )
            run_table.update_result(experiment_name, scheduler_name, run_data)

    report_id = reporter.write_report()
    write_power_profile(os.path.join(constants.REPORTS_ROOT, f"{report_id}"))
    print(f"\nreport_id={report_id}")


app = typer.Typer()


@app.command()
def main(
    experiment_names: list[str] = typer.Option(
        ...,
        "--experiment-names",
        "-e",
        help="Name of experiment folder",
    ),
    scheduler_names: list[str] = typer.Option(
        ...,
        "--scheduler-names",
        "-s",
        help="Name of scheduler algorithm to use",
    ),
    plain_progress: bool = typer.Option(
        False,
        "--debug",
        help="Disable the live run table. Useful when debugging with pdb/ipdb.",
    ),
    p_tx: float | None = typer.Option(
        None, "--p-tx", help="Electrical OCT power while transmitting [W]."
    ),
    p_base: float | None = typer.Option(
        None,
        "--p-base",
        help="Spacecraft baseline load [W] (sources and relays).",
    ),
    p0: float | None = typer.Option(
        None, "--p0", help="Initial RTG power [W] (sources and relays)."
    ),
    rho: float | None = typer.Option(
        None,
        "--rho",
        help="Margin ratio (P0 - P_base) / P_tx; sets P0 from P_base and P_tx.",
    ),
    relay_rho: float | None = typer.Option(
        None,
        "--relay-rho",
        help="Margin ratio for relays only (overrides --rho).",
    ),
    battery_wh: float | None = typer.Option(
        None, "--battery-wh", help="Battery capacity per spacecraft [Wh]."
    ),
    battery_reserve: float | None = typer.Option(
        None,
        "--battery-reserve",
        help="Fraction of the battery lls_budget never plans to use (1.0 = energy-neutral).",
    ),
    old_source_rho: float | None = typer.Option(
        None,
        "--old-source-rho",
        help="Mixed-age fleet: margin ratio of the 'old' sources (the rest keep --rho).",
    ),
    old_source_fraction: float = typer.Option(
        0.5,
        "--old-source-fraction",
        help="Expected share of sources that are 'old' (fixed pseudo-random subset).",
    ),
    store: str = typer.Option(
        "teg",
        "--store",
        help="What each report keeps per run: 'teg' (scheduled TEG pickle, default), "
        "'summary' (per-node transmit time and source data, src/reporting/run_summary.py) or 'both'.",
    ),
):
    np.random.seed(42)
    apply_power_profile(
        p_base=p_base,
        p_tx=p_tx,
        p0=p0,
        rho=rho,
        relay_rho=relay_rho,
        battery_wh=battery_wh,
        battery_reserve=battery_reserve,
        old_source_rho=old_source_rho,
        old_source_fraction=old_source_fraction,
    )
    print(f"power profile: {current_power_profile()}")
    with RunTablePrinter(
        experiment_names,
        scheduler_names,
        enable_live=not plain_progress,
    ) as run_table:
        multi_experiment_driver(
            experiment_names, scheduler_names, run_table, store
        )


if __name__ == "__main__":
    app()
