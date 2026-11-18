import csv
import os
import pickle
import time

from src import constants
from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
)
from src.topology.weights import EVAL_EFFECTIVE_CONTACT
from src.topology.weights import (
    compute_capacity,
    compute_jains_fairness_index,
    compute_node_capacities,
    compute_scheduled_delay,
    compute_wasted_buffer,
    compute_wasted_capacity,
)
from src.utils import FileType
from src.reporting.run_summary import save_summary, summarize_teg


class Reporter:
    def __init__(self, write_pkl: bool, write_summary: bool = False):
        self.reports = []
        self.time_expanded_graph_data = []
        self.summaries = []
        self.write_pkl = write_pkl
        self.write_summary = write_summary

    def generate_report(
        self,
        experiment_name: str,
        scheduler_name: str,
        duration: float,
        teg: TimeExpandedGraph,
    ) -> dict[str, str | float | int]:
        name = f"{scheduler_name.replace('_', '-')}_{experiment_name}"
        if self.write_pkl:
            self.time_expanded_graph_data.append({"name": name, "teg": teg})
        if self.write_summary:
            # Summarize now so the TEG can be garbage-collected (sweeps hold many runs).
            self.summaries.append(
                {"name": name, "summary": summarize_teg(teg)}
            )

        # EVAL_EFFECTIVE_CONTACT.clear()
        node_capacities = compute_node_capacities(
            teg.graphs,
            teg.state_durations,
            teg.K,
            teg.nodes,
            teg.graphs,
            teg.pos,
            teg.optical_interfaces_to_node,
            teg.node_to_optical_interfaces,
            False,
        )

        # for cap in node_capacities:
        #     print(cap.id, min(cap.capacity_in, cap.capacity_out))
        #
        # for (k, tx, rx, d), t in weights.EVAL_EFFECTIVE_CONTACT.items():
        #     print(k, tx, rx, t, d)

        network_capacity = int(compute_capacity(node_capacities))
        network_wasted_capacity = compute_wasted_capacity(node_capacities)

        wasted_buffer_capacity = int(compute_wasted_buffer(node_capacities))

        try:
            jains_fairness_index = compute_jains_fairness_index(
                teg.graphs,
                teg.state_durations,
                teg.nodes,
                teg.K,
                teg.N,
                teg.optical_interfaces_to_node,
            )
        except RuntimeError:
            # Undefined when nothing was scheduled (e.g. lls_budget with rho <= 0).
            jains_fairness_index = float("nan")

        scheduled_delay = compute_scheduled_delay(
            teg.graphs,
            teg.state_durations,
            teg.nodes,
            teg.K,
            teg.N,
            teg.optical_interfaces_to_node,
        )

        row = [
            scheduler_name,
            experiment_name,
            duration,
            network_capacity,
            network_wasted_capacity,
            wasted_buffer_capacity,
            jains_fairness_index,
            scheduled_delay,
        ]
        self.reports.append(row)

        return {
            "scheduler_name": scheduler_name,
            "experiment_name": experiment_name,
            "duration": duration,
            "network_capacity": network_capacity,
            "network_wasted_capacity": network_wasted_capacity,
            "wasted_buffer_capacity": wasted_buffer_capacity,
            "jains_fairness_index": jains_fairness_index,
            "scheduled_delay": scheduled_delay,
        }

    def write_report(self) -> int:
        # Create CSV of runtimes, capacity, and wasted capacity
        basic_report_headers = [
            "Algorithm",
            "Scenario",
            "Execution duration",
            "Capacity",
            "Wasted capacity",
            "Wasted buffer capacity",
            "Jain's fairness index",
            "Scheduled delay",
        ]

        report_id = int(time.time())

        os.makedirs(constants.REPORTS_ROOT, exist_ok=True)

        # Back-to-back or parallel runs (e.g. a rho sweep) can finish within the same second;
        # mkdir is atomic, so claim the first free id.
        while True:
            report_dir = os.path.join(constants.REPORTS_ROOT, f"{report_id}")
            try:
                os.mkdir(report_dir)
                break
            except FileExistsError:
                report_id += 1

        report_path = str(
            os.path.join(
                report_dir, f"{report_id}_{FileType.REPORT.value}.csv"
            )
        )
        with open(report_path, "w") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            writer.writerow(basic_report_headers)
            writer.writerows(self.reports)

        for item in self.summaries:
            save_summary(
                os.path.join(report_dir, f"{item['name']}.summary.npz"),
                item["summary"],
            )

        if not self.write_pkl:
            return report_id

        # Save each teg as a pkl file to the reports directory so that it can be used for analysis later
        for teg_data_dict in self.time_expanded_graph_data:
            pkl_path = str(
                os.path.join(report_dir, f"{teg_data_dict['name']}.pkl")
            )
            with open(pkl_path, "wb") as f:
                pickle.dump(teg_data_dict["teg"], f)

        return report_id
