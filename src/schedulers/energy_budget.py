import numpy as np

from src import constants
from src.constants import MLConfig, OCTConfig
from src.models import generated_energy, transmission_energy
from src.time_expanded_graph.time_expanded_graph import TimeExpandedGraph
from src.topology.weights import (
    compute_effective_contact_time,
    compute_node_capacity_by_graph,
    delta_capacity,
    disabled_contact_time,
    merge_many_node_capacities,
)
from src.utils import ProgressCallback
from .base_scheduler import BaseScheduler


class EnergyBudgetScheduler(BaseScheduler):
    """
    LLS with a per-node energy budget (v2 model: throughput is the objective, energy is the constraint).

    The edge weight is the LLS weight (delta capacity). Energy only enters as a feasibility filter:
    at the start of every budget period each spacecraft n receives

        B_n = max(0, E_gen,n(period) - P_base, n * period  +  max(0, E_b,n - reserve * E_b0,n))

    i.e. the RTG surplus over its baseline load plus the battery energy it is allowed to plan with.
    At each state k, an edge is a candidate only if both endpoints can still afford it
    (cost = P_tx * effective contact time, charged to each spacecraft endpoint); the blossom matching
    then picks the capacity-maximizing contacts among the affordable ones, and accepted contacts are
    charged against the budgets. Edges with no capacity gain are never scheduled, so the scheduler can
    say "do not transmit". The battery is carried across periods: B_n unspent tops the battery up,
    overspending (only possible with reserve < 1) drains it.

    With an unbounded budget (rho >> 1) this reduces to LLS minus its zero-gain contacts.
    """

    def __init__(
        self, should_bypass_retargeting_time=False, drop_zero_gain=True
    ):
        super().__init__(should_bypass_retargeting_time)
        self.drop_zero_gain = drop_zero_gain
        # Per-run diagnostics, read by tests/analysis.
        self.last_run_stats: dict = {}

    def _replenish(
        self,
        period_start: float,
        period_end: float,
        initial_powers: np.ndarray,
        baseline_powers: np.ndarray,
        battery: np.ndarray,
        battery_max: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (budget, surplus) in joules for the period [period_start, period_end]."""
        spacecraft = np.isfinite(initial_powers)
        surplus = np.full(initial_powers.shape, np.inf)
        surplus[spacecraft] = generated_energy(
            from_time=period_start,
            to_time=period_end,
            initial_power=initial_powers[spacecraft],
            decay_constant=MLConfig.DECAY_RATE,
        ) - baseline_powers[spacecraft] * (period_end - period_start)

        budget = np.full(initial_powers.shape, np.inf)
        reserve = MLConfig.BATTERY_RESERVE_FRACTION * battery_max[spacecraft]
        usable_battery = np.maximum(battery[spacecraft] - reserve, 0.0)
        budget[spacecraft] = np.maximum(
            surplus[spacecraft] + usable_battery, 0.0
        )
        return budget, surplus

    def schedule(
        self,
        teg: TimeExpandedGraph,
        progress_callback: ProgressCallback | None = None,
    ) -> TimeExpandedGraph:
        scheduled_graphs = np.zeros((teg.K, teg.N, teg.N), dtype="int64")
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        node_capacities = []
        W_dct = np.zeros((teg.N, teg.N), dtype="int64")

        node_ids = np.array([node.id for node in teg.nodes])
        oi_to_node = np.array(
            [teg.optical_interfaces_to_node[i] for i in range(teg.N)],
            dtype=int,
        )
        initial_powers = MLConfig.get_initial_powers(node_ids).astype(float)
        baseline_powers = MLConfig.get_baseline_powers(node_ids).astype(float)
        battery_max = (
            MLConfig.get_initial_batteries(node_ids).astype(float) * 3600.0
        )
        battery = battery_max.copy()
        spacecraft = np.isfinite(initial_powers)
        p_tx = OCTConfig.ELECTRICAL_TX_POWER

        horizon = float(np.sum(teg.state_durations))
        period = float(MLConfig.BUDGET_PERIOD)

        period_idx = -1
        budget = np.zeros(len(teg.nodes))
        surplus = np.zeros(len(teg.nodes))
        spent = np.zeros(len(teg.nodes))
        spent_total = np.zeros(len(teg.nodes))
        rejected_for_budget = 0
        accumulated_time = 0.0

        for k in range(teg.K):
            state_duration = teg.state_durations[k]

            # --- Budget period bookkeeping ---
            new_period_idx = int(accumulated_time // period)
            if new_period_idx != period_idx:
                if period_idx >= 0:
                    # Close previous period: battery absorbs the unspent surplus / covers the overspend.
                    battery[spacecraft] = np.clip(
                        battery[spacecraft]
                        + surplus[spacecraft]
                        - spent[spacecraft],
                        0.0,
                        battery_max[spacecraft],
                    )
                period_idx = new_period_idx
                period_start = period_idx * period
                period_end = min(period_start + period, horizon)
                budget, surplus = self._replenish(
                    period_start,
                    period_end,
                    initial_powers,
                    baseline_powers,
                    battery,
                    battery_max,
                )
                spent = np.zeros(len(teg.nodes))

            remaining = budget - spent

            # --- LLS weight: delta capacity ---
            W_delta_cap = delta_capacity(
                teg.graphs[k],
                scheduled_graphs[:k],
                node_capacities,
                teg.nodes,
                state_duration,
                teg.pos,
                teg.optical_interfaces_to_node,
                self.should_bypass_retargeting_time,
            )
            weights[k] = ((1 - constants.ALPHA) * W_delta_cap) + (
                constants.ALPHA * W_dct
            )

            # --- Energy feasibility filter ---
            sym_w = weights[k] + weights[k].T
            candidate = np.triu(teg.graphs[k] >= 1, k=1)
            if self.drop_zero_gain:
                candidate &= sym_w > 0
            cost = np.zeros((teg.N, teg.N))
            feasible = np.zeros((teg.N, teg.N), dtype=bool)
            for a, b in zip(*np.where(candidate)):
                t_eff = compute_effective_contact_time(
                    oi_idx1=a,
                    oi_idx2=b,
                    scheduled_contact_topology=scheduled_graphs[:k],
                    state_duration=state_duration,
                    positions=teg.pos,
                    optical_interfaces_to_node=teg.optical_interfaces_to_node,
                    nodes=teg.nodes,
                    should_bypass_retargeting_time=self.should_bypass_retargeting_time,
                )
                c = transmission_energy(power=p_tx, duration=t_eff)
                cost[a, b] = cost[b, a] = c
                na, nb = oi_to_node[a], oi_to_node[b]
                if remaining[na] >= c and remaining[nb] >= c:
                    feasible[a, b] = feasible[b, a] = True
                else:
                    rejected_for_budget += 1

            P_k = np.where(feasible, teg.graphs[k], 0)
            matched_edges = self._blossom(P_k, weights[k])

            # A node with several optical interfaces can appear in more than one matched edge;
            # charge edges in decreasing weight order and drop the ones it can no longer afford.
            accepted = set()
            for a, b in sorted(
                matched_edges, key=lambda e: -float(sym_w[e[0], e[1]])
            ):
                na, nb = oi_to_node[a], oi_to_node[b]
                c = cost[a, b]
                if remaining[na] >= c and remaining[nb] >= c:
                    for n in {na, nb}:
                        if spacecraft[n]:
                            spent[n] += c
                            spent_total[n] += c
                    remaining = budget - spent
                    accepted.add((a, b))
                else:
                    rejected_for_budget += 1

            L_k, contacts = self._build_graph(
                accepted, teg.graphs[k], teg.contacts[k], teg.node_map
            )
            scheduled_graphs[k] = L_k
            scheduled_contacts.append(contacts)

            scheduled_node_capacities = compute_node_capacity_by_graph(
                L_k,
                state_duration,
                teg.nodes,
                scheduled_graphs[:k],
                teg.pos,
                teg.optical_interfaces_to_node,
                teg.node_to_optical_interfaces,
                self.should_bypass_retargeting_time,
            )
            node_capacities = merge_many_node_capacities(
                node_capacities + scheduled_node_capacities
            )
            W_dct += disabled_contact_time(teg.graphs[k], L_k, state_duration)

            accumulated_time += state_duration
            if progress_callback is not None:
                progress_callback("schedule", k + 1, teg.K)

        self.last_run_stats = {
            "tx_energy_wh": spent_total / 3600.0,
            "rejected_for_budget": rejected_for_budget,
        }

        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=teg.K,
            N=teg.N,
            nodes=teg.nodes,
            node_map=teg.node_map,
            W=weights,
            pos=teg.pos,
            optical_interfaces_to_node=teg.optical_interfaces_to_node,
            node_to_optical_interfaces=teg.node_to_optical_interfaces,
            effective_contact_durations=teg.effective_contact_durations,
        )
