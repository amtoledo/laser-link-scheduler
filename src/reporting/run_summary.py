"""
Compact per-run summary of a scheduled TEG: what the energy analyses need, without the full TEG.

A scheduled TEG pickle is 3-30 MB per (scheduler, scenario); a sweep over rho, P_tx or epochs
writes hundreds of them. The mission-time analyses only need, per node, the effective transmit time
and the data each source delivered, so `main.py --store summary` writes this instead.
"""

import numpy as np

from src import constants
from src.topology.weights import (
    COORDINATE_CACHE,
    EFFECTIVE_CONTACT_TIME_CACHE,
    RELAY_OR_DESTINATION_IDS,
    SOURCE_IDS,
    compute_effective_contact_time,
)


def summarize_teg(teg) -> dict:
    """Per-node transmit time [s] and per-source delivered bits over the scenario."""
    EFFECTIVE_CONTACT_TIME_CACHE.clear()
    COORDINATE_CACHE.clear()
    n_nodes = len(teg.nodes)
    tx_time = np.zeros(n_nodes)
    source_bits = np.zeros(n_nodes)
    oi_to_node = teg.optical_interfaces_to_node
    for k in range(teg.K):
        active_tx, active_rx = np.where(teg.graphs[k] >= 1)
        for a, b in zip(active_tx.tolist(), active_rx.tolist()):
            t_eff = compute_effective_contact_time(
                oi_idx1=a,
                oi_idx2=b,
                scheduled_contact_topology=teg.graphs[:k],
                state_duration=int(teg.state_durations[k]),
                positions=teg.pos,
                optical_interfaces_to_node=oi_to_node,
                nodes=teg.nodes,
                should_bypass_retargeting_time=False,
            )
            na, nb = oi_to_node[a], oi_to_node[b]
            tx_time[na] += t_eff  # each direction charges its transmitter once
            ida, idb = teg.nodes[na].id, teg.nodes[nb].id
            if ida in SOURCE_IDS and idb in RELAY_OR_DESTINATION_IDS:
                rate = min(constants.BIT_RATES[ida], constants.BIT_RATES[idb])
                source_bits[na] += t_eff * rate
    EFFECTIVE_CONTACT_TIME_CACHE.clear()
    COORDINATE_CACHE.clear()
    return {
        "horizon": float(np.sum(teg.state_durations)),
        "tx_time": tx_time,
        "source_bits": source_bits,
        "node_ids": np.array([n.id for n in teg.nodes]),
    }


def save_summary(path: str, summary: dict) -> None:
    np.savez_compressed(path, **summary)


def load_summary(path: str) -> dict:
    with np.load(path) as z:
        return {
            "horizon": float(z["horizon"]),
            "tx_time": z["tx_time"],
            "source_bits": z["source_bits"],
            "node_ids": [str(x) for x in z["node_ids"]],
        }
