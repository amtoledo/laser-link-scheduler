"""
Runtime power profile.

The WiSEE 2026 evaluation used two hard-coded operating points (EOL: P0=70 W, P_base=69.9 W;
BOL: P0=250 W, P_base=200 W) with the 4 W optical output as transmission consumption. Both are
degenerate ends of one parameter, the margin ratio

    rho = (P0 - P_base) / P_tx

i.e. the largest transmit duty cycle a node can sustain from its RTG alone (rho >= 1: continuous
transmission is free; 0 < rho < 1: duty cycles above rho drain the battery; rho <= 0: the node
cannot even sustain its baseline load). Because the RTG decays, rho(t) = (P0 e^{-lambda t} - P_base) / P_tx.

This module lets the CLI set the operating point without editing constants.py, and records the
profile next to each report so the analysis knows what was simulated.
"""

import hashlib
import json
import os

from src.constants import SOURCE_NODES, MLConfig, OCTConfig


def apply_power_profile(
    p_base: float | None = None,
    p_tx: float | None = None,
    p0: float | None = None,
    rho: float | None = None,
    relay_rho: float | None = None,
    battery_wh: float | None = None,
    battery_reserve: float | None = None,
    old_source_rho: float | None = None,
    old_source_fraction: float = 0.5,
) -> None:
    """Override the class-level power constants in place. None keeps the current value.

    `rho` sets P0 = P_base + rho * P_tx for source nodes (and relays unless `relay_rho` is given).
    `p0` and `rho` are mutually exclusive.
    """
    if p0 is not None and rho is not None:
        raise ValueError("Pass either --p0 or --rho, not both.")

    if p_tx is not None:
        OCTConfig.ELECTRICAL_TX_POWER = float(p_tx)
    if p_base is not None:
        MLConfig.SOURCE_BASELINE_POWER_FOR_BASIC_OPERATION = float(p_base)
        MLConfig.RELAY_BASELINE_POWER_FOR_BASIC_OPERATION = float(p_base)
    if battery_wh is not None:
        MLConfig.SOURCE_NODE_INITIAL_BATTERY = float(battery_wh)
        MLConfig.RELAY_NODE_INITIAL_BATTERY = float(battery_wh)
    if battery_reserve is not None:
        if not 0.0 <= battery_reserve <= 1.0:
            raise ValueError("--battery-reserve must be in [0, 1].")
        MLConfig.BATTERY_RESERVE_FRACTION = float(battery_reserve)

    ptx = OCTConfig.ELECTRICAL_TX_POWER
    if p0 is not None:
        MLConfig.SOURCE_NODE_INITIAL_POWER = float(p0)
        MLConfig.RELAY_NODE_INITIAL_POWER = float(p0)
    if rho is not None:
        MLConfig.SOURCE_NODE_INITIAL_POWER = (
            MLConfig.SOURCE_BASELINE_POWER_FOR_BASIC_OPERATION + rho * ptx
        )
        MLConfig.RELAY_NODE_INITIAL_POWER = (
            MLConfig.RELAY_BASELINE_POWER_FOR_BASIC_OPERATION + rho * ptx
        )
    if relay_rho is not None:
        MLConfig.RELAY_NODE_INITIAL_POWER = (
            MLConfig.RELAY_BASELINE_POWER_FOR_BASIC_OPERATION + relay_rho * ptx
        )
    MLConfig.P0_OVERRIDES = {}
    if old_source_rho is not None:
        # Mixed-age fleet: a fixed pseudo-random subset of sources ("old", e.g. extended-mission
        # spacecraft) gets margin ratio `old_source_rho`; the rest keep `rho`. The subset is
        # chosen by hashing node ids, so it is the same in every scenario size and does not
        # line up with orbital planes (ids cycle through the planes).
        p0_old = (
            MLConfig.SOURCE_BASELINE_POWER_FOR_BASIC_OPERATION
            + old_source_rho * ptx
        )
        for node_id in old_sources(old_source_fraction):
            MLConfig.P0_OVERRIDES[node_id] = p0_old


def old_sources(
    fraction: float, candidates: list[str] | None = None
) -> list[str]:
    """Deterministic pseudo-random subset of source ids: those whose id hash falls below
    `fraction`. The expected share of old sources is `fraction` at every scenario size."""
    ids = list(SOURCE_NODES if candidates is None else candidates)

    def key(n: str) -> float:
        return int(hashlib.sha1(n.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    return [n for n in ids if key(n) < fraction]


def current_power_profile() -> dict[str, float | dict[str, float]]:
    ptx = OCTConfig.ELECTRICAL_TX_POWER
    src_p0 = MLConfig.SOURCE_NODE_INITIAL_POWER
    src_pb = MLConfig.SOURCE_BASELINE_POWER_FOR_BASIC_OPERATION
    rel_p0 = MLConfig.RELAY_NODE_INITIAL_POWER
    rel_pb = MLConfig.RELAY_BASELINE_POWER_FOR_BASIC_OPERATION
    return {
        "p_tx": ptx,
        "source_p0": src_p0,
        "source_p_base": src_pb,
        "source_rho": (src_p0 - src_pb) / ptx,
        "relay_p0": rel_p0,
        "relay_p_base": rel_pb,
        "relay_rho": (rel_p0 - rel_pb) / ptx,
        "battery_wh": MLConfig.SOURCE_NODE_INITIAL_BATTERY,
        "battery_reserve": MLConfig.BATTERY_RESERVE_FRACTION,
        "budget_period_s": MLConfig.BUDGET_PERIOD,
        "decay_rate_per_s": MLConfig.DECAY_RATE,
        "p0_overrides": dict(MLConfig.P0_OVERRIDES),
    }


def write_power_profile(report_dir: str) -> None:
    with open(os.path.join(report_dir, "power_profile.json"), "w") as f:
        json.dump(current_power_profile(), f, indent=2)
