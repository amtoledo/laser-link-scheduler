import copy

import numpy as np
import pytest

from src.constants import MLConfig, OCTConfig
from src.power_profile import apply_power_profile
from src.schedulers import EnergyBudgetScheduler, LaserLinkScheduler
from src.time_expanded_graph.time_expanded_graph import get_time_expanded_graph
from src.topology.weights import compute_capacity, compute_node_capacities
from src.utils import FileType

EXPERIMENT = "mars_relay_earth_scenario_inc_8"


@pytest.fixture(scope="module")
def teg():
    try:
        return get_time_expanded_graph(EXPERIMENT, FileType.TEG)
    except FileNotFoundError:
        pytest.skip(f"cached TEG for {EXPERIMENT} not built; run main.py once")


@pytest.fixture(autouse=True)
def restore_constants():
    saved = (dict(vars(MLConfig)), dict(vars(OCTConfig)))
    yield
    for cls, attrs in zip((MLConfig, OCTConfig), saved):
        for k, v in attrs.items():
            if not k.startswith("__") and not callable(v):
                try:
                    setattr(cls, k, v)
                except (AttributeError, TypeError):
                    pass


def capacity(t):
    caps = compute_node_capacities(
        t.graphs,
        t.state_durations,
        t.K,
        t.nodes,
        t.graphs,
        t.pos,
        t.optical_interfaces_to_node,
        t.node_to_optical_interfaces,
        False,
    )
    return compute_capacity(caps)


@pytest.mark.parametrize("rho", [0.05, 0.2, 1.0])
def test_budget_never_exceeded(teg, rho):
    apply_power_profile(p_tx=40.0, p_base=200.0, rho=rho, battery_reserve=1.0)
    sched = EnergyBudgetScheduler()
    sched.schedule(copy.deepcopy(teg))
    horizon = float(np.sum(teg.state_durations))
    allowed_wh = rho * 40.0 * horizon / 3600.0
    spent = sched.last_run_stats["tx_energy_wh"]
    finite = np.isfinite(spent)
    # RTG decay over one day is negligible, so the budget is rho * P_tx * horizon.
    assert np.all(spent[finite] <= allowed_wh * (1 + 1e-6))


def test_zero_margin_schedules_nothing(teg):
    apply_power_profile(p_tx=40.0, p_base=200.0, rho=0.0)
    out = EnergyBudgetScheduler().schedule(copy.deepcopy(teg))
    assert capacity(out) == 0


def test_unbounded_budget_matches_lls(teg):
    apply_power_profile(p_tx=40.0, p_base=200.0, rho=100.0)
    lls = LaserLinkScheduler().schedule(copy.deepcopy(teg))
    budget = EnergyBudgetScheduler(drop_zero_gain=False).schedule(
        copy.deepcopy(teg)
    )
    assert np.array_equal(lls.graphs, budget.graphs)
