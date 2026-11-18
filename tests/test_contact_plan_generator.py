from pathlib import Path

import pytest

from tools.compare_contact_plans import compare
from tools.contact_plan_generator import generate, write_ion

REF = Path(__file__).resolve().parents[1] / (
    "scenarios/experiments/mars_relay_earth_scenario_inc_{n}/"
    "mars_relay_earth_scenario_inc_{n}_contact_plan"
)


@pytest.mark.parametrize("size", [4, 16])
def test_day0_reproduces_ipnd_plan(size, tmp_path):
    ref = Path(str(REF).format(n=size))
    if not ref.exists():
        pytest.skip("IPN-D reference plan not present")
    out = tmp_path / "plan"
    write_ion(out, generate(size, 0.0, 86400.0))
    r = compare(str(ref), str(out))
    # Same communicating node pairs, and interval overlap close to the reference.
    assert r["pairs_only_ref"] == 0 and r["pairs_only_new"] == 0
    assert r["mean_jaccard"] > 0.88
    assert abs(r["contacts_new"] / r["contacts_ref"] - 1) < 0.05


def test_plan_is_parseable(tmp_path, monkeypatch):
    from src.topology import contact_plan as cp

    out = tmp_path / "x_contact_plan"
    write_ion(out, generate(4, 5 * 86400.0, 86400.0))
    monkeypatch.setattr(cp, "get_experiment_file", lambda *_: str(out))
    plan = cp.IONContactPlanParser().read("x")
    assert len(plan.contacts) > 0
    assert all(c.end_time > c.start_time for c in plan.contacts)
