"""
Line-of-sight contact plan generator for the mars_relay_earth scenarios, at any epoch and horizon.

Why this exists
---------------
The ION contact plans in scenarios/experiments/ were produced by IPN-D (gitlab.inria.fr/jfraire/ipn-v)
from Jason Gerard's orbit-generator CSVs, over a single 24 h window starting at J2000 + 725803264.184 s
(2023-01-01). orbit-generator only writes orbital elements; the horizon lives in IPN-D's config.json.
IPN-D is not reachable from this environment, so this module re-implements the same geometry
(two-body Kepler propagation + spherical occultation) and writes plans in the exact ION layout
IONContactPlanParser reads (contact / range / position sections).

Inputs are the Contact Plan Designer "as-run" scenarios (scenarios/cpd/as-run/*.yaml), which state
every element AT the IPN-D epoch and reproduce IPN-D's quirks (orbit radius = altitude + 2R,
mirrored ground-station longitudes, 100 km station altitude); see scenarios/cpd/FINDINGS.md.

Model
-----
* Planets: heliocentric Keplerian ellipses (IPN-D's J2000 elements, advanced to the epoch).
* Orbiters: circular Keplerian orbits around their central body, n = sqrt(mu / a^3).
* Landers: fixed on a rotating body, prograde spin, no obliquity (same as the CPD scenarios).
* Visibility: the segment between two nodes must not cross the Sun, Earth or Mars sphere
  (IPN-D style: a station at 100 km sees about 10 deg below its local horizon).
* Grid: visibility is sampled every `step` seconds; a sample at t that is visible opens the
  contact over [t, t + step]. Consecutive visible samples merge into one contact.
* Range (light seconds) and positions (heliocentric, km) are evaluated at each contact's start,
  as in the IPN-D plans. Positions are only used for pointing angles, so the inertial frame
  (ecliptic here, y-up in IPN-D) does not matter.

Usage
-----
    python tools/contact_plan_generator.py --size 20 --start-day 0 --days 1 --name mars_relay_earth_scenario_inc_20_d0
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CPD_DIR = REPO_ROOT / "scenarios" / "cpd" / "as-run"
EXPERIMENTS_DIR = REPO_ROOT / "scenarios" / "experiments"

C_KM_S = 299_792.458
SUN_RADIUS_KM = 696_340.0
DAY_S = 86_400.0


@dataclass
class Body:
    name: str
    radius_km: float
    mu: float
    rot_period_s: float
    prime_meridian_deg: float
    sma_km: float
    ecc: float
    inc_deg: float
    raan_deg: float
    argp_deg: float
    m0_deg: float
    period_s: float

    def position(self, t: np.ndarray) -> np.ndarray:
        """Heliocentric position (km), shape (len(t), 3)."""
        m = np.radians(self.m0_deg) + 2 * np.pi * t / self.period_s
        e = self.ecc
        E = m.copy()
        for _ in range(30):  # Newton on Kepler's equation
            E = E - (E - e * np.sin(E) - m) / (1 - e * np.cos(E))
        nu = 2 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2)
        )
        r = self.sma_km * (1 - e * np.cos(E))
        return _perifocal_to_inertial(
            r, nu, self.inc_deg, self.raan_deg, self.argp_deg
        )


@dataclass
class Node:
    node_id: str
    kind: str  # "orbiter" | "lander"
    body: str
    # orbiter
    sma_km: float = 0.0
    inc_deg: float = 0.0
    raan_deg: float = 0.0
    argp_deg: float = 0.0
    m0_deg: float = 0.0
    # lander
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    alt_km: float = 0.0


def _perifocal_to_inertial(r, nu, inc_deg, raan_deg, argp_deg) -> np.ndarray:
    i, O, w = np.radians(inc_deg), np.radians(raan_deg), np.radians(argp_deg)
    u = w + nu
    x = r * (np.cos(O) * np.cos(u) - np.sin(O) * np.sin(u) * np.cos(i))
    y = r * (np.sin(O) * np.cos(u) + np.cos(O) * np.sin(u) * np.cos(i))
    z = r * (np.sin(u) * np.sin(i))
    return np.stack([x, y, z], axis=-1)


def load_scenario(size: int) -> tuple[float, dict[str, Body], list[Node]]:
    path = CPD_DIR / f"mars_relay_earth_scenario_{size}.yaml"
    doc = yaml.safe_load(path.read_text())
    bodies = {}
    for b in doc["bodies"]:
        el = b["elements"]
        bodies[b["name"]] = Body(
            name=b["name"],
            radius_km=float(b["radius-km"]),
            mu=float(b["mu-km3-s2"]),
            rot_period_s=float(b["rotation-period-h"]) * 3600.0,
            prime_meridian_deg=float(b["prime-meridian-deg"]),
            sma_km=float(el["sma-km"]),
            ecc=float(el["eccentricity"]),
            inc_deg=float(el["inclination-deg"]),
            raan_deg=float(el["raan-deg"]),
            argp_deg=float(el["arg-periapsis-deg"]),
            m0_deg=float(el["mean-anomaly-deg"]),
            period_s=float(el["orbit-period-s"]),
        )
    nodes = []
    for n in doc["nodes"]:
        if n["type"] == "orbiter":
            el = n["elements"]
            nodes.append(
                Node(
                    node_id=str(n["id"]),
                    kind="orbiter",
                    body=n["central-body"],
                    sma_km=float(el["sma-km"]),
                    inc_deg=float(el["inclination-deg"]),
                    raan_deg=float(el["raan-deg"]),
                    argp_deg=float(el["arg-periapsis-deg"]),
                    m0_deg=float(el["mean-anomaly-deg"]),
                )
            )
        else:
            nodes.append(
                Node(
                    node_id=str(n["id"]),
                    kind="lander",
                    body=n["central-body"],
                    lat_deg=float(n["latitude-deg"]),
                    lon_deg=float(n["longitude-deg"]),
                    alt_km=float(n["altitude-m"]) / 1000.0,
                )
            )
    return float(doc["epoch-s"]), bodies, nodes


def node_positions(
    nodes: list[Node], bodies: dict[str, Body], t: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Heliocentric node positions (n_nodes, len(t), 3) and body centres."""
    centres = {name: b.position(t) for name, b in bodies.items()}
    out = np.empty((len(nodes), len(t), 3))
    for k, n in enumerate(nodes):
        body = bodies[n.body]
        if n.kind == "orbiter":
            nmot = math.sqrt(body.mu / n.sma_km**3)
            nu = (
                np.radians(n.m0_deg) + nmot * t
            )  # circular: true = mean anomaly
            rel = _perifocal_to_inertial(
                n.sma_km, nu, n.inc_deg, n.raan_deg, n.argp_deg
            )
        else:
            lam = np.radians(
                n.lon_deg
                + body.prime_meridian_deg
                + 360.0 * t / body.rot_period_s
            )
            lat = np.radians(n.lat_deg)
            r = body.radius_km + n.alt_km
            rel = np.stack(
                [
                    r * np.cos(lat) * np.cos(lam),
                    r * np.cos(lat) * np.sin(lam),
                    np.full_like(lam, r * np.sin(lat)),
                ],
                axis=-1,
            )
        out[k] = centres[n.body] + rel
    return out, centres


def segment_clear(
    a: np.ndarray, b: np.ndarray, centre: np.ndarray, radius: float
) -> np.ndarray:
    """True where segment a-b does not pass within `radius` of `centre` (arrays (..., 3))."""
    d = b - a
    dd = np.einsum("...i,...i->...", d, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.clip(np.einsum("...i,...i->...", centre - a, d) / dd, 0.0, 1.0)
    # Co-located nodes (e.g. two stations at one DSN site): IPN-D lists them as in contact.
    s = np.where(dd > 0, s, 0.0)
    closest = a + s[..., None] * d
    dist2 = np.einsum("...i,...i->...", closest - centre, closest - centre)
    return dist2 > radius**2


def generate(
    size: int,
    start_s: float,
    duration_s: float,
    step_s: float = 600.0,
    gs_count: int | None = None,
) -> list[tuple]:
    """Return contacts (t0, t1, tx, rx, range_ls, tx_pos, rx_pos), times relative to `start_s`.

    `gs_count` sets the number of ground stations. The scenarios add 3 stations per 16 sources,
    always cycling through the same three DSN sites (9001/9004/9007/9010 are co-located, etc.).
    A smaller count keeps the first N stations (gs_count=3 = one per site); a larger count
    continues the same cycle with new ids (9001 + k at site k mod 3).
    """
    _, bodies, nodes = load_scenario(size)
    if gs_count is not None:
        landers = sorted(
            (n for n in nodes if n.kind == "lander"),
            key=lambda n: int(n.node_id),
        )
        sites = landers[:3]
        others = [n for n in nodes if n.kind != "lander"]
        new_landers = []
        for k in range(gs_count):
            if k < len(landers):
                new_landers.append(landers[k])
            else:
                site = sites[k % len(sites)]
                new_landers.append(replace(site, node_id=str(9001 + k)))
        nodes = new_landers + others
    t_rel = np.arange(0.0, duration_s, step_s)
    t = start_s + t_rel
    pos, centres = node_positions(nodes, bodies, t)
    occluders = [(centres[name], b.radius_km) for name, b in bodies.items()]
    occluders.append((np.zeros((len(t), 3)), SUN_RADIUS_KM))

    contacts = []
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            vis = np.ones(len(t), dtype=bool)
            for centre, radius in occluders:
                vis &= segment_clear(pos[i], pos[j], centre, radius)
            if not vis.any():
                continue
            # Merge consecutive visible samples into [start, end) intervals.
            edges = np.diff(np.concatenate([[0], vis.astype(int), [0]]))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            for s_idx, e_idx in zip(starts, ends):
                t0 = t_rel[s_idx]
                t1 = min(t_rel[e_idx - 1] + step_s, duration_s)
                rng = np.linalg.norm(pos[i, s_idx] - pos[j, s_idx]) / C_KM_S
                for a, b in ((i, j), (j, i)):
                    contacts.append(
                        (
                            int(t0),
                            int(t1),
                            nodes[a].node_id,
                            nodes[b].node_id,
                            rng,
                            pos[a, s_idx],
                            pos[b, s_idx],
                        )
                    )
    contacts.sort(key=lambda c: (c[0], c[2], c[3]))
    return contacts


def write_ion(path: Path, contacts: list[tuple], rate: int = 1000) -> None:
    lines_c, lines_r, lines_p = [], [], []
    for t0, t1, tx, rx, rng, ptx, prx in contacts:
        lines_c.append(f"a contact +{t0} +{t1} {tx} {rx} {rate}")
        lines_r.append(f"a range +{t0} +{t1} {tx} {rx} {rng:.4f}")
        lines_p.append(
            f"a position +{t0} +{t1} {tx} {rx} "
            + " ".join(f"{v:.6f}" for v in (*ptx, *prx))
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines_c)
        + "\n\n"
        + "\n".join(lines_r)
        + "\n\n"
        + "\n".join(lines_p)
        + "\n\n"
    )


def experiment_name(
    size: int, start_day: float, days: float, gs_count: int | None = None
) -> str:
    gs = "" if gs_count is None else f"_gs{gs_count}"
    return f"mars_relay_earth_scenario_inc_{size}{gs}_d{start_day:g}" + (
        "" if days == 1 else f"_x{days:g}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--size", type=int, nargs="+", required=True)
    ap.add_argument(
        "--start-day",
        type=float,
        nargs="+",
        default=[0.0],
        help="days after the IPN-D epoch (2023-01-01)",
    )
    ap.add_argument("--days", type=float, default=1.0, help="horizon in days")
    ap.add_argument("--step", type=float, default=600.0)
    ap.add_argument(
        "--gs-count",
        type=int,
        default=None,
        help="keep only the first N ground stations (3 = one per DSN site)",
    )
    args = ap.parse_args()

    for size in args.size:
        for d in args.start_day:
            name = experiment_name(size, d, args.days, args.gs_count)
            contacts = generate(
                size, d * DAY_S, args.days * DAY_S, args.step, args.gs_count
            )
            out = EXPERIMENTS_DIR / name / f"{name}_contact_plan"
            write_ion(out, contacts)
            print(
                f"{name}: {len(contacts)} contacts -> {os.path.relpath(out)}"
            )


if __name__ == "__main__":
    main()
