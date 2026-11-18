#!/usr/bin/env python3
"""Convert Jason Gerard's IPN-D scenario CSVs into Contact Plan Designer scenarios.

Source of truth
---------------
* Node definitions: the orbit-generator CSVs Jason supplied, in
  ``tesis/grado-famaf-2025-ale-toledo-lls/ipn-d-inputs-json/``.  These are the
  exact inputs behind the ION contact plans in
  ``laser-link-scheduler/scenarios/experiments/`` that Alejandro Toledo runs with
  ``make run-mars_relay`` on the ``lifespan-aware-scheduling`` branch.
  ``mars_relay_earth_scenario_<N>.csv`` -> experiment ``..._inc_<N>``.
* Bodies and time: IPN-D's own ``config.json`` (identical in every scenario
  under ``laser-link-scheduler/scenarios/IPN-D_scenarios/``).  The planet
  elements are J2000-referenced and IPN-D propagates them to
  ``SimulationStartTime``; CPD does the same via ``propagation-epoch-s``.
  Verified: this places Earth at 1.0032 AU and Mars at 1.5647 AU, an Earth-Mars
  range of 2.2068 AU (OWLT 1101.2 s) against the 1105.09 light-seconds carried
  by the ``a range`` records of the reference plan.

CSV format (two node blocks, each preceded by its own ``#`` header line)
    Orbiter: node_type,node_name,id,central_object,altitude,inclination,
             eccentricity,raan,arg_of_perigee,mean_anomaly,rules
    Lander:  node_type,node_name,id,central_object,latitude,longitude,
             altitude,rules

Two unit questions the CSVs leave open, both handled by ``--radius-mode``
and both worth confirming with Jason:

1. Orbiter altitude.  The CSV says 1000 km (sources) and 3000 km (relays) above
   Mars (R = 3389.5 km), i.e. orbit radii of 4389.5 and 6389.5 km.  The plans
   IPN-D actually produced sit at radii of 7779.0 and 9779.0 km -- exactly
   ``altitude + 2R`` rather than ``altitude + R``, as if the body radius were
   added twice.  ``--radius-mode as-run`` reproduces the plans; ``as-specified``
   (the default) builds the scenario the CSV describes.  Settled empirically at
   N = 4: ``as-run`` gives 238 contact lines against the reference's 244, the
   same 42 node pairs, and per-pair Jaccard 1.000 (relay-relay), 0.983
   (source-source), 0.846 (source-relay); ``as-specified`` gives 688 contact
   lines, loses three pairs and drops to 0.586 / 0.572, with no relay-relay
   contact at all.  ``as-run`` is what IPN-D ran.
2. Lander altitude.  The CSV column reads ``100``.  Solving the three DSN
   stations' recorded positions for the Earth radius that reproduces their CSV
   latitudes gives 6471 km = 6371 + 100, so IPN-D read it as kilometres.  We
   emit ``altitude-m: 100000`` to match; it makes no difference to CPD's
   geometry, which clamps a lander's mask at its local horizon either way.

Note that the paper states 500 km and 1500 km for the two Mars constellations,
which matches neither the CSV nor the plans; see the report accompanying this
converter.

Ground-station longitudes are mirrored in IPN-D
-----------------------------------------------
In the reference plans the three DSN stations reach their Mars-visibility window
*later* the further east they sit: the midpoints fit ``dt/dlon = +240 s/deg``
(measured +237, +240.8, +242.2 over the three station pairs).  A prograde Earth
gives the opposite sign.  A station's inertial longitude is ``lon + pm + w*t``,
so it faces a fixed target at ``t = (lon_target - pm - lon)/w``, i.e.
``dt/dlon = -1/w = -240 s/deg``: the more easterly station gets there first.
IPN-D is therefore the mirrored one -- CPD (and both web GUIs) spin Earth the
right way.

Two placements reproduce IPN-D's windows and are indistinguishable from the
windows alone: negating the station longitudes, or reversing Earth's spin.  We
use the *longitude* flip, because it keeps Earth prograde and so keeps any
visualization physically correct.  Fitted against CPD at N = 4, mirrored
longitudes with a prograde Earth and ``prime-meridian-deg: 215`` reproduce all
three reference midpoints to within 300 s, half of one 600 s grid cell.
(Reversing the spin instead fits exactly, 0 s, the residual being the
latitude asymmetry between the two placements -- not worth a backwards Earth.)

Most likely IPN-D reads the longitude column as west-positive while Jason's CSV
writes the real east-positive DSN coordinates.  Worth confirming with him.
Only ``as-run`` mirrors; ``as-specified`` keeps the true DSN geography, with its
phase chosen so gs1 stays at the same time of day.

Elements are stated AT the epoch, not at J2000
----------------------------------------------
IPN-D's planet elements are J2000-referenced, and CPD can carry that with
``propagation-epoch-s``, which evaluates bodies AND nodes at ``times + t0``.
We do not use it, because the 3D view does not: ``cpd_api`` sends the globe a
spin of ``prime_meridian + spin_rate * t`` with a scenario-relative ``t``, while
``lander_position_rel`` is called with ``times + t0``.  The globe's texture and
its own ground stations therefore drift apart by ``omega * t0``, which for
Earth over this epoch is 127.54 deg -- enough to put Canberra in the South
Pacific.  (Reported to Juan; it is a CPD display bug, not a scenario error, and
it affects any scenario that sets ``propagation-epoch-s``.)

So we bake the epoch in instead and emit no ``propagation-epoch-s``: every mean
anomaly is advanced by ``360 * t0 / T`` and every prime meridian by
``360 * t0 / T_rot``.  Both are exact, since mean anomaly and rotation phase are
linear in time, and the resulting contact plan is byte-identical to the
``propagation-epoch-s`` form.  With ``t0 = 0`` the view's spin matches the
physics, so a station renders at its stated longitude.

Known residual: which node sits in which orbit slot
---------------------------------------------------
Both tools propagate the Mars orbiters Keplerian from J2000 to the simulation
start, roughly 24 700 (relays) and 34 800 (sources) revolutions.  A relative
period difference of one part in a million is 60 deg of accumulated phase there,
so the *physical* constellation comes out identical while the mapping of node id
to orbit slot is rotated: relabelling the relays by 240 deg of mean anomaly and
the sources by 180 deg of RAAN lifts the mean per-pair Jaccard from 0.85/0.91 to
0.92/0.95 and source-relay from 0.85 to 0.98, at every N.  Fixing this needs the
``OrbitPeriod`` IPN-D used per node, which the CSVs do not carry -- another one
for Jason.  Nothing is compensated here, because doing so would mean editing the
CSV's mean anomalies.

Usage
    python3 convert_ipnd_csv.py                     # all mars_relay_earth sizes
    python3 convert_ipnd_csv.py --pattern 'gs_mars_earth_scenario_*.csv'
    python3 convert_ipnd_csv.py --radius-mode as-run
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

IPND_INPUTS = Path(
    "/Users/jfraire/My Drive/tesis/grado-famaf-2025-ale-toledo-lls/ipn-d-inputs-json"
)
OUT_DIR = Path(__file__).resolve().parent / "scenarios"

# ---------------------------------------------------------------------------
# IPN-D config.json: time grid and planet database (J2000 elements).
# ---------------------------------------------------------------------------
SIM_START_S = 725803264.184          # J2000 seconds; 2023-01-01T00:00:00 TDB
SIM_DURATION_S = 86400.0             # contactPlan.json spans exactly 24 h
SIM_STEP_S = 600.0                   # grid of every `a contact` boundary

SUN_RADIUS_KM = 696340.0

@dataclass(frozen=True)
class IpndBody:
    name: str
    radius_km: float
    mu_km3_s2: float
    sma_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_periapsis_deg: float
    mean_anomaly_deg: float
    orbit_period_s: float
    obliquity_deg: float
    rotation_period_d: float

# Values copied verbatim from IPN-D's config.json (Planets[]).  mu is not in
# IPN-D's config -- it derives node motion from each node's OrbitPeriod -- so we
# supply the standard values CPD needs to propagate orbiters.
BODIES = {
    "Earth": IpndBody("Earth", 6371.0, 398600.4418, 149600000.0, 0.0167, 0.0, 0.0,
                      102.9, 100.46435, 31558863.596656322, 23.44, 0.99),
    "Mars": IpndBody("Mars", 3389.5, 42828.375214, 227936640.0, 0.093405, 1.849,
                     49.558, 286.502, 19.356, 59353338.82413626, 25.19, 1.03),
}

RATE_BPS = 1000  # laser-link-scheduler/src/constants.py default_BIT_RATE

# Every body spins prograde, as it physically should; the IPN-D mirroring is
# carried entirely by the lander longitudes (see the docstring).  The Earth phase
# is fitted per mode against the reference plans' DSN window midpoints; Mars
# carries no landers in this family, so its phase is unconstrained.
PRIME_MERIDIAN_DEG = {
    # as-run: station longitudes mirrored, reproducing IPN-D's window order.
    "as-run": {"Earth": 215.0, "Mars": 0.0},
    # as-specified: true DSN geography, so the stations run in the physically
    # correct order; the phase keeps gs1 (Goldstone) at the same time of day.
    "as-specified": {"Earth": 88.5, "Mars": 0.0},
}


@dataclass
class Orbiter:
    name: str
    node_id: int
    body: str
    altitude_km: float
    inclination_deg: float
    eccentricity: float
    raan_deg: float
    arg_perigee_deg: float
    mean_anomaly_deg: float


@dataclass
class Lander:
    name: str
    node_id: int
    body: str
    latitude_deg: float
    longitude_deg: float
    altitude_km: float


def parse_csv(path: Path) -> tuple[list[Orbiter], list[Lander], list[str]]:
    """Parse one orbit-generator CSV.  ``#`` lines are headers or section labels."""
    orbiters: list[Orbiter] = []
    landers: list[Lander] = []
    sections: list[str] = []
    for row in csv.reader(path.read_text().splitlines()):
        if not row or not row[0].strip():
            continue
        first = row[0].strip()
        if first.startswith("#"):
            if not first.startswith("#node_type"):
                sections.append(first.lstrip("#").strip())
            continue
        kind = first.lower()
        if kind == "orbiter":
            orbiters.append(Orbiter(
                name=row[1].strip(), node_id=int(row[2]), body=row[3].strip(),
                altitude_km=float(row[4]), inclination_deg=float(row[5]),
                eccentricity=float(row[6]), raan_deg=float(row[7]),
                arg_perigee_deg=float(row[8]), mean_anomaly_deg=float(row[9]),
            ))
        elif kind == "lander":
            landers.append(Lander(
                name=row[1].strip(), node_id=int(row[2]), body=row[3].strip(),
                latitude_deg=float(row[4]), longitude_deg=float(row[5]),
                altitude_km=float(row[6]),
            ))
        else:
            raise ValueError(f"{path.name}: unknown node type {first!r}")
    return orbiters, landers, sections


def num(x: float) -> str:
    """Render a float without a trailing ``.0``."""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:g}"


def advance_mean_anomaly(m0_deg: float, period_s: float, dt_s: float) -> float:
    """Mean anomaly after ``dt_s``.  Exact for any eccentricity: the mean anomaly
    is by definition linear in time (``M = M0 + n*t``, cf. cpd.ephem.kepler)."""
    return (m0_deg + 360.0 * dt_s / period_s) % 360.0


def orbit_period_s(sma_km: float, mu_km3_s2: float) -> float:
    """Keplerian period; CPD uses ``n = sqrt(mu/a^3)`` when no period is given."""
    return 2.0 * math.pi * math.sqrt(sma_km ** 3 / mu_km3_s2)


def body_block(name: str, propagate: bool, radius_mode: str) -> list[str]:
    b = BODIES[name]
    rot_h = b.rotation_period_d * 24.0
    # Bake the propagation epoch into the rotation phase (see BAKE_EPOCH note).
    pm = (PRIME_MERIDIAN_DEG[radius_mode].get(b.name, 0.0)
          + 360.0 * SIM_START_S / (rot_h * 3600.0)) % 360.0
    out = [
        f"  - name: {b.name}",
        "    parent: Sun",
        f"    mu-km3-s2: {b.mu_km3_s2}",
        f"    radius-km: {num(b.radius_km)}",
        f"    rotation-period-h: {rot_h:g}            "
        f"# IPN-D RotationPeriod {b.rotation_period_d} d; prograde",
        f"    prime-meridian-deg: {pm:.4f}       "
        "    # spin phase AT the epoch (epoch baked in, see header)",
        "    obliquity-deg: 0.0                 # IPN-D config says "
        f"{b.obliquity_deg}; CPD applies no obliquity to geometry",
    ]
    if propagate:
        m = advance_mean_anomaly(b.mean_anomaly_deg, b.orbit_period_s, SIM_START_S)
        out += [
            f"    elements: {{ sma-km: {num(b.sma_km)}, eccentricity: {b.eccentricity},"
            f" inclination-deg: {b.inclination_deg},",
            f"                raan-deg: {b.raan_deg}, arg-periapsis-deg: {b.arg_periapsis_deg},"
            f" mean-anomaly-deg: {m:.6f},   # J2000 value {b.mean_anomaly_deg}, advanced to the epoch",
            f"                orbit-period-s: {b.orbit_period_s} }}",
        ]
    return out


def scenario_yaml(csv_path: Path, radius_mode: str,
                  duration_s: float = SIM_DURATION_S,
                  step_s: float = SIM_STEP_S) -> str:
    orbiters, landers, sections = parse_csv(csv_path)
    stem = csv_path.stem
    bodies_used = sorted({o.body for o in orbiters} | {l.body for l in landers})

    lines: list[str] = []
    a = lines.append
    a(f"# {stem} -- IPN-D scenario converted for the Contact Plan Designer.")
    a(f"# Source: ipn-d-inputs-json/{csv_path.name} (Jason Gerard's orbit-generator")
    a("# output) plus the planet database and time grid from IPN-D's config.json.")
    a(f"# Sections in the CSV: {', '.join(sections) if sections else 'n/a'}.")
    a(f"# {len(orbiters)} orbiters, {len(landers)} landers.")
    if radius_mode == "as-run":
        a("# radius-mode=as-run: orbit radius = altitude + 2*R, reproducing what")
        a("#   IPN-D actually propagated (see convert_ipnd_csv.py docstring).")
    else:
        a("# radius-mode=as-specified: orbit radius = altitude + R, as the CSV states.")
    a("# Generated by convert_ipnd_csv.py -- edit that, not this file.")
    a(f"name: {stem.replace('_', '-')}")
    a(f"epoch-s: {SIM_START_S}          # IPN-D SimulationStartTime (J2000 s)")
    a("# No propagation-epoch-s: every element below is already stated AT that epoch.")
    a("# (Keeps the 3D view's globe aligned with its landers -- see the header.)")
    a(f"duration-s: {duration_s}"
      + ("" if duration_s == SIM_DURATION_S
         else f"           # SHORTENED from IPN-D's {SIM_DURATION_S:g} s for a lighter render"))
    a(f"step-s: {step_s}                   # IPN-D Step; `cpd compute --no-refine` keeps this grid")
    a("")
    a("bodies:")
    for name in bodies_used:
        lines.extend(body_block(name, propagate=True, radius_mode=radius_mode))
    a("")
    a("nodes:")

    if landers:
        a("  # ===== Landers (optical ground stations) =====")
        for l in landers:
            a(f"  - id: {l.node_id}")
            a(f"    name: {l.name}")
            a("    type: lander")
            a(f"    central-body: {l.body}")
            a(f"    latitude-deg: {l.latitude_deg}")
            if radius_mode == "as-run":
                a(f"    longitude-deg: {-l.longitude_deg:g}"
                  f"        # CSV says {l.longitude_deg:g}; mirrored, as IPN-D placed it")
            else:
                a(f"    longitude-deg: {l.longitude_deg:g}")
            a(f"    altitude-m: {num(l.altitude_km * 1000.0)}          "
              "# CSV column reads 100, read by IPN-D as km")
            a("    constraints: { min-elevation-deg: 0.0 }")
            a(f"    data-rate-bps: {RATE_BPS}")
        a("")

    by_body: dict[str, list[Orbiter]] = {}
    for o in orbiters:
        by_body.setdefault(o.body, []).append(o)
    for body, group in by_body.items():
        r_body = BODIES[body].radius_km
        alts = sorted({o.altitude_km for o in group})
        a(f"  # ===== {body} orbiters ===== (CSV altitudes: "
          f"{', '.join(num(x) for x in alts)} km)")
        mu = BODIES[body].mu_km3_s2
        for o in group:
            r = o.altitude_km + (2 * r_body if radius_mode == "as-run" else r_body)
            # Advance the CSV's mean anomaly to the epoch, so the scenario needs no
            # propagation-epoch-s.  Exact: M is linear in time (see BAKE note).
            m = advance_mean_anomaly(o.mean_anomaly_deg, orbit_period_s(r, mu), SIM_START_S)
            a(f"  - id: {o.node_id}")
            a(f"    name: {o.name}")
            a("    type: orbiter")
            a(f"    central-body: {o.body}")
            a(f"    elements: {{ sma-km: {num(r)}, eccentricity: {o.eccentricity},"
              f" inclination-deg: {num(o.inclination_deg)},")
            a(f"                raan-deg: {num(o.raan_deg)},"
              f" arg-periapsis-deg: {num(o.arg_perigee_deg)},"
              f" mean-anomaly-deg: {m:.6f} }}   # CSV {num(o.mean_anomaly_deg)}, advanced to the epoch")
            a(f"    data-rate-bps: {RATE_BPS}")
        a("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="mars_relay_earth_scenario_*.csv",
                    help="glob over the IPN-D input CSVs (default: the mars_relay family)")
    ap.add_argument("--radius-mode", choices=("as-specified", "as-run"),
                    default="as-specified",
                    help="orbit radius = altitude + R (default) or altitude + 2R (as IPN-D ran it)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--inputs", type=Path, default=IPND_INPUTS)
    ap.add_argument("--duration-s", type=float, default=SIM_DURATION_S,
                    help=f"horizon in seconds (default {SIM_DURATION_S:g}, IPN-D's 24 h). "
                         "Shorten it to cut the contact count for a lighter web render.")
    ap.add_argument("--step-s", type=float, default=SIM_STEP_S,
                    help=f"sampling step in seconds (default {SIM_STEP_S:g}, IPN-D's grid)")
    ap.add_argument("--suffix", default=None,
                    help="filename suffix before .yaml; defaults to '_asrun' in as-run mode "
                         "and '' otherwise. Pass '' when the mode is already in the directory name.")
    args = ap.parse_args()

    paths = sorted(args.inputs.glob(args.pattern),
                   key=lambda p: (len(p.stem), p.stem))
    if not paths:
        raise SystemExit(f"no CSVs matched {args.pattern} in {args.inputs}")
    if args.suffix is None:
        args.suffix = "" if args.radius_mode == "as-specified" else "_asrun"
    suffix = args.suffix
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        out = args.out_dir / f"{path.stem}{suffix}.yaml"
        out.write_text(scenario_yaml(path, args.radius_mode,
                                     args.duration_s, args.step_s))
        print("wrote", out.relative_to(args.out_dir.parent))


if __name__ == "__main__":
    main()
