# CPD scenarios for the lifespan-aware laser-link-scheduler paper

Contact Plan Designer (CPD) versions of the `mars_relay_earth_scenario_inc_<N>`
family that Ale runs with `make run-mars_relay` on the `lifespan-aware-scheduling`
branch of laser-link-scheduler. `N` = 4, 8, … 64.

Built from Jason's orbit-generator CSVs in `../ipn-d-inputs-json/`
(`mars_relay_earth_scenario_<N>.csv` → experiment `..._inc_<N>`), plus IPN-D's
planet database and time grid from its `config.json`. Nothing here is guessed
from the contact plans.

Generator: `convert_ipnd_csv.py` (copy of the canonical one in
`~/Documents/Code/cpd/lls-mars-relay/`). Regenerate with

```bash
python3 convert_ipnd_csv.py --radius-mode as-run --suffix "" --duration-s 21600 --out-dir as-run
python3 convert_ipnd_csv.py --radius-mode as-specified --suffix "" --duration-s 21600 --out-dir as-intended
```

## The two variants

| | Source orbit | Relay orbit | Ground stations | What it is |
|---|---|---|---|---|
| `as-run/` | r = 7779 km (alt 4389.5 km) | r = 9779 km (alt 6389.5 km) | Longitudes mirrored, as IPN-D placed them | Everything as IPN-D actually computed it, i.e. the scenario behind every number in the paper |
| `as-intended/` | r = 4389.5 km (alt 1000 km) | r = 6389.5 km (alt 3000 km) | True DSN geography | Everything as the CSVs specify it |

Use `as-run` for anything that has to agree with the results; use `as-intended`
if the figure is meant to show the scenario as designed. **Earth spins prograde
in both** — see finding 2.

## Horizons, and which size to open in the web UI

Both variants are generated at a **6 h horizon** (21600 s) to keep the web UI
responsive, starting at IPN-D's epoch, so they are the first quarter of the
reference day.

Contact count, the thing that makes the web UI slow, grows as N² (node pairs):

| N | nodes | as-run, 6 h | as-intended, 6 h | as-run, 24 h |
|---|---|---|---|---|
| 4 | 10 | 98 | 190 | 238 |
| 8 | 14 | 228 | — | 642 |
| 16 | 22 | 612 | 1,286 | 1,566 |
| 32 | 44 | 2,582 | — | 6,814 |
| 64 | 88 | 10,828 | 20,552 | 28,576 |

`as-intended` carries roughly twice the contacts of `as-run` at the same N: its
orbits are lower, so the constellation is denser and more mutually visible.

**N = 16 is the sweet spot for a figure**: 612 contacts at 6 h, and it still
shows all four orbital planes and the relay ring. Note that shortening the
horizon is a weak lever compared with dropping N — N = 64 still carries 10,828
contacts at 6 h, because the source-source links are dense and recur throughout.

**One consequence of the 6 h horizon:** Earth turns only 90°, so the DSN station
handover no longer appears — over a quarter day a station is largely either in
view or not. Regenerate at the full day if the figure needs to show the stations
rotating through:

```bash
python3 convert_ipnd_csv.py --radius-mode as-run --suffix "" --out-dir as-run
python3 convert_ipnd_csv.py --radius-mode as-specified --suffix "" --out-dir as-intended
```

The validation numbers below were measured against the reference at its **24 h**
horizon; they characterise the geometry, which the shorter span does not change.

## What the scenario actually is

| | |
|---|---|
| Sources (ids 2001+) | Mars, i = 87°, e = 0, 4 planes at RAAN 0/90/180/270, mean anomaly `s·360/(N/4) + p·1°` |
| Relays (ids 1001+) | Mars, equatorial (i = 0), e = 0, evenly phased |
| Ground stations (ids 9001+) | The three real DSN sites (Goldstone, Madrid, Canberra), 100 km altitude, cycling as the count grows |
| Counts | N sources, and 3·⌈N/16⌉ relays and ground stations |
| Time | IPN-D ran 24 h from 725803264.184 (J2000 s), 600 s step; these files are cut to the first 6 h |
| Bodies | IPN-D's J2000 planet elements, propagated to the start epoch |

## Findings

**1. The altitudes disagree three ways.** The paper says 500 km (sources) and
1500 km above them. The CSVs say 1000 km and 3000 km. The plans IPN-D produced
sit at orbit radii of 7779 and 9779 km, which is `altitude + 2R` rather than
`altitude + R` — the Mars radius added twice. This is decisive, not marginal:
building the CSV altitudes literally inflates the contact count roughly threefold,
produces no relay-relay contacts where the reference has six, and misses whole
node pairs (mean Jaccard 0.56 vs 0.91). **The paper's stated altitudes match
neither the inputs nor the runs and should be corrected.**

**2. IPN-D's ground-station longitudes are mirrored.** In the reference plans a
station reaches its Mars-visibility window *later* the further east it sits: the
three station pairs fit `dt/dlon = +237, +241, +242 s/deg`. A prograde Earth
gives the opposite sign — a station's inertial longitude is `lon + pm + ω·t`, so
it faces a fixed target at `t = (lon_target − pm − lon)/ω`, slope `−1/ω =
−240 s/deg`, and the more easterly station gets there first. **IPN-D is the
mirrored one; CPD spins Earth correctly**, as both web GUIs show.

Negating the station longitudes and reversing Earth's spin reproduce IPN-D's
windows equally well and cannot be told apart from the windows alone. `as-run`
uses the **longitude flip with a prograde Earth** (`prime-meridian-deg: 215`),
which matches all three reference midpoints to within 300 s — half a grid cell —
and keeps any visualization physically correct. Most likely IPN-D reads the
longitude column as west-positive while Jason's CSV gives the real east-positive
DSN coordinates. *Worth confirming with Jason.*

**3. Ground-station altitude is read as kilometres.** The CSV column says `100`.
Solving the recorded station positions for the Earth radius that reproduces their
CSV latitudes gives 6371 + 100 km, so IPN-D read metres as kilometres. Same
flavour of unit bug as (1), and harmless in CPD, which clamps a lander's mask at
its local horizon either way.

**4. A CPD display bug: `propagation-epoch-s` desynchronises the 3D globe from
its own ground stations.** `cpd_api` sends the viewer a spin of
`prime_meridian + spin_rate·t` with a scenario-relative `t`, but
`lander_position_rel` is called with `times + t0`. Texture and landers therefore
drift apart by `ω·t0`, which for Earth over this epoch is **127.54°** — enough to
render Canberra in the South Pacific. It affects any scenario setting
`propagation-epoch-s`, not just these.

Worked around here by stating every element **at the epoch** instead: mean
anomalies advanced by `360·t0/T`, prime meridians by `360·t0/T_rot`, and no
`propagation-epoch-s`. Both are exact (mean anomaly and rotation phase are linear
in time) and the contact plans come out bit-identical, verified at N = 4 and 16.
With `t0 = 0` the viewer's spin matches the physics and each station renders at
its stated longitude. *The underlying bug is still worth fixing in CPD.*

Note that `as-run` still renders its stations mirrored (Goldstone at +116.89°
rather than −116.89°). That is finding 2, and it is deliberate — it is what
reproduces IPN-D's contact times. Use `as-intended` for a figure that has to show
the DSN sites where they really are.

**5. The `a position` records in the ION plans are not trustworthy.** They are
partly frozen: node 2001 has an identical Mars-relative position from t = 79200
through 85800 s, impossible for a real orbit. Do not reverse-engineer geometry
from them.

## How closely `as-run` reproduces Ale's plans

| N | CPD / ref contacts | node pairs | mean Jaccard | src–src | src–relay | relay–relay |
|---|---|---|---|---|---|---|
| 4 | 238 / 244 | 42 (identical) | 0.848 | 0.983 | 0.846 | 1.000 |
| 16 | 1566 / 1600 | 220 (identical) | 0.912 | 0.974 | 0.847 | 1.000 |
| 32 | 6810 / 6942 | 883 (identical) | 0.910 | 0.971 | 0.847 | 1.000 |
| 64 | 28560 / 29012 | 3602 (identical) | 0.911 | 0.971 | 0.847 | 1.000 |

Jaccard = overlap of contact intervals per node pair. At every size the set of
communicating pairs is **exactly** the reference's — none extra, none missing.

Residuals, all understood:

- **Ground stations (0.86).** IPN-D's stations at 100 km over an occluding sphere
  give a −10.06° horizon; CPD clamps a lander at its local horizon, so CPD's
  windows are 5400–6600 s shorter but correctly centred. Clipping the reference
  to CPD's horizon lifts source–GS to **0.986**, accounting for it entirely.
- **Contact counts 1–2 % low.** The 600 s grid; use `cpd compute --no-refine` to
  reproduce it rather than CPD's default refined boundaries.
- **Source–relay 0.847 is a labelling offset, not geometry.** Both tools
  propagate ~24,700 relay revolutions from J2000, where one part in 10⁶ of period
  difference is 60° of accumulated phase. Relabelling relays by 240° of mean
  anomaly lifts source–relay to 0.980 and the overall mean to 0.947 — the same
  constellation, different node ids in the slots. Closing it properly needs the
  per-node `OrbitPeriod` values IPN-D used, which the CSVs do not carry.
  *Second thing to ask Jason.*

Rejected hypotheses: no time-reversal signature; de-rotating node mean anomalies
by the propagation-epoch phase makes agreement worse, so CPD and IPN-D treat node
phases the same way. Agreement is flat across the day (source–relay 0.832–0.848
in every 3 h window), so there is no drift and hence no mean-motion error, only
the fixed phase offset.

## Two things not modelled

- **Apparent vs geometric positions.** IPN-V/IPN-D compute contacts on
  light-time-corrected apparent positions; these scenarios use CPD's default
  instantaneous geometry. CPD has an opt-in apparent mode (`light-time: true`).
  Untested here — the flat-across-the-day agreement suggests it is not the main
  residual, but at an 18-minute OWLT it is worth a check before quoting these
  plans as equivalent.
- **Obliquity.** IPN-D's config carries 23.44° (Earth) and 25.19° (Mars); CPD
  applies no obliquity to contact geometry, so the YAMLs set it to 0.

## CPD output does not feed back into laser-link-scheduler

`IONContactPlanParser.read` (`src/topology/contact_plan.py`) splits the ION file
on blank lines expecting four sections and zips contacts, ranges and positions
row-for-row. CPD's output has three: it emits no `a position` section, prefixes
an `@ <epoch>` line, and aggregates range lines per node pair (42 lines for 238
contacts) rather than one per contact. Reading a CPD plan into the scheduler
would need changes on one side or the other.
