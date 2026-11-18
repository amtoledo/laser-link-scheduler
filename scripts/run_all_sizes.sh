#!/bin/bash
# Full v2 sweep over every mars_relay_earth_scenario_inc_<N> size.
# Lane A: rho sweep (lls_budget at each rho) + unconstrained lls/fcp.
# Lane B: Lifespan-Aware v1 for every (P_tx, margin) pair.
set -uo pipefail
cd "$(dirname "$0")/.."
SIZES=${SIZES:-"4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64"}
EXPERIMENTS=""
for n in $SIZES; do EXPERIMENTS="$EXPERIMENTS mars_relay_earth_scenario_inc_$n"; done
export EXPERIMENTS

lane_a() {
  IDS_FILE=output/all_rho_ids.txt P_TX=40 P_BASE=200 bash scripts/run_rho_sweep.sh > output/lane_a.log 2>&1
  echo "lane A done $(date)" >> output/lane_a.log
}
lane_b() {
  IDS_FILE=output/all_v1_ids.txt P_BASE=200 bash scripts/run_ptx_sweep.sh > output/lane_b.log 2>&1
  echo "lane B done $(date)" >> output/lane_b.log
}
lane_a & lane_b & wait
