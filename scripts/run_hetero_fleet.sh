#!/bin/bash
# Heterogeneous fleets (LLS-Budget). Set A: mixed-age sources (old share ~50% at OLD_RHOS, the rest
# at RHO_NEW). Set B: repurposed relays at RELAY_RHOS with sources at RHO_NEW. Summaries only.
# Usage: SET=A bash scripts/run_hetero_fleet.sh   (or SET=B, SET=U for the unconstrained baselines)
set -uo pipefail
cd "$(dirname "$0")/.."
SIZES=${SIZES:-"8 20 64"}
P_TX=${P_TX:-40}; P_BASE=${P_BASE:-200}; RHO_NEW=${RHO_NEW:-0.5}
OLD_RHOS=${OLD_RHOS:-"0 0.025 0.05 0.1 0.2"}
RELAY_RHOS=${RELAY_RHOS:-"0 0.025 0.05 0.1 0.2"}
IDS_FILE=${IDS_FILE:-output/hetero_ids.txt}
E_ARGS=""; for n in $SIZES; do E_ARGS="$E_ARGS -e mars_relay_earth_scenario_inc_$n"; done
run(){ uv run --frozen main.py $E_ARGS "$@" --p-tx "$P_TX" --p-base "$P_BASE" --store summary --debug \
        | grep -o 'report_id=[0-9]*' | cut -d= -f2 >> "$IDS_FILE"; }
case "${SET:-A}" in
  A) for r in $OLD_RHOS; do run -s lls_budget --rho "$RHO_NEW" --old-source-rho "$r"; done ;;
  B) for r in $RELAY_RHOS; do run -s lls_budget --rho "$RHO_NEW" --relay-rho "$r"; done ;;
  H) run -s lls_budget --rho "$RHO_NEW" ;;   # homogeneous reference
  U) run -s lls -s fcp --rho "$RHO_NEW" ;;   # schedules do not depend on power
esac
