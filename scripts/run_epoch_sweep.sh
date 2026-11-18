#!/bin/bash
# Epoch sweep: generate 24 h plans at many start days (tools/contact_plan_generator.py) and run
# the unconstrained schedulers + lls_budget at several rho on each. One report per rho.
set -uo pipefail
cd "$(dirname "$0")/.."
SIZE=${SIZE:-20}
DAYS=${DAYS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13"}
RHOS=${RHOS:-"0.025 0.05 0.1 0.15 0.2 0.3 0.5"}
P_TX=${P_TX:-40}; P_BASE=${P_BASE:-200}
IDS_FILE=${IDS_FILE:-output/epoch_ids_$SIZE.txt}

uv run --frozen python tools/contact_plan_generator.py --size "$SIZE" --start-day $DAYS
E_ARGS=""
for d in $DAYS; do E_ARGS="$E_ARGS -e mars_relay_earth_scenario_inc_${SIZE}_d${d}"; done
[ "${APPEND:-0}" = "1" ] || : > "$IDS_FILE"
uv run --frozen main.py $E_ARGS -s lls -s fcp --p-tx "$P_TX" --p-base "$P_BASE" --rho 1 --store summary --debug \
    | grep -o 'report_id=[0-9]*' | cut -d= -f2 >> "$IDS_FILE"
for rho in $RHOS; do
    uv run --frozen main.py $E_ARGS -s lls_budget --p-tx "$P_TX" --p-base "$P_BASE" --rho "$rho" --store summary --debug \
        | grep -o 'report_id=[0-9]*' | cut -d= -f2 >> "$IDS_FILE"
done
echo "epoch sweep done $(date)"
