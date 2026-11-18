#!/bin/bash
# Lifespan-Aware v1 for one (P_tx, margin) pair over every mars_relay_earth size, in 4 parallel
# groups balanced by cost. Stores run summaries only (main.py --store summary).
# Usage: PTX=40 MARGIN=10 bash scripts/run_ptx_v1_pair.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${PTX:?}" "${MARGIN:?}"
P_BASE=${P_BASE:-200}
IDS_FILE=${IDS_FILE:-output/ptx_v1_all_ids.txt}
RHO=$(python3 -c "print($MARGIN/$PTX)")
E(){ for n in "$@"; do printf -- "-e mars_relay_earth_scenario_inc_%s " "$n"; done; }
run(){ uv run --frozen main.py $1 -s lifespan_aware --p-tx "$PTX" --p-base "$P_BASE" --rho "$RHO" --store summary --debug \
        | grep -o 'report_id=[0-9]*' | cut -d= -f2 >> "$IDS_FILE"; }
run "$(E 64 4 8 12 40)" & run "$(E 60 16 20 44)" & run "$(E 56 24 28 48)" & run "$(E 52 32 36)" & wait
