#!/bin/bash
# v2 model: rho sweep. One report per rho for lls_budget, plus one report with the unconstrained
# schedulers (their schedule does not depend on rho). Report ids are appended to output/rho_sweep_ids.txt;
# then run: uv run python analysis/scripts/rho_sweep.py --report-ids $(cat output/rho_sweep_ids.txt)
set -euo pipefail

P_TX=${P_TX:-40}        # electrical OCT draw [W] (placeholder until sourced)
P_BASE=${P_BASE:-200}   # spacecraft baseline load [W]
RHOS=${RHOS:-"0 0.025 0.05 0.075 0.1 0.15 0.2 0.3 0.5 0.75 1 1.5"}
EXPERIMENTS=${EXPERIMENTS:-"mars_relay_earth_scenario_inc_20"}
IDS_FILE=${IDS_FILE:-output/rho_sweep_ids.txt}

E_ARGS=""
for e in $EXPERIMENTS; do E_ARGS="$E_ARGS -e $e"; done

[ "${APPEND:-0}" = "1" ] || : > "$IDS_FILE"
for rho in $RHOS; do
    uv run main.py $E_ARGS -s lls_budget --p-tx "$P_TX" --p-base "$P_BASE" --rho "$rho" --debug \
        | grep -o 'report_id=[0-9]*' | cut -d= -f2 >> "$IDS_FILE"
done
[ "${SKIP_UNCONSTRAINED:-0}" = "1" ] || uv run main.py $E_ARGS -s lls -s fcp -s lifespan_aware --p-tx "$P_TX" --p-base "$P_BASE" --rho 1 --debug \
    | grep -o 'report_id=[0-9]*' | cut -d= -f2 >> "$IDS_FILE"
cat "$IDS_FILE"
