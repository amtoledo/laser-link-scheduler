#!/bin/bash
# Lifespan-Aware v1 runs for the P_tx x margin grid (its weights depend on the power profile).
# LLS-Budget and the unconstrained schedulers come from the rho sweep (scripts/run_rho_sweep.sh),
# because with a fixed data rate their schedules depend on rho only.
set -euo pipefail
P_BASE=${P_BASE:-200}
PTXS=${PTXS:-"20 40 60 80 100 120"}
MARGINS=${MARGINS:-"5 10 25 50"}
EXPERIMENTS=${EXPERIMENTS:-"mars_relay_earth_scenario_inc_20"}
IDS_FILE=${IDS_FILE:-output/ptx_v1_ids.txt}

E_ARGS=""
for e in $EXPERIMENTS; do E_ARGS="$E_ARGS -e $e"; done
[ "${APPEND:-0}" = "1" ] || : > "$IDS_FILE"
for m in $MARGINS; do
    for ptx in $PTXS; do
        rho=$(python3 -c "print($m/$ptx)")
        uv run main.py $E_ARGS -s lifespan_aware --p-tx "$ptx" --p-base "$P_BASE" --rho "$rho" --debug \
            | grep -o 'report_id=[0-9]*' | cut -d= -f2 >> "$IDS_FILE"
    done
done
cat "$IDS_FILE"
