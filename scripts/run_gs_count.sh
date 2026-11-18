#!/bin/bash
# Same scenarios with all ground stations vs only 3 (one per DSN site), day 0, from
# tools/contact_plan_generator.py. VARIANT=full|gs3|gs12 (gsN = --gs-count N). Summaries only.
set -uo pipefail
cd "$(dirname "$0")/.."
SIZES=${SIZES:-"4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64"}
RHOS=${RHOS:-"0.05 0.1 0.2 0.3 0.5"}
VARIANT=${VARIANT:-gs3}
case "$VARIANT" in full) SUFFIX="_d0" ;; *) SUFFIX="_${VARIANT}_d0" ;; esac
IDS_FILE=${IDS_FILE:-output/gs_count_${VARIANT}_ids.txt}
E_ARGS=""; for n in $SIZES; do E_ARGS="$E_ARGS -e mars_relay_earth_scenario_inc_${n}${SUFFIX}"; done
: > "$IDS_FILE"
run(){ uv run --frozen main.py $E_ARGS "$@" --p-tx 40 --p-base 200 --store summary --debug \
        | grep -o 'report_id=[0-9]*' | cut -d= -f2 >> "$IDS_FILE"; }
run -s lls -s fcp --rho 1
for r in $RHOS; do run -s lls_budget --rho "$r"; done
echo done >> "$IDS_FILE.done"
