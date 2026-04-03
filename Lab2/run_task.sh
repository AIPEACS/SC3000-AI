#!/usr/bin/env bash
# run_task.sh
# Runs the Exercise 1 Prolog task and captures trace output for the report.
#
# Usage:
#   bash run_task.sh               # auto trace, saves to part1/trace_output.txt
#   bash run_task.sh --interactive # opens swipl shell for manual stepping

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PL_FILE="$SCRIPT_DIR/part1/task.pl"
OUT_FILE="$SCRIPT_DIR/part1/trace_output.txt"

if [ ! -f "$PL_FILE" ]; then
    echo "ERROR: Could not find: $PL_FILE" >&2
    exit 1
fi

if [ "$1" = "--interactive" ]; then
    echo "Opening SWI-Prolog interactive shell..."
    echo "  Inside, type:  trace.  then  unethical(stevey).  to step through."
    echo ""
    swipl -s "$PL_FILE"
else
    echo "Running Exercise 1: The Smart Phone Rivalry (auto trace)..."
    echo "Saving output to: $OUT_FILE"
    echo ""

    # leash(-all) makes trace non-interactive (prints all steps without pausing)
    swipl -s "$PL_FILE" \
          -g 'leash(-all), trace, unethical(stevey), notrace, halt.' \
          2>&1 | sed 's/^[[:space:]]*//' | tee "$OUT_FILE"

    echo ""
    echo "Done. Trace saved to: $OUT_FILE"
    echo "Copy the content into your PDF report."
fi
