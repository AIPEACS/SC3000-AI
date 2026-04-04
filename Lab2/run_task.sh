#!/usr/bin/env bash
# run_task.sh
# Runs Lab 2 Prolog tasks and captures trace output for the report.
#
# Usage:
#   bash run_task.sh               # prompts for part selection
#   bash run_task.sh --interactive # opens swipl shell for manual stepping

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Part selection ---
echo "Select part to run:"
echo "  1 - Exercise 1: Smart Phone Rivalry"
echo "  2 - Exercise 2: Royal Family Succession"
read -rp "Enter 1 or 2: " choice

if [ "$choice" = "1" ]; then

    PL_FILE="$SCRIPT_DIR/part1/task.pl"
    OUT_FILE="$SCRIPT_DIR/part1/trace_output.txt"

    if [ ! -f "$PL_FILE" ]; then echo "ERROR: Could not find: $PL_FILE" >&2; exit 1; fi

    if [ "$1" = "--interactive" ]; then
        echo "Opening SWI-Prolog interactive shell..."
        echo "  Inside, type:  trace.  then  unethical(stevey)."
        echo ""
        swipl -s "$PL_FILE"
    else
        echo ""
        echo "Running Exercise 1: The Smart Phone Rivalry..."
        echo "Saving output to: $OUT_FILE"
        echo ""

        swipl -s "$PL_FILE" \
              -g 'leash(-all), trace, unethical(stevey), notrace, halt.' \
              2>&1 | sed 's/^[[:space:]]*//' | tee "$OUT_FILE"

        echo ""
        echo "Done. Trace saved to: $OUT_FILE"
    fi

elif [ "$choice" = "2" ]; then

    PL21="$SCRIPT_DIR/part2/task2_1.pl"
    PL22="$SCRIPT_DIR/part2/task2_2.pl"
    OUT21="$SCRIPT_DIR/part2/task2_1_LOS.txt"
    OUT22="$SCRIPT_DIR/part2/task2_2_LOS.txt"

    if [ ! -f "$PL21" ]; then echo "ERROR: Could not find: $PL21" >&2; exit 1; fi
    if [ ! -f "$PL22" ]; then echo "ERROR: Could not find: $PL22" >&2; exit 1; fi

    if [ "$1" = "--interactive" ]; then
        echo "Select sub-task:"
        echo "  1 - Old succession rule (task2_1.pl)"
        echo "  2 - New succession rule (task2_2.pl)"
        read -rp "Enter 1 or 2: " sub
        if [ "$sub" = "1" ]; then swipl -s "$PL21"
        else                       swipl -s "$PL22"
        fi
    else
        # --- 2.1: Old rule ---
        echo ""
        echo "Running Exercise 2.1: Old Succession Rule (males first, then females)..."
        echo "Saving output to: $OUT21"
        echo ""

        swipl -s "$PL21" \
              -g 'leash(-all), trace, line_of_succession(elizabeth, X), notrace, halt.' \
              2>&1 | sed 's/^[[:space:]]*//' | tee "$OUT21"

        # --- 2.2: New rule ---
        echo ""
        echo "Running Exercise 2.2: New Succession Rule (birth order, gender-independent)..."
        echo "Saving output to: $OUT22"
        echo ""

        swipl -s "$PL22" \
              -g 'leash(-all), trace, line_of_succession(elizabeth, X), notrace, halt.' \
              2>&1 | sed 's/^[[:space:]]*//' | tee "$OUT22"

        echo ""
        echo "Done. Traces saved to part2/ for your report."
    fi

else
    echo "ERROR: Invalid choice. Please enter 1 or 2." >&2
    exit 1
fi
