#!/bin/bash
# Run CUDA SA on a single problem instance and print a formatted summary.
#
# Usage:
#   ./run_single_cuda.sh <N> [options]
#
# Required:
#   <N>                    problem suffix (matches J_values_<N>.csv etc. in bin_SI/)
#
# Options:
#   -x, --start-temp <T>   starting temperature       (default: 100.0)
#   -y, --stop-temp  <T>   stopping temperature       (default: 0.1)
#   -c, --alpha      <a>   geometric cooling rate     (default: 0.95)
#   -m, --sweeps     <M>   sweeps per beta            (default: 10)
#   -s, --seed       <S>   RNG seed                   (default: auto)
#   -e, --no-early-stop    disable early termination
#   -d, --debug            enable debug output from the binary

set -euo pipefail

BINARY="./build/annealer_gpu_SI/annealer_gpu_SI"
DIR="bin_SI"

# Defaults
START_TEMP=100.0
STOP_TEMP=0.1
ALPHA=0.95
SWEEPS=10
SEED=""
EARLY_STOP_FLAG=""
DEBUG_FLAG=""

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <N> [options]" >&2
    echo "Run '$0 --help' for details." >&2
    exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '2,21p' "$0"
    exit 0
fi

N="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        -x|--start-temp)   START_TEMP="$2";   shift 2 ;;
        -y|--stop-temp)    STOP_TEMP="$2";    shift 2 ;;
        -c|--alpha)        ALPHA="$2";        shift 2 ;;
        -m|--sweeps)       SWEEPS="$2";       shift 2 ;;
        -s|--seed)         SEED="$2";         shift 2 ;;
        -e|--no-early-stop) EARLY_STOP_FLAG="-e"; shift ;;
        -d|--debug)        DEBUG_FLAG="-d";   shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

R="$DIR/row_ptr_${N}.csv"
C="$DIR/col_idx_${N}.csv"
V="$DIR/J_values_${N}.csv"
H="$DIR/h_vector_${N}.csv"

for f in "$R" "$C" "$V" "$H"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing input file: $f" >&2
        exit 1
    fi
done

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binary not found or not executable: $BINARY" >&2
    exit 1
fi

CMD=("$BINARY" -R "$R" -C "$C" -V "$V" -l "$H"
     -x "$START_TEMP" -y "$STOP_TEMP" -c "$ALPHA" -m "$SWEEPS")
[[ -n "$SEED" ]]            && CMD+=(-s "$SEED")
[[ -n "$EARLY_STOP_FLAG" ]] && CMD+=("$EARLY_STOP_FLAG")
[[ -n "$DEBUG_FLAG" ]]      && CMD+=("$DEBUG_FLAG")

echo "===== Running N=$N ====="
echo "Command: ${CMD[*]}"
echo

# Stream output live AND capture it for later parsing.
TMP_OUT=$(mktemp)
trap 'rm -f "$TMP_OUT"' EXIT

START_NS=$(date +%s%N)
set +e
"${CMD[@]}" 2>&1 | tee "$TMP_OUT"
STATUS=${PIPESTATUS[0]}
set -e
END_NS=$(date +%s%N)

if [[ $STATUS -ne 0 ]]; then
    echo "ERROR: binary exited with status $STATUS" >&2
    exit 1
fi

WALL_SEC=$(awk "BEGIN { printf \"%.6f\", ($END_NS - $START_NS) / 1e9 }")
OUTPUT=$(cat "$TMP_OUT")
echo

# --- Parse fields out of the binary's stdout -----------------------------
grab() { echo "$OUTPUT" | grep -E "$1" | head -1; }

NUM_SPINS=$(grab "num_spins = "     | sed -n 's/.*num_spins = \([0-9]*\).*/\1/p')
NNZ=$(grab       "nnz = "           | sed -n 's/.*nnz = \([0-9]*\).*/\1/p')
NUM_DENSE=$(grab "Bin sizes"        | sed -n 's/.*(dense): *\([0-9]*\).*/\1/p')
NUM_SPARSE=$(grab "Bin sizes"       | sed -n 's/.*sparse: *\([0-9]*\).*/\1/p')
NUM_COLORS=$(grab "num_colors = "   | sed -n 's/.*num_colors = \([0-9]*\).*/\1/p')
COLOR_SIZES=$(grab "color-class"    | sed -n 's/.*color-class sizes: *//p')
NNZ_FB=$(grab    "non-hub fallbacks"| sed -n 's/.*non-hub fallbacks=\([0-9]*\) .*/\1/p')
FB_PCT=$(grab    "non-hub fallbacks"| sed -n 's/.*(\([0-9.]*\)%).*/\1/p')

INIT_ENERGY=$(grab  "initial energy"     | sed -n 's/.*initial energy:* *\(-\?[0-9eE.+-]*\).*/\1/p')
FINAL_ENERGY=$(grab "total energy value" | sed -n 's/.*total energy value: *\(-\?[0-9eE.+-]*\).*/\1/p')
BEST_ENERGY=$(grab  "best engy"          | sed -n 's/.*best engy *\(-\?[0-9eE.+-]*\).*/\1/p')

T_LOAD=$(grab   "Load data:"    | awk '{print $3}')
T_SETUP=$(grab  "Setup (GPU):"  | awk '{print $3}')
T_ANNEAL=$(grab "Annealing:"    | awk '{print $2}')
T_TOTAL=$(grab  "Total:"        | awk '{print $2}')

fmt() { [[ -z "$1" ]] && echo "n/a" || echo "$1"; }

# --- Summary -------------------------------------------------------------
echo "===================== Summary ====================="
printf "  N (suffix):           %s\n" "$N"
printf "  num_spins:            %s\n" "$(fmt "$NUM_SPINS")"
printf "  nnz:                  %s\n" "$(fmt "$NNZ")"
printf "  dense / sparse bins:  %s / %s\n" "$(fmt "$NUM_DENSE")" "$(fmt "$NUM_SPARSE")"
printf "  num_colors:           %s\n" "$(fmt "$NUM_COLORS")"
if [[ -n "$COLOR_SIZES" ]]; then
    printf "  color-class sizes:    %s\n" "$COLOR_SIZES"
fi
if [[ -n "$NNZ_FB" ]]; then
    printf "  non-hub fallbacks:    %s (%s%%)\n" "$NNZ_FB" "$(fmt "$FB_PCT")"
fi
echo "----------------------------------------------------"
printf "  initial energy:       %s\n" "$(fmt "$INIT_ENERGY")"
printf "  final energy:         %s\n" "$(fmt "$FINAL_ENERGY")"
printf "  best energy (anneal): %s\n" "$(fmt "$BEST_ENERGY")"
echo "----------------------------------------------------"
printf "  t_load:               %s s\n" "$(fmt "$T_LOAD")"
printf "  t_setup:              %s s\n" "$(fmt "$T_SETUP")"
printf "  t_anneal:             %s s\n" "$(fmt "$T_ANNEAL")"
printf "  t_total (binary):     %s s\n" "$(fmt "$T_TOTAL")"
printf "  wall clock:           %s s\n" "$WALL_SEC"
echo "===================================================="
