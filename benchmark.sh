#!/bin/bash
# Sweep factorize.sh across a range of bit sizes.
# For each B in BITS_LIST, generate two random B-bit primes (P, Q), run
# factorize.sh on N=P*Q, and record solution-quality + timing metrics to CSV.
#
# Columns: bits, P, Q, N, P_pred, Q_pred, correct, t_construct, t_anneal, best_energy
#
# Usage: ./benchmark.sh [options] [output_csv]
#
# Options (all forwarded to factorize.sh):
#   -x <T>    start temp; "auto" for Ben-Ameur estimation     (default: auto)
#   -y <T>    stop temp                                        (default: 1e-8)
#   -c <a>    geometric cooling rate                           (default: 0.95)
#   -m <M>    sweeps per beta                                  (default: 10)
#   -a <F>    target uphill accept rate (only for -x auto)     (default: 0.5)
#   -B <list> comma-separated bit sizes (e.g. 8,16,24,32)
#             default: 8,10,12,...,62
#   -o <csv>  output csv path (also accepted as positional)    (default: benchmark_results.csv)
#   -h        show this help

set -u

# ── SA defaults (overridable via CLI) ────────────────────────
START_TEMP="auto"        # "auto" -> Ben-Ameur estimation in CUDA SA
STOP_TEMP=1e-8
ALPHA=0.95
SWEEPS=10
ACCEPT_RATE=0.5          # target uphill accept rate when -x auto

# ── Bit sizes default: 8, 10, 12, ..., 62 ────────────────────
BITS_LIST=()
for b in $(seq 8 2 62); do BITS_LIST+=("$b"); done

OUT=""

# ── Parse CLI ────────────────────────────────────────────────
while getopts ":x:y:c:m:a:B:o:h" opt; do
    case "$opt" in
        x) START_TEMP="$OPTARG" ;;
        y) STOP_TEMP="$OPTARG" ;;
        c) ALPHA="$OPTARG" ;;
        m) SWEEPS="$OPTARG" ;;
        a) ACCEPT_RATE="$OPTARG" ;;
        B) IFS=',' read -ra BITS_LIST <<< "$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        h) sed -n '2,20p' "$0"; exit 0 ;;
        \?) echo "Unknown option: -$OPTARG" >&2; exit 1 ;;
        :)  echo "Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# Positional fallback for output csv
[[ -z "$OUT" && $# -gt 0 ]] && OUT="$1"
[[ -z "$OUT" ]] && OUT="benchmark_results.csv"

echo "Settings: start=$START_TEMP stop=$STOP_TEMP alpha=$ALPHA sweeps=$SWEEPS accept=$ACCEPT_RATE"
echo "Bits: ${BITS_LIST[*]}"
echo "Output: $OUT"
echo

# ── Random prime generator ───────────────────────────────────
gen_prime() {
    # gen_prime <bits> -> stdout
    python3 -c "
import os, random
from sympy import randprime
random.seed(int.from_bytes(os.urandom(8), 'little'))
b = $1
print(randprime(2**(b-1), 2**b))
"
}

# Header
echo "bits,P,Q,N,P_pred,Q_pred,correct,t_construct,t_anneal,best_energy" > "$OUT"

for BITS in "${BITS_LIST[@]}"; do
    P=$(gen_prime "$BITS")
    Q=$(gen_prime "$BITS")
    while [[ "$P" == "$Q" ]]; do
        Q=$(gen_prime "$BITS")
    done
    N=$(python3 -c "print($P * $Q)")

    echo "===== bits=$BITS  P=$P  Q=$Q  N=$N ====="

    LOG=$(./factorize.sh -p "$P" -q "$Q" \
            -x "$START_TEMP" -y "$STOP_TEMP" -c "$ALPHA" -m "$SWEEPS" \
            --auto-accept-rate "$ACCEPT_RATE" 2>&1)
    rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "  factorize.sh failed (rc=$rc)"
        echo "$BITS,$P,$Q,$N,ERROR,ERROR,FAIL,?,?,?" >> "$OUT"
        continue
    fi

    # ── Parse metrics from log ──────────────────────────────
    # Construction time: first occurrence (Step 1, QUBO build)
    T_CONSTRUCT=$(echo "$LOG" | grep -E "^Total construction time:" | awk '{print $4}' | head -1)
    T_ANNEAL=$(echo "$LOG"    | grep -E "^[[:space:]]*Annealing:"   | awk '{print $2}' | tail -1)
    # Use the recomputed best-state energy (re-evaluated after annealing — less drift than the running tracker)
    BEST_E=$(echo "$LOG" | grep -E "best-state[[:space:]]+energy[[:space:]]+\(recomputed\)" \
                        | awk -F': ' '{print $2}' | tail -1)

    PRED_LINE=$(echo "$LOG" | grep "^Factors of " | tail -1)
    P_PRED=$(echo "$PRED_LINE" | sed -n 's/.*P = \([0-9]*\), Q = .*/\1/p')
    Q_PRED=$(echo "$PRED_LINE" | sed -n 's/.*Q = \([0-9]*\).*/\1/p')

    if echo "$LOG" | grep -q "(CORRECT)"; then
        STATUS="CORRECT"
    else
        STATUS="INCORRECT"
    fi

    [[ -z "$P_PRED" ]]      && P_PRED="?"
    [[ -z "$Q_PRED" ]]      && Q_PRED="?"
    [[ -z "$T_CONSTRUCT" ]] && T_CONSTRUCT="?"
    [[ -z "$T_ANNEAL" ]]    && T_ANNEAL="?"
    [[ -z "$BEST_E" ]]      && BEST_E="?"

    echo "  P_pred=$P_PRED  Q_pred=$Q_PRED  best_E=$BEST_E  t_J=${T_CONSTRUCT}s  t_SA=${T_ANNEAL}s  [$STATUS]"
    echo "$BITS,$P,$Q,$N,$P_PRED,$Q_PRED,$STATUS,$T_CONSTRUCT,$T_ANNEAL,$BEST_E" >> "$OUT"
done

echo ""
echo "=== Results ==="
cat "$OUT"
echo ""
echo "Saved to $OUT"
