#!/bin/bash
# Benchmark factorize.sh across a table of (P, Q, start_temp) cases.
# Produces CSV with P, Q, start_temp, end_temp, alpha, P_pred, Q_pred, t_anneal.
#
# Usage: ./benchmark.sh [output_csv]

set -u

OUT="${1:-benchmark_results.csv}"
START_TEMP=10
ALPHA=0.95
SWEEPS=10
# End temp schedule: case idx 1..3 -> 1e-6, 4..6 -> 1e-7, 7..9 -> 1e-8, ...
# i.e. END_TEMP = 10^(-6 - floor((idx-1)/3))

# ── Test cases: P Q ──────────────────────────────────────────
CASES=(
    "151             163"
    "197             199"
    "223             229"
    "397             401"
    "577             587"
    "317             331"
    "547             691"
    "691             761"
    "1301            1471"
    "2351            2879"
    "4637            5021"
    "10831           12007"
    "17749           24611"
    "52009           68863"
    "148061          183343"
    "288803          312229"
    "591113          617401"
    "1348271         1821233"
    "3200003         3711091"
    "2499023255567   3099023255561"
    "642949953421331 812949953421371"
)

# Header
echo "P,Q,start_temp,end_temp,alpha,P_pred,Q_pred,t_anneal_sec,correct" > "$OUT"

idx=0
for case in "${CASES[@]}"; do
    idx=$((idx + 1))
    read -r P Q <<< "$case"

    # End-temp step: 1e-6 for cases 1-3, 1e-7 for 4-6, 1e-8 for 7-9, ...
    exp=$(( 6 + (idx - 1) / 3 ))
    END_TEMP="1e-${exp}"

    echo "===== [$idx] P=$P  Q=$Q  start_temp=$START_TEMP  end_temp=$END_TEMP ====="
    LOG=$(./factorize.sh -p "$P" -q "$Q" -x "$START_TEMP" -y "$END_TEMP" -c "$ALPHA" -m "$SWEEPS" 2>&1)
    rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "  factorize.sh failed (rc=$rc)"
        echo "$P,$Q,$START_TEMP,$END_TEMP,$ALPHA,ERROR,ERROR,ERROR,FAIL" >> "$OUT"
        continue
    fi

    # Parse annealing time: "  Annealing:   <t> s"
    T_ANNEAL=$(echo "$LOG" | grep -E "^\s*Annealing:" | awk '{print $2}' | tail -1)

    # Parse predicted factors: "Factors of N: P = <val>, Q = <val>"
    PRED_LINE=$(echo "$LOG" | grep "^Factors of " | tail -1)
    P_PRED=$(echo "$PRED_LINE" | sed -n 's/.*P = \([0-9]*\), Q = .*/\1/p')
    Q_PRED=$(echo "$PRED_LINE" | sed -n 's/.*Q = \([0-9]*\).*/\1/p')

    # Check CORRECT / INCORRECT
    if echo "$LOG" | grep -q "(CORRECT)"; then
        STATUS="CORRECT"
    else
        STATUS="INCORRECT"
    fi

    [[ -z "$P_PRED" ]]   && P_PRED="?"
    [[ -z "$Q_PRED" ]]   && Q_PRED="?"
    [[ -z "$T_ANNEAL" ]] && T_ANNEAL="?"

    echo "  P_pred=$P_PRED  Q_pred=$Q_PRED  t_anneal=${T_ANNEAL}s  [$STATUS]"
    echo "$P,$Q,$START_TEMP,$END_TEMP,$ALPHA,$P_PRED,$Q_PRED,$T_ANNEAL,$STATUS" >> "$OUT"
done

echo ""
echo "=== Results ==="
cat "$OUT"
echo ""
echo "Saved to $OUT"
