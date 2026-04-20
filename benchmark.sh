#!/bin/bash
# Benchmark factorize.sh across a table of (P, Q, start_temp) cases.
# Produces CSV with P, Q, start_temp, end_temp, alpha, P_pred, Q_pred, t_anneal.
#
# Usage: ./benchmark.sh [output_csv]

set -u

OUT="${1:-benchmark_results.csv}"
END_TEMP=0.01
ALPHA=0.75
SWEEPS=10

# ── Test cases: P Q start_temp ───────────────────────────────
# (matches the attached table)
CASES=(
    "151        163       1e4"
    "197        199       1e4"
    "223        229       5e4"
    "397        401       1e5"
    "577        587       1e6"
    "317        331       1e6"
    "547        691       1e6"
    "691        761       5e6"
    "1301       1471      5e7"
    "2351       2879      1e11"
    "4637       5021      5e9"
    "10831      12007     1e11"
    "17749      24611     1e12"
    "52009      68863     5e13"
    "148061     183343    1e14"
    "288803     312229    5e15"
    "591113     617401    5e16"
    "1348271    1821233   5e18"
    "3200003    3711091   5e19"
)

# Header
echo "P,Q,start_temp,end_temp,alpha,P_pred,Q_pred,t_anneal_sec,correct" > "$OUT"

for case in "${CASES[@]}"; do
    read -r P Q START_TEMP <<< "$case"

    echo "===== P=$P  Q=$Q  start_temp=$START_TEMP ====="
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
