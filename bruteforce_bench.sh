#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/Bruteforce.cpp"
BIN="$SCRIPT_DIR/Bruteforce"
CSV="$SCRIPT_DIR/benchmark_results.csv"

# Compile if binary is missing or source is newer
if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
    echo "[build] Compiling Bruteforce..."
    g++ -O3 -std=c++17 -pthread -o "$BIN" "$SRC" || exit 1
    echo "[build] Done."
fi

MAX_BITS=50
TIMEOUT_SEC=600   # per case

# Round val up to (leading_digit + 1) * 10^(ndigits-1).
# Examples: 2345 -> 3000, 56776 -> 60000, 1 -> 2, 9 -> 10.
ceil_delta() {
    local val=$1
    (( val == 0 )) && { echo 1; return; }
    local digits=${#val}
    local pow=1
    for (( i=1; i<digits; i++ )); do pow=$(( pow * 10 )); done
    local leading=$(( val / pow ))
    echo $(( (leading + 1) * pow ))
}

printf "%-6s  %-18s  %-18s  %-18s  %-18s  %-10s\n" \
    "bits" "delta_p" "delta_q" "P_found" "Q_found" "total_s"
printf '%s\n' "$(printf '%.0s-' {1..90})"

while IFS=',' read -r bits P Q P_pred Q_pred; do
    # Skip header
    [[ "$bits" == "bits" ]] && continue
    # Stop at MAX_BITS
    (( bits > MAX_BITS )) && continue

    # Compute deltas from known error, rounded up to next leading-digit boundary
    raw_dp=$(( P >= P_pred ? P - P_pred : P_pred - P ))
    raw_dq=$(( Q >= Q_pred ? Q - Q_pred : Q_pred - Q ))
    delta_p=$(ceil_delta "$raw_dp")
    delta_q=$(ceil_delta "$raw_dq")

    # Run with timeout; capture combined stdout+stderr
    result=$(timeout "$TIMEOUT_SEC" "$BIN" -pq \
        "$P" "$Q" "$P_pred" "$Q_pred" \
        "$delta_p" "$delta_q" 2>&1) && rc=0 || rc=$?

    if (( rc == 124 )); then
        printf "%-6s  %-18s  %-18s  %-18s  %-18s  %-10s\n" \
            "$bits" "$delta_p" "$delta_q" "TIMEOUT" "" ">${TIMEOUT_SEC}s"
        continue
    fi

    P_found=$(echo "$result"  | grep '^P = '          | awk '{print $3}')
    Q_found=$(echo "$result"  | grep '^Q = '          | awk '{print $3}')
    total_s=$(echo "$result"  | grep 'total time'     | grep -oE '[0-9]+\.[0-9]+')

    if [[ -n "$P_found" ]]; then
        status="${total_s}s"
    else
        status="FAIL"
        P_found="not found"
        Q_found="not found"
    fi

    printf "%-6s  %-18s  %-18s  %-18s  %-18s  %-10s\n" \
        "$bits" "$delta_p" "$delta_q" "$P_found" "$Q_found" "$status"

done < "$CSV"
