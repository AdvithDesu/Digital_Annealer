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

THREADS=$(nproc 2>/dev/null || echo 4)
MAX_BITS=50
TIMEOUT_SEC=300   # per case

printf "%-6s  %-18s  %-18s  %-18s  %-18s  %-10s\n" \
    "bits" "delta_p" "delta_q" "P_found" "Q_found" "total_s"
printf '%s\n' "$(printf '%.0s-' {1..90})"

while IFS=',' read -r bits P Q P_pred Q_pred; do
    # Skip header
    [[ "$bits" == "bits" ]] && continue
    # Stop at MAX_BITS
    (( bits > MAX_BITS )) && continue

    # Compute exact deltas from known error
    delta_p=$(( P >= P_pred ? P - P_pred : P_pred - P ))
    delta_q=$(( Q >= Q_pred ? Q - Q_pred : Q_pred - Q ))

    # Run with timeout; capture combined stdout+stderr
    result=$(timeout "$TIMEOUT_SEC" "$BIN" -pq \
        "$P" "$Q" "$P_pred" "$Q_pred" \
        "$delta_p" "$delta_q" \
        "$THREADS" 2>&1) && rc=0 || rc=$?

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
        P_found="—"
        Q_found="—"
    fi

    printf "%-6s  %-18s  %-18s  %-18s  %-18s  %-10s\n" \
        "$bits" "$delta_p" "$delta_q" "$P_found" "$Q_found" "$status"

done < "$CSV"
