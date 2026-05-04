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
TIMEOUT_SEC=600

printf "%-6s  %-20s  %-20s  %-20s  %-10s\n" \
    "bits" "P_found" "Q_found" "N" "total_s"
printf '%s\n' "$(printf '%.0s-' {1..80})"

while IFS=',' read -r bits P Q P_pred Q_pred; do
    [[ "$bits" == "bits" ]] && continue
    (( bits > MAX_BITS )) && continue

    result=$(timeout "$TIMEOUT_SEC" "$BIN" -pq \
        "$P" "$Q" "$P_pred" "$Q_pred" \
        "$THREADS" 2>&1) && rc=0 || rc=$?

    if (( rc == 124 )); then
        printf "%-6s  %-20s  %-20s  %-20s  %-10s\n" \
            "$bits" "TIMEOUT" "" "" ">${TIMEOUT_SEC}s"
        continue
    fi

    P_found=$(echo "$result" | grep '^P = '      | awk '{print $3}')
    Q_found=$(echo "$result" | grep '^Q = '      | awk '{print $3}')
    N_val=$(  echo "$result" | grep '^N   = '    | awk '{print $3}')
    total_s=$(echo "$result" | grep 'total time' | grep -oE '[0-9]+\.[0-9]+')

    if [[ -n "$P_found" ]]; then
        printf "%-6s  %-20s  %-20s  %-20s  %-10s\n" \
            "$bits" "$P_found" "$Q_found" "$N_val" "${total_s}s"
    else
        printf "%-6s  %-20s  %-20s  %-20s  %-10s\n" \
            "$bits" "FAIL" "" "" ""
    fi

done < "$CSV"
