#!/usr/bin/env bash
# batch_coppersmith.sh -- run the Coppersmith pipeline over every row of a CSV
# and record timing/results. Survives SSH disconnects when launched via nohup.
#
# Input CSV columns (header required):  bits,P,Q,P_pred,Q_pred
#   - The order of P/Q and P_pred/Q_pred in the file is IGNORED: this script
#     normalizes so the smaller is "P"/"P_pred" and the larger "Q"/"Q_pred",
#     then targets the SMALLER factor with
#         guess = average( P_pred , N / Q_pred ) ,   N = P*Q.
#
# Output:
#   coppersmith_results.csv  -- one row per case: bits,P,Q,guess,err%,Pfound,Qfound,wall_s,status
#   batch_logs/case_<bits>bit.log -- full per-case output (config + [search] + result)
#
# Usage:
#   ./batch_coppersmith.sh [input.csv] [threads] [timeout_sec_per_case] [max_bits]
# Defaults: input=all_runs.csv  threads=70  timeout=21600 (6h)  max_bits=52
#   max_bits: skip any row whose 'bits' exceeds this (the big cases dominate time).
#
# Run detached (survives disconnect):
#   nohup ./batch_coppersmith.sh all_runs.csv 70 21600 52 > batch.out 2>&1 &
#   echo $! > batch.pid ; disown
#   tail -f batch.out          # or: tail -f coppersmith_results.csv

set -uo pipefail
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$SD/coppersmith}"
INPUT="${1:-$SD/all_runs.csv}"
THREADS="${2:-70}"
TIMEOUT_PER="${3:-21600}"
MAX_BITS="${4:-52}"

# Tuned lattice params (from tune.sh on the GH200). Edit here if needed.
M=5; T=5; SAFETY=1.3; DELTA=0.9; ROWS=3

OUT="$SD/coppersmith_results.csv"
LOGDIR="$SD/batch_logs"
mkdir -p "$LOGDIR"

[ -x "$BIN" ]   || { echo "[error] binary not found/executable: $BIN" >&2; exit 1; }
[ -f "$INPUT" ] || { echo "[error] input csv not found: $INPUT" >&2; exit 1; }
command -v python3 >/dev/null || { echo "[error] python3 required (big-int arithmetic)" >&2; exit 1; }

echo "bits,P,Q,P_guess,err_pct,P_found,Q_found,wall_s,status" > "$OUT"
echo "[batch] input=$INPUT"
echo "[batch] bin=$BIN  threads=$THREADS  timeout=${TIMEOUT_PER}s/case  max_bits=$MAX_BITS"
echo "[batch] params: m=$M t=$T safety=$SAFETY delta=$DELTA rows=$ROWS"
echo "[batch] results -> $OUT   per-case logs -> $LOGDIR/"
echo

# Strip CR (Windows line endings), skip header, iterate rows.
tr -d '\r' < "$INPUT" | tail -n +2 | while IFS=, read -r bits Praw Qraw Ppraw Qpraw; do
    bits="$(printf '%s' "$bits" | tr -dc '0-9')"   # clean to pure integer
    [ -z "$bits" ] && continue                     # skip blank lines
    if [ "$bits" -gt "$MAX_BITS" ]; then
        echo "[batch] bits=$bits > max_bits=$MAX_BITS -> skipping"
        continue
    fi

    # Normalize (sort) + compute N and guess with big-int python.
    vals=$(python3 -c "
P,Q,Pp,Qp=int($Praw),int($Qraw),int($Ppraw),int($Qpraw)
P,Q=min(P,Q),max(P,Q)
Pp,Qp=min(Pp,Qp),max(Pp,Qp)
N=P*Q
g=(Pp + N//Qp)//2
err=abs(g-P)/P*100
print(P,Q,g,'%.3f'%err)
") || { echo "[batch] bits=$bits  PYTHON ERROR, skipping"; continue; }
    read -r Ps Qs G ERR <<< "$vals"

    log="$LOGDIR/case_${bits}bit.log"
    printf "[batch] bits=%-3s err=%6s%% ... " "$bits" "$ERR"

    t0=$(date +%s.%N)
    timeout "$TIMEOUT_PER" "$BIN" -pq "$Ps" "$Qs" "$G" \
        "$THREADS" "$M" "$T" "$SAFETY" "$DELTA" "$ROWS" > "$log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print('%.2f'%($t1-$t0))")

    Pf=$(grep -m1 '^P = ' "$log" | awk '{print $3}')
    Qf=$(grep -m1 '^Q = ' "$log" | awk '{print $3}')

    if   [ "$rc" -eq 124 ]; then status=TIMEOUT
    elif [ -n "$Pf" ];      then status=OK
    else                         status=FAIL; fi

    echo "$bits,$Ps,$Qs,$G,$ERR,$Pf,$Qf,$wall,$status" >> "$OUT"
    printf "%-7s wall=%ss  P_found=%s\n" "$status" "$wall" "${Pf:-<none>}"
done

echo
echo "[batch] DONE. Summary:"
column -t -s, "$OUT" 2>/dev/null || cat "$OUT"
