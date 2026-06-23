#!/usr/bin/env bash
# tune.sh -- sweep (m=t, safety, delta) for coppersmith and rank by throughput.
#
# Metric: each combo is run time-boxed for TIMEBOX seconds; whichever reaches the
# LARGEST spiral radius (radius ~ 2^R) in that fixed wall-time is covering the
# number line fastest, i.e. will find a far-off P soonest. (radius = step*rate,
# so this folds block-rate AND block-width into one directly-comparable number --
# no need to multiply rate*step by hand.)
#
# Usage:
#   ./tune.sh                       # uses the built-in 128-bit test triple
#   ./tune.sh P Q guess             # your own instance (guess should be ~2% off)
#   ./tune.sh P Q guess THREADS TIMEBOX
#
# Notes:
#   - Build coppersmith first (tuned NTL/GMP, -O3 -mcpu=neoverse-v2).
#   - The guess must be far enough off (~2%) that NO combo finishes inside
#     TIMEBOX, otherwise the comparison isn't apples-to-apples. The default does.
#   - Run ./coppersmith --selftest 64 once beforehand: it MUST print PASS, or the
#     row-limiting / delta settings are too aggressive and results are invalid.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/coppersmith}"

# Built-in 128-bit test triple (two 64-bit values; guess ~2% below P).
# Primality is irrelevant here -- it's a pure throughput probe, and the spiral
# can't reach any real factor within TIMEBOX anyway.
DEF_P=18446744073709551557
DEF_Q=18446744073709551533
DEF_G=18077809192235360526

P="${1:-$DEF_P}"
Q="${2:-$DEF_Q}"
GUESS="${3:-$DEF_G}"
THREADS="${4:-72}"
TIMEBOX="${5:-30}"

# Sweep grids (edit freely).
MS=(3 4 5 6)
SAFE=(2.0 1.5 1.3)
DELTA=(0.99 0.95 0.90)
ROWS=3

if [ ! -x "$BIN" ]; then
    echo "[error] binary not found/executable: $BIN" >&2
    echo "        build it, or set BIN=/path/to/coppersmith" >&2
    exit 1
fi

echo "[tune] binary  = $BIN"
echo "[tune] N = P*Q with P=$P Q=$Q"
echo "[tune] guess   = $GUESS"
echo "[tune] threads = $THREADS   timebox = ${TIMEBOX}s/combo"
echo "[tune] grid: m=t in {${MS[*]}}  safety in {${SAFE[*]}}  delta in {${DELTA[*]}}  rows=$ROWS"
total=$(( ${#MS[@]} * ${#SAFE[@]} * ${#DELTA[@]} ))
echo "[tune] $total combos x ${TIMEBOX}s ~ $(( total * TIMEBOX / 60 )) min"
echo

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

run=0
for m in "${MS[@]}"; do
  for s in "${SAFE[@]}"; do
    for d in "${DELTA[@]}"; do
      run=$((run+1))
      printf "[%2d/%2d] m=t=%s safety=%s delta=%s ... " "$run" "$total" "$m" "$s" "$d"

      out=$(timeout "${TIMEBOX}" "$BIN" -pq "$P" "$Q" "$GUESS" "$THREADS" "$m" "$m" "$s" "$d" "$ROWS" 2>&1)

      step=$(printf '%s\n' "$out" | grep -m1 'step ~'    | sed -E 's/.*step ~ 2\^([0-9.]+).*/\1/')
      last=$(printf '%s\n' "$out" | grep '\[search\]'    | tail -1)
      rad=$( printf '%s\n' "$last" | sed -E 's/.*radius~2\^([0-9.]+).*/\1/')
      rate=$(printf '%s\n' "$last" | sed -E 's/.*rate=([0-9]+).*/\1/')
      found=$(printf '%s\n' "$out" | grep -c '^P = ')

      if [ "$found" -gt 0 ]; then
          # Finished early -> effectively infinite throughput for this probe.
          sortkey=999 ; radshow="FOUND"
      else
          sortkey="${rad:-0}" ; radshow="${rad:-NA}"
      fi
      printf "radius=2^%s  rate=%s blk/s\n" "$radshow" "${rate:-NA}"
      # store: sortkey  m  safety  delta  radius  rate  step
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$sortkey" "$m/$m" "$s" "$d" "$radshow" "${rate:-NA}" "${step:-NA}" >> "$TMP"
    done
  done
done

echo
echo "================ ranked (highest radius after ${TIMEBOX}s = fastest) ================"
printf "%-7s %-8s %-8s %-12s %-12s %-10s\n" "m/t" "safety" "delta" "radius_2^" "rate(blk/s)" "step_2^"
sort -t$'\t' -k1 -rn "$TMP" | while IFS=$'\t' read -r key m s d rad rate step; do
    printf "%-7s %-8s %-8s %-12s %-12s %-10s\n" "$m" "$s" "$d" "$rad" "$rate" "$step"
done

best=$(sort -t$'\t' -k1 -rn "$TMP" | head -1)
bm=$(echo "$best" | cut -f2); bs=$(echo "$best" | cut -f3); bd=$(echo "$best" | cut -f4)
echo
echo "[tune] best combo: m=t=$bm safety=$bs delta=$bd"
echo "[tune] run it with:"
echo "       numactl --physcpubind=0-$((THREADS-1)) --localalloc \\"
echo "         $BIN -pq <P> <Q> <guess> $THREADS ${bm%/*} ${bm#*/} $bs $bd $ROWS"
echo
echo "[tune] IMPORTANT: confirm the winning delta still recovers:"
echo "       $BIN --selftest 64    # must print PASS"
