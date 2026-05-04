#!/bin/bash
# Multi-seed solution-quality benchmark.
#
# For each bit size B in BITS_LIST:
#   1. generate random B-bit primes P, Q
#   2. build the QUBO once (cheap, deterministic for given N)
#   3. run SA `num_seeds` times with independent random seeds, storing each
#      run's best energy and best spin state
#   4. across the runs, pick the best (lowest energy) and copy its spin state
#      to a separate "winner" file; record its energy
#   5. post-process the winner to get predicted P, Q and verify correctness
#
# Outputs (default location: multiseed_bench/, alongside this script):
#   multiseed_bench/results.csv               one summary row per bit size
#   multiseed_bench/results_per_run.csv       one row per (B, seed)
#   multiseed_bench/spins/spins_<N>_seed<i>   per-run spin states
#   multiseed_bench/spins/spins_<N>_best      winning spin state
#   multiseed_bench/spins/best_energy_<N>.txt winning best-energy value
#
# Usage: ./multiseed_bench/multiseed_benchmark.sh [options]
#
# Options:
#   -k <K>    independent SA runs per case             (default: 10)
#   -x <T>    SA start temp; "auto" for Ben-Ameur      (default: auto)
#   -y <T>    SA stop temp                              (default: 1e-8)
#   -c <a>    geometric cooling rate                    (default: 0.95)
#   -m <M>    sweeps per beta                           (default: 10)
#   -a <F>    target uphill accept rate (-x auto)       (default: 0.5)
#   -B <list> comma-separated bit sizes                 (default: 8,10,...,62)
#   -o <name> output prefix relative to repo root       (default: multiseed_bench/results)
#   -K        clean per-seed spin files after each B (only keep winner)
#   -h        show this help

set -u

# ── Locate script + repo root, then operate from repo root so the existing
#    relative binary paths (./QUBO_Construction/, ./build/) keep working
#    regardless of where the user invokes this script from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ─────────────────────────────────────────────────
NUM_SEEDS=10
START_TEMP="auto"
STOP_TEMP=1e-8
ALPHA=0.95
SWEEPS=10
ACCEPT_RATE=0.5
OUT_PREFIX="multiseed_bench/results"
KEEP_PER_SEED_SPINS=true

BITS_LIST=()
for b in $(seq 8 2 62); do BITS_LIST+=("$b"); done

# ── Binaries / paths (relative to REPO_ROOT, which is the cwd) ───────
QUBO_BIN="./QUBO_Construction/qubo_factorization"
SA_BIN="./build/annealer_gpu_SI/annealer_gpu_SI"
QUBO_SRC="QUBO_Construction/QUBO_Integer_Factorization.cpp"
CSR_DIR="bin_SI"
META_DIR="qubo_metadata"
RESULTS_DIR="results"
MULTISEED_DIR="multiseed_bench/spins"

# ── Parse CLI ────────────────────────────────────────────────
while getopts ":k:x:y:c:m:a:B:o:Kh" opt; do
    case "$opt" in
        k) NUM_SEEDS="$OPTARG" ;;
        x) START_TEMP="$OPTARG" ;;
        y) STOP_TEMP="$OPTARG" ;;
        c) ALPHA="$OPTARG" ;;
        m) SWEEPS="$OPTARG" ;;
        a) ACCEPT_RATE="$OPTARG" ;;
        B) IFS=',' read -ra BITS_LIST <<< "$OPTARG" ;;
        o) OUT_PREFIX="$OPTARG" ;;
        K) KEEP_PER_SEED_SPINS=false ;;
        h) sed -n '2,33p' "$0"; exit 0 ;;
        \?) echo "Unknown option: -$OPTARG" >&2; exit 1 ;;
        :)  echo "Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

OUT_SUMMARY="${OUT_PREFIX}.csv"
OUT_PERRUN="${OUT_PREFIX}_per_run.csv"
OUT_DIR=$(dirname "$OUT_PREFIX")
[[ -n "$OUT_DIR" && "$OUT_DIR" != "." ]] && mkdir -p "$OUT_DIR"
mkdir -p "$CSR_DIR" "$META_DIR" "$RESULTS_DIR" "$MULTISEED_DIR"

echo "Settings: num_seeds=$NUM_SEEDS  start=$START_TEMP stop=$STOP_TEMP alpha=$ALPHA sweeps=$SWEEPS accept=$ACCEPT_RATE"
echo "Bits: ${BITS_LIST[*]}"
echo "Outputs: $OUT_SUMMARY (summary), $OUT_PERRUN (per run)"
echo "Spins dir: $MULTISEED_DIR (keep per-seed: $KEEP_PER_SEED_SPINS)"
echo

# ── Build binaries if missing ────────────────────────────────
if [[ ! -x "$QUBO_BIN" || "$QUBO_SRC" -nt "$QUBO_BIN" ]]; then
    echo "Building QUBO binary..."
    g++ -O2 -std=c++17 -o "$QUBO_BIN" "$QUBO_SRC"
fi
if [[ ! -x "$SA_BIN" ]]; then
    echo "Building SA binary..."
    mkdir -p build
    cmake -S . -B build
    make -C build -j"$(nproc)"
fi

# ── Helpers ──────────────────────────────────────────────────
gen_prime() {
    python3 -c "
import os, random
from sympy import randprime
random.seed(int.from_bytes(os.urandom(8), 'little'))
b = $1
print(randprime(2**(b-1), 2**b))
"
}

# ── CSV headers ──────────────────────────────────────────────
echo "bits,P,Q,N,num_seeds,t_construct,e_min,e_mean,e_max,t_anneal_mean,P_pred,Q_pred,correct" > "$OUT_SUMMARY"
echo "bits,P,Q,N,seed_idx,seed,best_energy,t_anneal" > "$OUT_PERRUN"

# ── Main loop ────────────────────────────────────────────────
for BITS in "${BITS_LIST[@]}"; do
    P=$(gen_prime "$BITS")
    Q=$(gen_prime "$BITS")
    while [[ "$P" == "$Q" ]]; do Q=$(gen_prime "$BITS"); done
    N=$(python3 -c "print($P * $Q)")

    echo "===== bits=$BITS  P=$P  Q=$Q  N=$N ====="

    # --- Step 1: build QUBO once ---------------------------------------
    echo "  [1] building QUBO ..."
    QUBO_LOG=$("$QUBO_BIN" "$N" --csr-dir "$CSR_DIR/" --meta-dir "$META_DIR/" 2>&1)
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  QUBO build failed:"
        echo "$QUBO_LOG" | tail -10
        echo "$BITS,$P,$Q,$N,$NUM_SEEDS,?,?,?,?,?,ERROR,ERROR,FAIL" >> "$OUT_SUMMARY"
        continue
    fi
    T_CONSTRUCT=$(echo "$QUBO_LOG" | grep -E "^Total construction time:" | awk '{print $4}' | head -1)
    [[ -z "$T_CONSTRUCT" ]] && T_CONSTRUCT="?"

    # If the simplifier resolves N entirely, no SA needed.
    NVARS=$(echo "$QUBO_LOG" | grep -E "^Active QUBO variables:" | awk '{print $4}' | head -1)
    if [[ -n "$NVARS" && "$NVARS" == "0" ]]; then
        echo "  QUBO has 0 active variables -- fully solved by preprocessing."
        PRED_LINE=$(echo "$QUBO_LOG" | grep "^Factors of " | tail -1)
        P_PRED=$(echo "$PRED_LINE" | sed -n 's/.*P = \([0-9]*\), Q = .*/\1/p')
        Q_PRED=$(echo "$PRED_LINE" | sed -n 's/.*Q = \([0-9]*\).*/\1/p')
        [[ -z "$P_PRED" ]] && P_PRED="?"
        [[ -z "$Q_PRED" ]] && Q_PRED="?"
        if echo "$QUBO_LOG" | grep -q "(CORRECT)"; then STATUS="CORRECT"; else STATUS="INCORRECT"; fi
        echo "$BITS,$P,$Q,$N,0,$T_CONSTRUCT,0,0,0,0,$P_PRED,$Q_PRED,$STATUS" >> "$OUT_SUMMARY"
        continue
    fi

    # --- Step 2: SA num_seeds times ------------------------------------
    R="$CSR_DIR/row_ptr_${N}.csv"
    Ci="$CSR_DIR/col_idx_${N}.csv"
    V="$CSR_DIR/J_values_${N}.csv"
    H="$CSR_DIR/h_vector_${N}.csv"

    BEST_E=""
    BEST_IDX=""
    BEST_SEED=""
    declare -a ALL_E=()
    declare -a ALL_T=()

    for ((i=1; i<=NUM_SEEDS; i++)); do
        SEED=$(python3 -c "import os; print(int.from_bytes(os.urandom(8), 'little'))")
        echo "  [2.$i/$NUM_SEEDS] SA seed=$SEED"

        SA_LOG=$("$SA_BIN" -R "$R" -C "$Ci" -V "$V" -l "$H" \
                  -x "$START_TEMP" -y "$STOP_TEMP" -c "$ALPHA" -m "$SWEEPS" \
                  --auto-accept-rate "$ACCEPT_RATE" \
                  -s "$SEED" -O "$RESULTS_DIR/" 2>&1)
        rc=$?
        if [[ $rc -ne 0 ]]; then
            echo "    SA failed (rc=$rc); skipping seed"
            continue
        fi

        E=$(echo "$SA_LOG" | grep -E "best-state[[:space:]]+energy[[:space:]]+\(recomputed\)" \
                          | awk -F': ' '{print $2}' | tail -1)
        T=$(echo "$SA_LOG" | grep -E "^[[:space:]]*Annealing:" | awk '{print $2}' | tail -1)
        [[ -z "$E" ]] && E="?"
        [[ -z "$T" ]] && T="?"

        # Snapshot the spin state for this seed (SA writes to spins_<N>; copy aside).
        SPIN_SRC="$RESULTS_DIR/spins_${N}"
        SPIN_DST="$MULTISEED_DIR/spins_${N}_seed${i}"
        [[ -f "$SPIN_SRC" ]] && cp "$SPIN_SRC" "$SPIN_DST"

        echo "    energy=$E  t_anneal=${T}s"
        echo "$BITS,$P,$Q,$N,$i,$SEED,$E,$T" >> "$OUT_PERRUN"

        ALL_E+=("$E")
        ALL_T+=("$T")

        if [[ "$E" != "?" ]]; then
            if [[ -z "$BEST_E" ]] \
               || [[ "$(awk -v a="$E" -v b="$BEST_E" 'BEGIN{print (a<b)?1:0}')" == "1" ]]; then
                BEST_E="$E"; BEST_IDX="$i"; BEST_SEED="$SEED"
            fi
        fi
    done

    # --- Step 3: aggregate + post-process winner -----------------------
    if [[ -z "$BEST_E" ]]; then
        echo "  All SA runs failed."
        echo "$BITS,$P,$Q,$N,$NUM_SEEDS,$T_CONSTRUCT,?,?,?,?,ERROR,ERROR,FAIL" >> "$OUT_SUMMARY"
        continue
    fi

    STATS=$(printf "%s\n" "${ALL_E[@]}" | grep -v '^?$' | awk '
        NR==1 {mn=$1; mx=$1}
        {if($1<mn) mn=$1; if($1>mx) mx=$1; s+=$1; n++}
        END {printf "%.10g %.10g %.10g", mn, s/n, mx}')
    read -r E_MIN E_MEAN E_MAX <<< "$STATS"
    T_MEAN=$(printf "%s\n" "${ALL_T[@]}" | grep -v '^?$' \
             | awk '{s+=$1; n++} END {if(n>0) printf "%.6f", s/n; else print "?"}')

    # Persist winner: copy spin state and write energy.
    BEST_SPIN_DST="$MULTISEED_DIR/spins_${N}_best"
    cp "$MULTISEED_DIR/spins_${N}_seed${BEST_IDX}" "$BEST_SPIN_DST"
    echo "$BEST_E" > "$MULTISEED_DIR/best_energy_${N}.txt"

    echo "  Best of $NUM_SEEDS: idx=$BEST_IDX seed=$BEST_SEED energy=$BEST_E"

    # Post-process the winning spin state.
    echo "  [3] post-processing winner ..."
    POST_LOG=$("$QUBO_BIN" "$N" "$BEST_SPIN_DST" --csr-dir "$CSR_DIR/" --meta-dir "$META_DIR/" 2>&1)
    PRED_LINE=$(echo "$POST_LOG" | grep "^Factors of " | tail -1)
    P_PRED=$(echo "$PRED_LINE" | sed -n 's/.*P = \([0-9]*\), Q = .*/\1/p')
    Q_PRED=$(echo "$PRED_LINE" | sed -n 's/.*Q = \([0-9]*\).*/\1/p')
    [[ -z "$P_PRED" ]] && P_PRED="?"
    [[ -z "$Q_PRED" ]] && Q_PRED="?"
    if echo "$POST_LOG" | grep -q "(CORRECT)"; then STATUS="CORRECT"; else STATUS="INCORRECT"; fi

    echo "  P_pred=$P_PRED  Q_pred=$Q_PRED  e_min=$E_MIN e_mean=$E_MEAN e_max=$E_MAX  [$STATUS]"
    echo "$BITS,$P,$Q,$N,$NUM_SEEDS,$T_CONSTRUCT,$E_MIN,$E_MEAN,$E_MAX,$T_MEAN,$P_PRED,$Q_PRED,$STATUS" >> "$OUT_SUMMARY"

    # Clean per-seed spins if requested.
    if [[ "$KEEP_PER_SEED_SPINS" != "true" ]]; then
        for ((i=1; i<=NUM_SEEDS; i++)); do
            rm -f "$MULTISEED_DIR/spins_${N}_seed${i}"
        done
    fi
done

echo
echo "=== Summary ==="
cat "$OUT_SUMMARY"
echo
echo "Saved: $OUT_SUMMARY"
echo "       $OUT_PERRUN"
echo "       (winner spins/energy in $MULTISEED_DIR/)"
