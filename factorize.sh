#!/bin/bash
# End-to-end integer factorization pipeline:
#   1. QUBO construction  (N -> Ising CSR + metadata)
#   2. Simulated annealing (Ising CSR -> spins)
#   3. Post-processing     (spins -> factors P, Q)
#
# Usage:
#   ./factorize.sh <N> [options]
#
# Required:
#   <N>                    integer to factorize (product of two primes)
#
# SA options:
#   -x, --start-temp <T>   starting temperature       (default: 100.0)
#   -y, --stop-temp  <T>   stopping temperature       (default: 0.1)
#   -c, --alpha      <a>   geometric cooling rate     (default: 0.95)
#   -m, --sweeps     <M>   sweeps per beta            (default: 10)
#   -s, --seed       <S>   RNG seed                   (default: auto)
#   -d, --debug            enable debug output

set -euo pipefail

# ── Binaries ──────────────────────────────────────────────────
QUBO_BIN="./QUBO_Construction/qubo_factorization"
SA_BIN="./build/annealer_gpu_SI/annealer_gpu_SI"

# ── Directories ───────────────────────────────────────────────
CSR_DIR="bin_SI"
META_DIR="qubo_metadata"
RESULTS_DIR="results"

# ── SA defaults ───────────────────────────────────────────────
START_TEMP=100.0
STOP_TEMP=0.1
ALPHA=0.95
SWEEPS=10
SEED=""
DEBUG_FLAG=""

# ── Parse arguments ──────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <N> [options]" >&2
    echo "Run '$0 --help' for details." >&2
    exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '2,18p' "$0"
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
        -d|--debug)        DEBUG_FLAG="-d";   shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Build binaries if needed ─────────────────────────────────
if [[ ! -x "$QUBO_BIN" ]]; then
    echo "Building QUBO construction binary..."
    g++ -O2 -std=c++17 -o "$QUBO_BIN" QUBO_Construction/QUBO_Integer_Factorization.cpp
    echo "Built: $QUBO_BIN"
fi

if [[ ! -x "$SA_BIN" ]]; then
    echo "Building SA binary (cmake + make)..."
    mkdir -p build
    cmake -S . -B build
    make -C build -j"$(nproc)"
    if [[ ! -x "$SA_BIN" ]]; then
        echo "ERROR: SA build failed, binary not found: $SA_BIN" >&2
        exit 1
    fi
    echo "Built: $SA_BIN"
fi

# ── Create output directories ────────────────────────────────
mkdir -p "$CSR_DIR"
mkdir -p "$META_DIR"
mkdir -p "$RESULTS_DIR"

# ==============================================================
# Step 1: QUBO Construction
# ==============================================================
echo "===== Step 1: QUBO Construction (N=$N) ====="
echo

"$QUBO_BIN" "$N" --csr-dir "$CSR_DIR/" --meta-dir "$META_DIR/"

# Verify CSR files were created
for f in "$CSR_DIR/row_ptr_${N}.csv" \
         "$CSR_DIR/col_idx_${N}.csv" \
         "$CSR_DIR/J_values_${N}.csv" \
         "$CSR_DIR/h_vector_${N}.csv"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: expected QUBO output not found: $f" >&2
        exit 1
    fi
done

echo
echo "CSR files written to $CSR_DIR/"
echo "Metadata written to $META_DIR/"
echo

# ==============================================================
# Step 2: Simulated Annealing
# ==============================================================
echo "===== Step 2: Simulated Annealing ====="
echo

R="$CSR_DIR/row_ptr_${N}.csv"
C="$CSR_DIR/col_idx_${N}.csv"
V="$CSR_DIR/J_values_${N}.csv"
H="$CSR_DIR/h_vector_${N}.csv"

SA_CMD=("$SA_BIN" -R "$R" -C "$C" -V "$V" -l "$H"
        -x "$START_TEMP" -y "$STOP_TEMP" -c "$ALPHA" -m "$SWEEPS"
        -O "$RESULTS_DIR/")
[[ -n "$SEED" ]]       && SA_CMD+=(-s "$SEED")
[[ -n "$DEBUG_FLAG" ]] && SA_CMD+=("$DEBUG_FLAG")

echo "Command: ${SA_CMD[*]}"
echo

"${SA_CMD[@]}"

SPINS_FILE="$RESULTS_DIR/spins_${N}"
if [[ ! -f "$SPINS_FILE" ]]; then
    echo "ERROR: spins file not found: $SPINS_FILE" >&2
    exit 1
fi

echo
echo "Spins written to $SPINS_FILE"
echo "Energy history written to $RESULTS_DIR/energy_history_${N}"
echo

# ==============================================================
# Step 3: Post-processing (factor extraction)
# ==============================================================
echo "===== Step 3: Post-processing ====="
echo

"$QUBO_BIN" "$N" "$SPINS_FILE" --csr-dir "$CSR_DIR/" --meta-dir "$META_DIR/"

echo
echo "===== Pipeline complete ====="
