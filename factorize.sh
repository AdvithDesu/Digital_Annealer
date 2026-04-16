#!/bin/bash
# End-to-end integer factorization pipeline:
#   1. QUBO construction  (N -> Ising CSR + metadata)
#   2. Simulated annealing (Ising CSR -> spins)
#   3. Post-processing     (spins -> factors P, Q)
#
# Usage:
#   ./factorize.sh <N> [options]
#   ./factorize.sh -p <P> -q <Q> [options]
#
# Required (one of):
#   <N>                    integer to factorize (product of two primes)
#   -p <P> -q <Q>         two primes; N is computed as P*Q (supports 128-bit)
#
# SA options:
#   -x, --start-temp <T>   starting temperature       (default: 100.0)
#   -y, --stop-temp  <T>   stopping temperature       (default: 0.1)
#   -c, --alpha      <a>   geometric cooling rate     (default: 0.95)
#   -m, --sweeps     <M>   sweeps per beta            (default: 10)
#   -s, --seed       <S>   RNG seed                   (default: auto)
#   -d, --debug            enable debug output
#
# QUBO options:
#   -b, --backtrack        enable replacement backtracking (smaller QUBO)

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
BACKTRACK_FLAG=""
INPUT_P=""
INPUT_Q=""

# ── Parse arguments ──────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <N> [options]" >&2
    echo "   or: $0 -p <P> -q <Q> [options]" >&2
    echo "Run '$0 --help' for details." >&2
    exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '2,24p' "$0"
    exit 0
fi

# First positional arg is N, unless it starts with '-'
N=""
if [[ "$1" != -* ]]; then
    N="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -x|--start-temp)   START_TEMP="$2";   shift 2 ;;
        -y|--stop-temp)    STOP_TEMP="$2";    shift 2 ;;
        -c|--alpha)        ALPHA="$2";        shift 2 ;;
        -m|--sweeps)       SWEEPS="$2";       shift 2 ;;
        -s|--seed)         SEED="$2";         shift 2 ;;
        -d|--debug)        DEBUG_FLAG="-d";   shift ;;
        -b|--backtrack)    BACKTRACK_FLAG="--backtrack"; shift ;;
        -p)                INPUT_P="$2";      shift 2 ;;
        -q)                INPUT_Q="$2";      shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Compute N from P and Q if provided ───────────────────────
if [[ -n "$INPUT_P" && -n "$INPUT_Q" ]]; then
    N=$(python3 -c "print($INPUT_P * $INPUT_Q)")
    echo "P=$INPUT_P, Q=$INPUT_Q => N=$N"
elif [[ -n "$INPUT_P" || -n "$INPUT_Q" ]]; then
    echo "ERROR: both -p and -q must be specified together" >&2
    exit 1
fi

if [[ -z "$N" ]]; then
    echo "ERROR: no N specified. Provide <N> or use -p <P> -q <Q>" >&2
    exit 1
fi

# ── Build binaries if needed ─────────────────────────────────
QUBO_SRC="QUBO_Construction/QUBO_Integer_Factorization.cpp"
if [[ ! -x "$QUBO_BIN" || "$QUBO_SRC" -nt "$QUBO_BIN" ]]; then
    echo "Building QUBO construction binary..."
    g++ -O2 -std=c++17 -o "$QUBO_BIN" "$QUBO_SRC"
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

"$QUBO_BIN" "$N" --csr-dir "$CSR_DIR/" --meta-dir "$META_DIR/" $BACKTRACK_FLAG

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

"$QUBO_BIN" "$N" "$SPINS_FILE" --csr-dir "$CSR_DIR/" --meta-dir "$META_DIR/" $BACKTRACK_FLAG

echo
echo "===== Pipeline complete ====="
