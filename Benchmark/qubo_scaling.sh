#!/bin/bash
# Benchmark QUBO construction only (no SA, no post-processing).
# For each bit size B, generates two random B-bit primes, builds the QUBO,
# then computes degree stats from row_ptr.csv.
#
# CSV columns:
#   bits, P, Q, N, num_spins, nnz, num_dense, num_sparse, frac_dense, t_construct_s
#
# Usage: ./qubo_scaling.sh [options]
#
# Options:
#   -B <list>  comma-separated bit sizes  (default: 8,10,12,...,62)
#   -o <file>  output CSV                 (default: qubo_scaling.csv)
#   -h         show this help

set -u

# ── Always run from repo root so existing relative paths keep working ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ─────────────────────────────────────────────────
BITS_LIST=()
for b in $(seq 8 2 62); do BITS_LIST+=("$b"); done
OUT="qubo_scaling.csv"

# ── Parse CLI ────────────────────────────────────────────────
while getopts ":B:o:h" opt; do
    case "$opt" in
        B) IFS=',' read -ra BITS_LIST <<< "$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        h) sed -n '2,16p' "$0"; exit 0 ;;
        \?) echo "Unknown option: -$OPTARG" >&2; exit 1 ;;
        :)  echo "Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

# ── Paths ────────────────────────────────────────────────────
QUBO_BIN="./QUBO_Construction/qubo_factorization"
QUBO_SRC="QUBO_Construction/QUBO_Integer_Factorization.cpp"
CSR_DIR="bin_SI"
META_DIR="qubo_metadata"
MAX_HUB_SPINS=256     # must match #define MAX_HUB_SPINS in optimized_main_single_flip.cu
# DENSE_THRESHOLD is computed adaptively per problem (see Python block below)

# ── Build QUBO binary if needed ──────────────────────────────
if [[ ! -x "$QUBO_BIN" || "$QUBO_SRC" -nt "$QUBO_BIN" ]]; then
    echo "Building QUBO binary..."
    g++ -O2 -std=c++17 -o "$QUBO_BIN" "$QUBO_SRC"
fi

mkdir -p "$CSR_DIR" "$META_DIR"

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

# ── CSV header ───────────────────────────────────────────────
echo "bits,P,Q,N,num_spins,nnz,mean_degree,max_degree,adaptive_thresh,num_dense,num_sparse,frac_dense,t_construct_s" > "$OUT"

# ── Main loop ────────────────────────────────────────────────
for BITS in "${BITS_LIST[@]}"; do
    P=$(gen_prime "$BITS")
    Q=$(gen_prime "$BITS")
    while [[ "$P" == "$Q" ]]; do Q=$(gen_prime "$BITS"); done
    N=$(python3 -c "print($P * $Q)")

    echo "===== bits=$BITS  P=$P  Q=$Q  N=$N ====="

    LOG=$("$QUBO_BIN" "$N" --csr-dir "$CSR_DIR/" --meta-dir "$META_DIR/" 2>&1)
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  QUBO build failed:"
        echo "$LOG" | tail -5
        echo "$BITS,$P,$Q,$N,?,?,?,?,?,?" >> "$OUT"
        continue
    fi

    T_CONSTRUCT=$(echo "$LOG" | grep -E "^Total construction time:" | awk '{print $4}' | head -1)
    [[ -z "$T_CONSTRUCT" ]] && T_CONSTRUCT="?"

    ROW_PTR_FILE="$CSR_DIR/row_ptr_${N}.csv"
    if [[ ! -f "$ROW_PTR_FILE" ]]; then
        echo "  row_ptr file not found: $ROW_PTR_FILE"
        echo "$BITS,$P,$Q,$N,?,?,?,?,?,?,?,?,?" >> "$OUT"
        continue
    fi

    # Compute degree stats from row_ptr without loading the full J matrix.
    # row_ptr has (num_spins+1) entries; degree[i] = row_ptr[i+1] - row_ptr[i].
    #
    # Adaptive threshold = sqrt(max_degree * mean_degree), bounded so that
    # num_dense <= MAX_HUB_SPINS.  This separates the hub cluster (near-full
    # rows in the sparsity plot) from the low-degree bulk at every scale:
    #   - small N (n~160):  mean~10, max~160  -> threshold~40
    #   - large N (n~4M):   mean~3.5, max~500 -> threshold~42 (vs hardcoded 128)
    # The MAX_HUB_SPINS cap prevents SA kernel overflow for edge cases where
    # the geometric-mean threshold still admits too many hubs.
    read -r NUM_SPINS NNZ MEAN_DEG MAX_DEG ADAPTIVE_THRESH NUM_DENSE NUM_SPARSE FRAC_DENSE < <(python3 - <<PYEOF
import math

import re
with open("$ROW_PTR_FILE") as f:
    rp = list(map(int, re.split(r'[,\s]+', f.read().strip())))

n   = len(rp) - 1
nnz = rp[-1]
degrees = [rp[i+1] - rp[i] for i in range(n)]

mean_deg = nnz / n if n > 0 else 0
max_deg  = max(degrees) if degrees else 0

# Geometric mean of max and mean degree — sits in the valley of the
# bimodal hub vs bulk degree distribution.
thresh = max(16, int(math.sqrt(max_deg * mean_deg)))

# Safety cap: lower threshold until num_dense <= MAX_HUB_SPINS.
max_hub = $MAX_HUB_SPINS
while thresh > 16:
    if sum(1 for d in degrees if d >= thresh) <= max_hub:
        break
    thresh += 1   # shouldn't happen normally; just a guard

dense  = sum(1 for d in degrees if d >= thresh)
sparse = n - dense
frac   = dense / n if n > 0 else 0.0

print(n, nnz, f"{mean_deg:.2f}", max_deg, thresh, dense, sparse, f"{frac:.6f}")
PYEOF
)

    echo "  num_spins=$NUM_SPINS  nnz=$NNZ  mean_deg=$MEAN_DEG  max_deg=$MAX_DEG  thresh=$ADAPTIVE_THRESH  dense=$NUM_DENSE  sparse=$NUM_SPARSE  frac=$FRAC_DENSE  t=${T_CONSTRUCT}s"
    echo "$BITS,$P,$Q,$N,$NUM_SPINS,$NNZ,$MEAN_DEG,$MAX_DEG,$ADAPTIVE_THRESH,$NUM_DENSE,$NUM_SPARSE,$FRAC_DENSE,$T_CONSTRUCT" >> "$OUT"
done

echo ""
echo "=== Results ==="
cat "$OUT"
echo ""
echo "Saved to $OUT"
