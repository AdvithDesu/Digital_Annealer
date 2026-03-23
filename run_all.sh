#!/bin/bash
# Run SA for all numbers in bin_SI and collect timing into a single file.

SCRIPT="Non_CUDA_GPU_Simulated_Annealing.py"
DIR="bin_SI"
OUT="timing_results.txt"

# SA parameters
X=100.0   # start temp
Y=0.1     # final temp
C=0.95    # cooling rate
M=10      # sweeps per beta

# Header
printf "%-25s %12s %12s %12s %12s %12s %12s\n" \
       "N" "num_spins" "nnz" "t_load" "t_setup" "t_anneal" "t_total" > "$OUT"

# Extract unique N values from J_values_*.csv filenames
for jfile in "$DIR"/J_values_*.csv; do
    N=$(basename "$jfile" .csv | sed 's/J_values_//')

    # Check all four files exist
    R="$DIR/row_ptr_${N}.csv"
    Ci="$DIR/col_idx_${N}.csv"
    V="$DIR/J_values_${N}.csv"
    H="$DIR/h_vector_${N}.csv"

    if [[ ! -f "$R" || ! -f "$Ci" || ! -f "$V" || ! -f "$H" ]]; then
        echo "Skipping N=$N (missing files)"
        continue
    fi

    echo "===== Running N=$N ====="
    output=$(python3 "$SCRIPT" -R "$R" -C "$Ci" -V "$V" -l "$H" \
             -x "$X" -y "$Y" -c "$C" -m "$M" --device auto 2>&1)

    # Parse timing lines
    t_load=$(echo "$output"   | grep "Load data:"   | awk '{print $3}')
    t_setup=$(echo "$output"  | grep "Setup (GPU):"  | awk '{print $3}')
    t_anneal=$(echo "$output" | grep "Annealing:"    | awk '{print $2}')
    t_total=$(echo "$output"  | grep "Total:"        | awk '{print $2}')

    # Parse num_spins and nnz
    num_spins=$(echo "$output" | grep "num_spins" | sed 's/.*num_spins = \([0-9]*\).*/\1/')
    nnz=$(echo "$output"       | grep "nnz"       | sed 's/.*nnz = \([0-9]*\).*/\1/')

    printf "%-25s %12s %12s %12s %12s %12s %12s\n" \
           "$N" "$num_spins" "$nnz" "$t_load" "$t_setup" "$t_anneal" "$t_total" >> "$OUT"

    echo "  Done: t_anneal=${t_anneal}s  t_total=${t_total}s"
done

echo ""
echo "=== All results ==="
cat "$OUT"
echo ""
echo "Results saved to $OUT"
