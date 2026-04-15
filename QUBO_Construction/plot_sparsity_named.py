"""Plot J sparsity with variables reordered by name (lex sort).
Usage: python plot_sparsity_named.py <csr_dir> <label> [<index_to_var_file>]
If index_to_var_file omitted, no reorder is done (plot in CSR-native order).
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

d = sys.argv[1]
label = sys.argv[2]
idx_file = sys.argv[3] if len(sys.argv) > 3 else None
N = "159197"

row_ptr = np.loadtxt(f"{d}/row_ptr_{N}.csv", dtype=int, delimiter=",")
col_idx = np.loadtxt(f"{d}/col_idx_{N}.csv", dtype=int, delimiter=",")
J_vals  = np.loadtxt(f"{d}/J_values_{N}.csv", dtype=float, delimiter=",")

n = len(row_ptr) - 1
M = np.zeros((n, n))
for i in range(n):
    for k in range(row_ptr[i], row_ptr[i+1]):
        j = col_idx[k]
        M[i, j] = J_vals[k]
        M[j, i] = J_vals[k]

# Reorder by name if mapping available
if idx_file and os.path.exists(idx_file):
    names = [None] * n
    with open(idx_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                i = int(parts[0])
                if 0 <= i < n:
                    names[i] = parts[1]
    # Sort indices by name
    order = sorted(range(n), key=lambda i: names[i] if names[i] else "")
    M = M[np.ix_(order, order)]
    print(f"Reordered {n} vars by name. First 5: {[names[order[i]] for i in range(5)]}")
else:
    print(f"No reorder (no index file given)")

print(f"n={n}, nnz(upper)={len(J_vals)}, |J|.max={np.abs(M).max():.1f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].spy(M, markersize=1)
ax[0].set_title(f"{label} J sparsity (name-sorted) N={N}, n={n}")
vmax = np.max(np.abs(M))
im = ax[1].imshow(M, cmap="seismic", vmin=-vmax, vmax=vmax)
ax[1].set_title(f"{label} J magnitudes N={N}")
plt.colorbar(im, ax=ax[1])
plt.tight_layout()
out = f"{d}/sparsity_named_{label}_{N}.png"
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
