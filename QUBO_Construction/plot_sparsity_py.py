import numpy as np
import matplotlib.pyplot as plt
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
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

print(f"n={n}, nnz(upper)={len(J_vals)}")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].spy(M, markersize=1)
ax[0].set_title(f"Python J sparsity N={N}, n={n}")
im = ax[1].imshow(M, cmap="seismic", vmin=-np.max(np.abs(M)), vmax=np.max(np.abs(M)))
ax[1].set_title(f"Python J magnitudes N={N}")
plt.colorbar(im, ax=ax[1])
plt.tight_layout()
out = f"{d}/sparsity_py_{N}.png"
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
