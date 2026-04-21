"""Analyze the distribution of |J| values for a given N.
Reports histogram on log scale, percentiles, and how many entries fall into
each order-of-magnitude bucket. Also plots the distribution.

Usage: python analyze_J.py <N> [csr_dir]
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

N = sys.argv[1]
d = sys.argv[2] if len(sys.argv) > 2 else "bin_SI"

J = np.loadtxt(f"{d}/J_values_{N}.csv", dtype=float, delimiter=",")
h = np.loadtxt(f"{d}/h_vector_{N}.csv", dtype=float, delimiter=",")
row_ptr = np.loadtxt(f"{d}/row_ptr_{N}.csv", dtype=int, delimiter=",")

n = len(row_ptr) - 1
nnz = len(J)

absJ = np.abs(J)
absh = np.abs(h)
nz_J = absJ[absJ > 0]
nz_h = absh[absh > 0]

print(f"=== N = {N} ===")
print(f"n_vars = {n},  nnz(J) = {nnz}")
print()

def summarize(name, arr):
    if len(arr) == 0:
        print(f"{name}: all zero"); return
    print(f"{name}:  count={len(arr)}")
    print(f"  min     = {arr.min():.3e}")
    print(f"  max     = {arr.max():.3e}")
    print(f"  median  = {np.median(arr):.3e}")
    print(f"  dynamic range (max/min) = {arr.max()/arr.min():.3e}")
    # percentiles
    pct = [1, 10, 25, 50, 75, 90, 99, 99.9]
    vals = np.percentile(arr, pct)
    print("  percentiles:")
    for p, v in zip(pct, vals):
        print(f"    {p:>5.1f}% = {v:.3e}")

summarize("|J| (nonzero)", nz_J)
print()
summarize("|h| (nonzero)", nz_h)
print()

# Order-of-magnitude histogram
print("=== |J| by order of magnitude ===")
log_bins = np.arange(-5, 45)
hist, _ = np.histogram(np.log10(nz_J), bins=log_bins)
for i, c in enumerate(hist):
    if c > 0:
        lo, hi = 10.0**log_bins[i], 10.0**log_bins[i+1]
        pct = 100 * c / len(nz_J)
        bar = "#" * min(60, int(pct * 2))
        print(f"  [1e{log_bins[i]:+3d}, 1e{log_bins[i+1]:+3d})  {c:>10d}  {pct:5.1f}%  {bar}")

# Save plot
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].hist(np.log10(nz_J), bins=60)
ax[0].set_xlabel("log10(|J|)")
ax[0].set_ylabel("count")
ax[0].set_title(f"|J| distribution (N={N}, n={n}, nnz={nnz})")
ax[0].set_yscale("log")

ax[1].hist(np.log10(nz_h), bins=60, color="orange")
ax[1].set_xlabel("log10(|h|)")
ax[1].set_ylabel("count")
ax[1].set_title(f"|h| distribution")
ax[1].set_yscale("log")

plt.tight_layout()
out = f"{d}/J_distribution_{N}.png"
plt.savefig(out, dpi=120)
print(f"\nPlot saved: {out}")
