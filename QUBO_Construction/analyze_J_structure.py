"""For each J entry, identify which variables it connects, and see if
|J| correlates with variable 'importance' (bit position for p_i / q_i).

Usage: python analyze_J_structure.py <N> [csr_dir] [meta_dir]
"""
import numpy as np
import matplotlib.pyplot as plt
import re, sys

N = sys.argv[1]
d = sys.argv[2] if len(sys.argv) > 2 else "bin_SI"
md = sys.argv[3] if len(sys.argv) > 3 else "qubo_metadata"

row_ptr = np.loadtxt(f"{d}/row_ptr_{N}.csv", dtype=int, delimiter=",")
col_idx = np.loadtxt(f"{d}/col_idx_{N}.csv", dtype=int, delimiter=",")
J       = np.loadtxt(f"{d}/J_values_{N}.csv", dtype=float, delimiter=",")
h       = np.loadtxt(f"{d}/h_vector_{N}.csv", dtype=float, delimiter=",")

# Load var names
names = [None] * (len(row_ptr) - 1)
with open(f"{md}/index_to_var_{N}.txt") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            names[int(parts[0])] = parts[1]

def classify(nm):
    """Return ('p', bit) / ('q', bit) / ('s', col_from, col_to) / ('aux', idx)."""
    if nm is None: return ("?",)
    m = re.match(r"^p_(\d+)$", nm)
    if m: return ("p", int(m.group(1)))
    m = re.match(r"^q_(\d+)$", nm)
    if m: return ("q", int(m.group(1)))
    m = re.match(r"^s_(\d+)_(\d+)$", nm)
    if m: return ("s", int(m.group(1)), int(m.group(2)))
    return ("aux", nm)

# Build (i, j, |J|) tuples
n = len(row_ptr) - 1
ii, jj, vv = [], [], []
for i in range(n):
    for k in range(row_ptr[i], row_ptr[i+1]):
        ii.append(i); jj.append(col_idx[k]); vv.append(abs(J[k]))
ii = np.array(ii); jj = np.array(jj); vv = np.array(vv)

# Correlate |J| with variable 'position'
# For p_k or q_k, use k. For s_i_j, use max(i,j). For aux, use -1 (unknown).
def pos(nm):
    cl = classify(nm)
    if cl[0] == "p" or cl[0] == "q": return cl[1]
    if cl[0] == "s": return max(cl[1], cl[2])
    return -1

pos_i = np.array([pos(names[x]) for x in ii])
pos_j = np.array([pos(names[x]) for x in jj])
max_pos = np.maximum(pos_i, pos_j)
valid = max_pos >= 0

print(f"N = {N}")
print(f"nnz = {len(vv)},  valid (pos known on both) = {valid.sum()}")

# Scatter: max_pos vs log10|J|
logJ = np.log10(vv[valid] + 1e-30)
pos_plot = max_pos[valid]

# Binned stats: for each position, median log|J|
unique_pos = np.unique(pos_plot)
medians = []
p90 = []
p10 = []
for p in unique_pos:
    mask = pos_plot == p
    medians.append(np.median(logJ[mask]))
    p90.append(np.percentile(logJ[mask], 90))
    p10.append(np.percentile(logJ[mask], 10))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
# Hexbin (dense scatter)
ax[0].hexbin(pos_plot, logJ, gridsize=40, cmap="viridis", mincnt=1)
ax[0].set_xlabel("max(bit position) of connected vars")
ax[0].set_ylabel("log10(|J|)")
ax[0].set_title(f"|J| vs bit position (N={N})")

# Median per position
ax[1].plot(unique_pos, medians, 'o-', label="median")
ax[1].fill_between(unique_pos, p10, p90, alpha=0.3, label="10-90%")
ax[1].set_xlabel("max(bit position)")
ax[1].set_ylabel("log10(|J|)")
ax[1].set_title("|J| magnitude by variable bit position")
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
out = f"{d}/J_by_position_{N}.png"
plt.savefig(out, dpi=120)
print(f"Plot saved: {out}")
