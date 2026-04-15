import numpy as np

def load(d):
    rp = np.loadtxt(f"{d}/row_ptr_159197.csv", dtype=int, delimiter=",")
    ci = np.loadtxt(f"{d}/col_idx_159197.csv", dtype=int, delimiter=",")
    jv = np.loadtxt(f"{d}/J_values_159197.csv", dtype=float, delimiter=",")
    h  = np.loadtxt(f"{d}/h_vector_159197.csv", dtype=float, delimiter=",")
    return rp, ci, jv, h

def stats(name, rp, ci, jv, h):
    print(f"--- {name} ---")
    print(f" n = {len(rp)-1}, nnz = {len(jv)}")
    print(f" J: min={jv.min():.3f}, max={jv.max():.3f}, mean={jv.mean():.3f}, std={jv.std():.3f}")
    print(f" |J|: sum={np.abs(jv).sum():.1f}, max={np.abs(jv).max():.3f}")
    print(f" h: min={h.min():.3f}, max={h.max():.3f}, mean={h.mean():.3f}, std={h.std():.3f}")
    print(f" |h|: sum={np.abs(h).sum():.1f}, max={np.abs(h).max():.3f}")
    # Bin J values by magnitude
    bins = [0, 1, 5, 25, 100, 500, 2500, 1e9]
    counts, _ = np.histogram(np.abs(jv), bins=bins)
    print(f" |J| histogram: {dict(zip([f'<{b}' for b in bins[1:]], counts))}")

rp1, ci1, jv1, h1 = load("d:/Projects/Digital_Annealer/QUBO_Construction")
rp2, ci2, jv2, h2 = load("C:/Users/advi/AppData/Local/Temp/bt_out")

stats("Python", rp1, ci1, jv1, h1)
stats("C++   ", rp2, ci2, jv2, h2)

# Multiset comparison: is the SET of J values the same (just permuted)?
sj1 = np.sort(np.abs(jv1))
sj2 = np.sort(np.abs(jv2))
print(f"\nSorted |J| identical? {np.allclose(sj1, sj2)}")
if not np.allclose(sj1, sj2):
    diff = sj1 - sj2
    print(f"  max abs diff in sorted |J|: {np.abs(diff).max():.4f}")
    print(f"  count of |J| values differing > 1e-3: {(np.abs(diff) > 1e-3).sum()}")
sh1 = np.sort(np.abs(h1))
sh2 = np.sort(np.abs(h2))
print(f"Sorted |h| identical? {np.allclose(sh1, sh2)}")
