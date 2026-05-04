"""Build an idealized spins file. Usage: make_ideal_spins.py <N> <P> <Q> <out>."""
import sys

N = int(sys.argv[1])
P = int(sys.argv[2])
Q = int(sys.argv[3])
out_path = sys.argv[4] if len(sys.argv) > 4 else f"spins_{N}_ideal"

# Read index_to_var_<N>.txt
idx_to_var = []
with open(f"index_to_var_{N}.txt") as f:
    for line in f:
        parts = line.strip().split()
        idx = int(parts[0])
        name = parts[1]
        idx_to_var.append((idx, name))

# Build spin per index
spins = [0] * len(idx_to_var)
for i, name in idx_to_var:
    if name.startswith("p_"):
        bit = int(name[2:])
        spins[i] = (P >> bit) & 1
    elif name.startswith("q_"):
        bit = int(name[2:])
        spins[i] = (Q >> bit) & 1
    else:
        # w_*: leave at 0 (irrelevant for p_1, p_2 chain in this case)
        spins[i] = 0

# Write spins file: format expected is +1/-1 separated by whitespace (per readSpinsFile)
with open(out_path, "w") as f:
    for s in spins:
        f.write(("1" if s else "-1") + " ")
    f.write("\n")
print(f"Wrote {len(spins)} spins to {out_path}")
print(f"  p/q bits set per (P={P}, Q={Q}); w-vars set to 0")
print(f"  Expected post-process output: P={P}, Q={Q}")
