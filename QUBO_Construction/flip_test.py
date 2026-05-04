"""Stress test: flip a small subset of bits in the ideal spins and run BOTH
post-processes. Are the outputs identical, or do they diverge?"""
import sys
import subprocess
import random
import re


def flip_spins(in_path, out_path, num_flips, seed, indices=None):
    rng = random.Random(seed)
    spins = open(in_path).read().split()
    if indices is None:
        indices = list(range(len(spins)))
    if num_flips > len(indices):
        num_flips = len(indices)
    flip_idx = rng.sample(indices, num_flips)
    for i in flip_idx:
        spins[i] = "-1" if spins[i] == "1" else "1"
    with open(out_path, "w") as f:
        f.write(" ".join(spins) + "\n")
    return flip_idx


def get_pq_indices(N):
    """Read index_to_var_<N>.txt; return indices of p_/q_ vars only."""
    pq = []
    with open(f"index_to_var_{N}.txt") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            idx = int(parts[0])
            name = parts[1]
            if name.startswith("p_") or name.startswith("q_"):
                pq.append(idx)
    return pq


def run_cpp(N, spins_path):
    out = subprocess.run(
        ["./qubo_factorization_diag.exe", str(N), spins_path],
        capture_output=True, text=True
    ).stdout
    m = re.search(r"Factors of \d+: P = (\d+), Q = (\d+)", out)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def run_py(N, spins_path):
    out = subprocess.run(
        ["python", "post_process_py.py", str(N), spins_path],
        capture_output=True, text=True
    ).stdout
    m = re.search(r"P = (\d+), Q = (\d+)", out)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 51959
    base_spins = sys.argv[2] if len(sys.argv) > 2 else f"spins_{N}_ideal"

    pq_indices = get_pq_indices(N)
    print(f"=== Flip-test for N={N}: comparing C++ vs Python post-process ===")
    print(f"Flipping ONLY p/q bits ({len(pq_indices)} of them) in spins\n")
    print(f"{'#flips':>7s}  {'seed':>5s}  {'C++ (P, Q)':>22s}  {'Py  (P, Q)':>22s}  match")
    print("-" * 78)
    matches = 0
    total = 0
    # ~20 trials: 5 flip levels * 4 seeds
    for nflips in [0, 1, 2, 4, 8]:
        for seed in range(4):
            test_spins = f"spins_test_{N}_{nflips}_{seed}"
            flip_spins(base_spins, test_spins, nflips, seed, indices=pq_indices)
            cpp = run_cpp(N, test_spins)
            py = run_py(N, test_spins)
            ok = cpp == py
            matches += ok
            total += 1
            print(f"{nflips:>7d}  {seed:>5d}  {str(cpp):>22s}  {str(py):>22s}  {'OK' if ok else '*** DIFFER'}")
    print(f"\n{matches}/{total} match")


if __name__ == "__main__":
    main()
