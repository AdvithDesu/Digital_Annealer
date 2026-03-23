#!/usr/bin/env python3
"""
GPU-accelerated Simulated Annealing for Ising ground-state search.

Python port of optimized_main_single_flip.cu -- runs on Mac (MPS) or
CUDA GPUs. Uses PyTorch for GPU acceleration.

The J matrix is kept in CSR form throughout. The per-spin neighbour sum
(the expensive step that the CUDA code parallelises with dense/sparse
kernels, shared-memory reductions, and warp shuffles) is computed here
via a gather-multiply-scatter_add pattern on the flat CSR arrays:

    neighbor_spins  = spins[col_idx]                   # gather   O(nnz)
    products        = J_values * neighbor_spins         # multiply O(nnz)
    neighbor_sums.scatter_add_(0, row_idx, products)    # reduce   O(nnz)

All three ops are natively parallel on MPS and CUDA.

Usage:
    python sa_ising.py \
        -R bin_SI/row_ptr_39203.csv \
        -C bin_SI/col_idx_39203.csv \
        -V bin_SI/J_values_39203.csv \
        -l bin_SI/h_vector_39203.csv \
        -x 20 -y 0.001 -c 0.95 -d
"""

import torch
import numpy as np
import argparse
import math
import time
import os


# ---------------------------------------------------------------------------
# I/O helpers -- handle both newline-separated and comma-separated CSV files
# ---------------------------------------------------------------------------

def load_csv_int(path: str) -> np.ndarray:
    with open(path, "r") as f:
        text = f.read().strip()
    tokens = text.replace("\n", ",").split(",")
    return np.array([int(float(t)) for t in tokens if t.strip()], dtype=np.int64)


def load_csv_float(path: str) -> np.ndarray:
    with open(path, "r") as f:
        text = f.read().strip()
    tokens = text.replace("\n", ",").split(",")
    return np.array([float(t) for t in tokens if t.strip()], dtype=np.float32)


# ---------------------------------------------------------------------------
# CSR preprocessing -- expand row_ptr into per-nnz row indices (done once)
# ---------------------------------------------------------------------------

def expand_row_ptr(row_ptr: torch.Tensor, num_spins: int) -> torch.Tensor:
    """Convert CSR row_ptr to a flat row-index array (COO-style).

    row_ptr   = [0, 3, 5, 9, ...]   (length num_spins + 1)
    row_idx   = [0, 0, 0, 1, 1, 2, 2, 2, 2, ...]   (length nnz)

    Each non-zero k knows which spin (row) it belongs to.
    """
    counts = row_ptr[1:] - row_ptr[:-1]  # degree of each spin
    return torch.repeat_interleave(
        torch.arange(num_spins, device=row_ptr.device), counts
    )


# ---------------------------------------------------------------------------
# Sparse neighbour-sum (the core GPU-parallel operation)
# ---------------------------------------------------------------------------

def sparse_neighbor_sums(
    spins: torch.Tensor,
    col_idx: torch.Tensor,
    J_values: torch.Tensor,
    row_idx: torch.Tensor,
    num_spins: int,
) -> torch.Tensor:
    """Compute  neighbor_sums[i] = sum_j J_ij * s_j   for all i.

    Equivalent to the dense-kernel shared-mem reduction and the
    sparse-kernel warp-shuffle reduction in the CUDA code.
    All three steps are O(nnz) and GPU-parallel.
    """
    neighbor_spins = spins[col_idx]                         # gather
    products = J_values * neighbor_spins                    # multiply
    neighbor_sums = torch.zeros(num_spins, dtype=spins.dtype,
                                device=spins.device)
    neighbor_sums.scatter_add_(0, row_idx, products)        # segmented sum
    return neighbor_sums


# ---------------------------------------------------------------------------
# Energy computation (init + final verification only)
# ---------------------------------------------------------------------------

def compute_total_energy(
    spins: torch.Tensor,
    h: torch.Tensor,
    col_idx: torch.Tensor,
    J_values: torch.Tensor,
    row_idx: torch.Tensor,
    num_spins: int,
) -> float:
    """E = 0.5 * s^T J s  +  h^T s   (computed via CSR, no dense matrix)."""
    Js = sparse_neighbor_sums(spins, col_idx, J_values, row_idx, num_spins)
    return (0.5 * torch.dot(spins, Js) + torch.dot(h, spins)).item()


# ---------------------------------------------------------------------------
# Beta (inverse-temperature) schedule
# ---------------------------------------------------------------------------

def create_beta_schedule(temp_start: float, temp_end: float,
                         alpha: float) -> list[float]:
    num_steps = int(math.log(temp_end / temp_start) / math.log(alpha))
    betas: list[float] = []
    T = temp_start
    for _ in range(num_steps):
        betas.append(1.0 / T)
        T *= alpha
        if T < temp_end:
            T = temp_end
    return betas


# ---------------------------------------------------------------------------
# Main SA loop
# ---------------------------------------------------------------------------

def run_sa(
    spins: torch.Tensor,
    h: torch.Tensor,
    col_idx: torch.Tensor,
    J_values: torch.Tensor,
    row_idx: torch.Tensor,
    num_spins: int,
    beta_schedule: list[float],
    num_sweeps_per_beta: int = 1,
    disable_early_stop: bool = False,
    debug: bool = False,
) -> tuple[torch.Tensor, float, float, list[float]]:
    """
    Single-flip simulated annealing using CSR directly on GPU.

    Each sweep (all O(nnz) on GPU):
      1. gather-multiply-scatter_add  ->  neighbour sums for all spins
      2. dE_i = -2 (neighbour_sum_i + h_i) * s_i
      3. Metropolis filter -> candidate mask
      4. Pick one candidate uniformly at random, flip it
    """
    device = spins.device

    total_energy = compute_total_energy(
        spins, h, col_idx, J_values, row_idx, num_spins)
    best_energy = total_energy
    energy_history: list[float] = []

    print(f"Start annealing with initial energy: {total_energy:.6f}")
    t0 = time.perf_counter()

    for i, beta in enumerate(beta_schedule):
        no_update = 0

        for _ in range(num_sweeps_per_beta):
            # 1. Neighbour sums via CSR gather-multiply-scatter
            neighbor_sums = sparse_neighbor_sums(
                spins, col_idx, J_values, row_idx, num_spins)

            # 2. Delta energy for every possible single-spin flip
            dE = -2.0 * (neighbor_sums + h) * spins

            # 3. Metropolis acceptance
            acceptance = torch.clamp(torch.exp(-beta * dE), max=1.0)
            rand_vals = torch.rand(num_spins, device=device)
            candidates = torch.where(rand_vals < acceptance)[0]

            if candidates.numel() > 0:
                # 4. Pick one candidate uniformly at random
                chosen_local = torch.randint(
                    candidates.numel(), (1,), device=device)
                chosen_spin = candidates[chosen_local].item()
                chosen_dE = dE[chosen_spin].item()

                # 5. Flip & update running energy
                spins[chosen_spin] *= -1
                total_energy += chosen_dE

                # 6. Track best
                if total_energy < best_energy:
                    best_energy = total_energy
                    no_update = 0
                else:
                    no_update += 1

                # 7. Early stopping
                if (not disable_early_stop
                        and no_update > num_sweeps_per_beta):
                    if debug:
                        print(f"  Breaking early at temp iteration {i}")
                    break

        energy_history.append(best_energy)

        if debug and (i % 100 == 0 or i == len(beta_schedule) - 1):
            print(f"  Iter {i:>5}/{len(beta_schedule)}  "
                  f"T={1.0/beta:.4f}  E={total_energy:.1f}  "
                  f"best={best_energy:.1f}")

    elapsed = time.perf_counter() - t0
    print(f"Total annealing time: {elapsed:.6f} seconds")

    return spins, total_energy, best_energy, energy_history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated SA for Ising ground-state search "
                    "(Mac MPS / CUDA)")

    # CSR input files (same flags as the CUDA binary)
    parser.add_argument("-R", "--row_ptr", required=True,
                        help="Row-pointer CSV file")
    parser.add_argument("-C", "--col_idx", required=True,
                        help="Column-index CSV file")
    parser.add_argument("-V", "--values", required=True,
                        help="J-values CSV file")
    parser.add_argument("-l", "--linear", required=True,
                        help="h-vector CSV file")

    # SA parameters
    parser.add_argument("-x", "--start_temp", type=float, default=20.0)
    parser.add_argument("-y", "--stop_temp", type=float, default=0.001)
    parser.add_argument("-c", "--alpha", type=float, default=0.95)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-n", "--num_temps", type=int, default=1000,
                        help="(ignored -- step count derived from alpha)")
    parser.add_argument("-m", "--sweeps_per_beta", type=int, default=1)

    # Flags
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-e", "--no_early_stop", action="store_true")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "mps", "cuda", "cpu"],
                        help="Compute device (default: auto-detect)")

    args = parser.parse_args()

    # ---- Seed ----
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    else:
        seed = int(time.time() * os.getpid()) & 0x7FFFFFFF
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Using seed: {seed}")

    # ---- Device selection ----
    if args.device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---- Load CSR arrays ----
    print("Loading sparse J matrix ...")
    row_ptr_np = load_csv_int(args.row_ptr)
    col_idx_np = load_csv_int(args.col_idx)
    J_values_np = load_csv_float(args.values)

    num_spins = len(row_ptr_np) - 1
    nnz = len(J_values_np)
    print(f"Sparse J loaded: num_spins = {num_spins}, nnz = {nnz}")

    # ---- Load h vector ----
    h_np = load_csv_float(args.linear)
    if len(h_np) != num_spins:
        raise ValueError(
            f"h vector length ({len(h_np)}) != num_spins ({num_spins})")

    # ---- Move CSR arrays to GPU as flat tensors ----
    row_ptr_t = torch.tensor(row_ptr_np, dtype=torch.long, device=device)
    col_idx_t = torch.tensor(col_idx_np, dtype=torch.long, device=device)
    J_values_t = torch.tensor(J_values_np, dtype=torch.float32, device=device)
    h_t = torch.tensor(h_np, dtype=torch.float32, device=device)

    # Expand row_ptr -> per-nnz row indices (one-time, stays on GPU)
    row_idx_t = expand_row_ptr(row_ptr_t, num_spins)
    print(f"CSR on device: row_idx[{row_idx_t.shape[0]}], "
          f"col_idx[{col_idx_t.shape[0]}], J_values[{J_values_t.shape[0]}]")

    # ---- Random spin initialisation (-1 / +1) ----
    spins = (2 * torch.randint(0, 2, (num_spins,),
                                device=device) - 1).float()

    # ---- Build cooling schedule ----
    beta_schedule = create_beta_schedule(
        args.start_temp, args.stop_temp, args.alpha)
    print(f"Beta schedule: {len(beta_schedule)} steps, "
          f"T: {args.start_temp} -> {args.stop_temp}, alpha = {args.alpha}")

    # ---- Run SA ----
    spins, final_energy, best_energy, energy_history = run_sa(
        spins, h_t, col_idx_t, J_values_t, row_idx_t, num_spins,
        beta_schedule,
        num_sweeps_per_beta=args.sweeps_per_beta,
        disable_early_stop=args.no_early_stop,
        debug=args.debug,
    )

    # ---- Verify energy from scratch ----
    verify_energy = compute_total_energy(
        spins, h_t, col_idx_t, J_values_t, row_idx_t, num_spins)

    print(f"\n--- Results ---")
    print(f"  Final energy (tracked):  {final_energy:.6f}")
    print(f"  Final energy (verified): {verify_energy:.6f}")
    print(f"  Best energy:             {best_energy:.6f}")

    # ---- Extract run suffix (e.g. "39203" from J_values_39203.csv) ----
    base = os.path.splitext(os.path.basename(args.values))[0]
    parts = base.rsplit("_", 1)
    run_suffix = parts[-1] if len(parts) > 1 else base

    # ---- Write energy history ----
    energy_filename = f"energy_history_{run_suffix}"
    with open(energy_filename, "w") as f:
        f.write("# Iteration\tBest_Energy\n")
        for idx, e in enumerate(energy_history):
            f.write(f"{idx}\t{e:.6f}\n")
    print(f"  Energy history -> {energy_filename}")

    # ---- Optionally write final spins ----
    if args.debug:
        spins_filename = f"spins_{run_suffix}"
        cpu_spins = spins.cpu().numpy().astype(int)
        with open(spins_filename, "w") as f:
            f.write("\t".join(str(s) for s in cpu_spins))
            f.write(f"\n\n\n\ttotal energy value: {verify_energy:.6f}\n")
        print(f"  Final spins   -> {spins_filename}")


if __name__ == "__main__":
    main()
