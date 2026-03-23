from subprocess import check_output as probe, DEVNULL
import torch as pt
import time

GB = 1024**3


def chonkyness() -> int:
    try:
        ram = int(probe(["sysctl", "-n", "hw.memsize"], stderr=DEVNULL))
    except Exception:
        ram = 8 * GB

    return int(ram * 0.9)


def get_batch_size(num_spins: int, nnz: int, req: int) -> int:
    perChain = (2 * nnz + 5 * num_spins) * 4
    budget = chonkyness()

    B_max = max(1, int(budget // perChain))
    if B_max < req:
        print(
            f"Capping chains {req} -> {B_max} to stay within "
            f"{budget // GB} GB (nnz={nnz}, N={num_spins})"
        )

    return min(req, B_max)


def refuse_small(device: pt.device, num_spins: int, perBeta: int) -> pt.device:
    if perBeta <= 1 and num_spins < 1000:
        print(f"Small for MPS ({num_spins} spins, 1 sweep), fallback CPU")

        return pt.device("cpu")

    return device


def run_sa_mps(
    spins: pt.Tensor,
    h: pt.Tensor,
    col_idx: pt.Tensor,
    J_values: pt.Tensor,
    row_idx: pt.Tensor,
    num_spins: int,
    beta_schedule: list[float],
    num_perBeta: int,
    debug: bool = False,
) -> tuple[pt.Tensor, float, float, list[float]]:
    device = spins.device
    B = get_batch_size(num_spins, col_idx.shape[0], num_perBeta)

    # B independent random chains
    spins_b = (2 * pt.randint(0, 2, (B, num_spins), device=device) - 1).float()

    # Pre-allocate
    ns_buf = pt.zeros(B, num_spins, dtype=pt.float32, device=device)
    row_idx_2d = row_idx.unsqueeze(0).expand(B, -1)  # (B, nnz)

    ns_buf.scatter_add_(1, row_idx_2d, J_values * spins_b[:, col_idx])
    total_energies = 0.5 * (spins_b * ns_buf).sum(dim=1) + (h * spins_b).sum(dim=1)
    best_energies = total_energies.clone()

    energy_history: list[float] = []
    print(
        f"Start annealing: {B} parallel chains, "
        f"best initial energy: {best_energies.min().item():.6f}"
    )
    t0 = time.perf_counter()

    for i, beta in enumerate(beta_schedule):
        ns_buf.zero_()
        ns_buf.scatter_add_(1, row_idx_2d, J_values * spins_b[:, col_idx])

        dE = -2.0 * (ns_buf + h) * spins_b  # (B, N)

        # Metropolis acceptance + 50 % random halving
        flip_mask = (
            pt.rand(B, num_spins, device=device) < pt.clamp(pt.exp(-beta * dE), max=1.0)
        ) & (pt.rand(B, num_spins, device=device) < 0.5)

        spins_b *= 1.0 - 2.0 * flip_mask.float()

        ns_buf.zero_()
        ns_buf.scatter_add_(1, row_idx_2d, J_values * spins_b[:, col_idx])
        total_energies = 0.5 * (spins_b * ns_buf).sum(dim=1) + (h * spins_b).sum(dim=1)
        best_energies = pt.minimum(best_energies, total_energies)

        # One GPU->CPU sync per beta step (not per sweep)
        best_e_now = best_energies.min().item()
        energy_history.append(best_e_now)

        if debug and (i % 100 == 0 or i == len(beta_schedule) - 1):
            print(
                f"  Iter {i:>5}/{len(beta_schedule)}  "
                f"T={1.0/beta:.4f}  best={best_e_now:.1f}"
            )

    elapsed = time.perf_counter() - t0
    print(f"Total annealing time: {elapsed:.6f} seconds")

    best_chain = best_energies.argmin().item()
    return (
        spins_b[best_chain],
        total_energies[best_chain].item(),
        best_energies[best_chain].item(),
        energy_history,
    )
