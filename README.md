# Digital Annealer

GPU-accelerated simulated annealing for integer factorization. Given a semiprime `N = P * Q`, the pipeline builds a QUBO whose ground state encodes the factors, anneals it on a CUDA GPU (currently tuned for NVIDIA Hopper / GH200, `sm_90`), and recovers `P` and `Q` from the resulting spin configuration.

## Pipeline

```
                         (C++)                    (CUDA)                     (C++)
   N (integer)  ──►  QUBO construction  ──►  Simulated annealing  ──►  Factor extraction
                     row_ptr_N.csv            spins_N                   P, Q
                     col_idx_N.csv            energy_history_N
                     J_values_N.csv
                     h_vector_N.csv
```

Each stage is a standalone binary; the scripts in [Factorize/](Factorize/), [Benchmark/](Benchmark/), and [Multiseed_Benchmark/](Multiseed_Benchmark/) wire them together.

## Repository layout

| Path | What it is |
|---|---|
| [QUBO_Construction/](QUBO_Construction/) | C++ builder that turns `N` into Ising CSR + metadata. Includes Python reference port and diagnostic tools. |
| [Simulated Annealing/](Simulated%20Annealing/) | CUDA SA implementation. Three `.cu` entry points in [src/](Simulated%20Annealing/src/); CMake builds [Graph_Colored_SA.cu](Simulated%20Annealing/src/Graph_Colored_SA.cu). |
| [Factorize/](Factorize/) | `factorize.sh` — end-to-end one-shot factorization of a single `N`. |
| [Benchmark/](Benchmark/) | Sweeps. `benchmark.sh` runs the full pipeline across bit sizes; `qubo_scaling.sh` measures QUBO construction only. |
| [Multiseed_Benchmark/](Multiseed_Benchmark/) | Solution-quality study: multiple independent SA runs per `N`, keeps the best. |
| [Bruteforce/](Bruteforce/) | C++ multi-threaded wheel-sieve divisor search; used as a quality baseline against SA output. |
| [Annealing on GH200/](Annealing%20on%20GH200/) | Thin wrappers to run SA standalone on pre-built CSR files (`run_single_cuda.sh`, `run_all_cuda.sh`). |
| [Annealing on Mac/](Annealing%20on%20Mac/) | PyTorch port of the CUDA SA that runs on Apple Silicon (MPS) or CUDA. Also holds historical timing logs. |
| [bin_SI/](bin_SI/) | CSR + h-vector files (output of QUBO construction). Also the install destination for the SA binary. |
| [cmake/](cmake/) | `FindCUDALibs.cmake` and helpers used at configure time. |

## Prerequisites

- **Linux + CUDA toolkit ≥ 12.x** (path `/usr/local/cuda/bin/nvcc`; configurable via `cmake -DCMAKE_CUDA_COMPILER=...`).
- **CMake ≥ 3.8**, **g++** with C++17, **GNU Make**.
- **Python 3** with `sympy` for prime generation, plus `torch` if using the Mac/MPS variant.
- GPU target is `sm_90` (Hopper / GH200). Other architectures: edit `CMAKE_CUDA_ARCHITECTURES` in [CMakeLists.txt](CMakeLists.txt).

## Build

```bash
cmake -S . -B build
make -C build -j$(nproc)
```

This produces the SA binary at `build/Simulated Annealing/annealer_gpu_SI`. The QUBO and Bruteforce binaries are compiled on first use by their wrapper scripts (no manual step needed).

## Quick start

Factor a semiprime, end-to-end:

```bash
./Factorize/factorize.sh 11875502333273
```

Or generate two primes yourself and factor their product:

```bash
./Factorize/factorize.sh -p 1234567 -q 7654321
```

Every script can be run from anywhere — they auto-locate the repo root and operate relative to it.

## Detailed usage

### Single factorization — [Factorize/factorize.sh](Factorize/factorize.sh)

```bash
./Factorize/factorize.sh <N> [options]
./Factorize/factorize.sh -p <P> -q <Q> [options]
```

Runs QUBO construction → SA → factor extraction. Key options:

| Flag | Default | Meaning |
|---|---|---|
| `-x <T>` | 100.0 | Start temperature. Pass `-x auto` to estimate via Ben-Ameur sampling. |
| `-y <T>` | 0.1 | Stop temperature. |
| `-c <a>` | 0.95 | Geometric cooling rate. |
| `-m <M>` | 10 | Sweeps per beta. |
| `-s <S>` | random | RNG seed (omit for nondeterministic). |
| `-b` | off | Replacement backtracking (smaller QUBO). |
| `-n` | off | Per-clause max-abs normalization of `|J|`. |
| `-d` | off | Verbose debug output from SA. |
| `--auto-accept-rate <F>` | 0.5 | Target uphill accept rate when `-x auto`. |

Outputs go to `results/` (spins + per-iteration energy history) and `bin_SI/` (CSR files). `factorize.sh -h` for the full list.

### Bit-size sweep — [Benchmark/benchmark.sh](Benchmark/benchmark.sh)

For each `B` in a list, generates two random `B`-bit primes, runs the full pipeline, and writes a CSV row with timings and correctness:

```bash
./Benchmark/benchmark.sh -B 8,16,24,32 -o results.csv
```

Default bit range is `8,10,12,...,62`. SA flags from `factorize.sh` are forwarded (`-x`, `-y`, `-c`, `-m`, `-a`).

CSV columns: `bits, P, Q, N, P_pred, Q_pred, correct, t_construct, t_anneal, best_energy`.

### QUBO scaling — [Benchmark/qubo_scaling.sh](Benchmark/qubo_scaling.sh)

Construction-only sweep — no SA, no post-processing. Used to characterize how QUBO size and connectivity scale with bit size, including an adaptive dense/sparse threshold for the hub-vs-bulk split that `Graph_Colored_SA.cu` consumes.

```bash
./Benchmark/qubo_scaling.sh -B 8,16,24,32,40 -o qubo_scaling.csv
```

### Multi-seed quality benchmark — [Multiseed_Benchmark/multiseed_benchmark.sh](Multiseed_Benchmark/multiseed_benchmark.sh)

For each bit size, runs SA `K` times (default 10) with independent seeds, keeps the best-energy run, and post-processes only the winner. Useful for measuring SA's solution-quality variance.

```bash
./Multiseed_Benchmark/multiseed_benchmark.sh -k 10 -B 16,24,32
```

Per-run spin states are kept under `Multiseed_Benchmark/spins/` (use `-K` to delete losers after each `B`).

### Bruteforce baseline — [Bruteforce/bruteforce_bench.sh](Bruteforce/bruteforce_bench.sh)

Reads [Benchmark/benchmark_results.csv](Benchmark/benchmark_results.csv) and, for each row, runs a multi-threaded wheel-sieve divisor search starting near the SA-predicted factor. Compares SA-guided bruteforce time to a 600-second timeout.

```bash
./Bruteforce/bruteforce_bench.sh
```

The standalone binary [Bruteforce.cpp](Bruteforce/Bruteforce.cpp) accepts both `Bruteforce N P_pred Q_pred [threads]` and `Bruteforce -pq P Q P_pred Q_pred [threads]` forms.

### Re-running SA on existing CSR files

To re-anneal a problem whose CSR is already in `bin_SI/` without rebuilding the QUBO:

```bash
# Single instance (with parsed summary)
./"Annealing on GH200/run_single_cuda.sh" 11875502333273 -x auto -y 1e-8 -c 0.95 -m 10

# Sweep every CSR pair in bin_SI/
./"Annealing on GH200/run_all_cuda.sh"
```

### Non-CUDA / Mac variant — [Annealing on Mac/Non_CUDA_GPU_Simulated_Annealing.py](Annealing%20on%20Mac/Non_CUDA_GPU_Simulated_Annealing.py)

PyTorch port of `Graph_Colored_SA.cu`. Runs the same CSR-form SA on Apple Silicon (MPS) or CUDA via PyTorch's gather/scatter primitives. Same CLI shape as the CUDA binary:

```bash
python3 "Annealing on Mac/Non_CUDA_GPU_Simulated_Annealing.py" \
    -R bin_SI/row_ptr_39203.csv \
    -C bin_SI/col_idx_39203.csv \
    -V bin_SI/J_values_39203.csv \
    -l bin_SI/h_vector_39203.csv \
    -x 20 -y 0.001 -c 0.95 -d
```

## Output files

| File | Produced by | Contents |
|---|---|---|
| `bin_SI/row_ptr_<N>.csv` | QUBO construction | CSR row pointers for J. |
| `bin_SI/col_idx_<N>.csv` | QUBO construction | CSR column indices. |
| `bin_SI/J_values_<N>.csv` | QUBO construction | CSR non-zero couplings. |
| `bin_SI/h_vector_<N>.csv` | QUBO construction | Linear (Ising bias) terms. |
| `qubo_metadata/*_<N>.txt` | QUBO construction | Variable map + constraint logs (post-processing input). |
| `results/spins_<N>` | SA | Best spin configuration (one `±1` per spin). |
| `results/energy_history_<N>` | SA | Best energy after each beta step. |

## SA variants in `Simulated Annealing/src/`

| File | Built? | Notes |
|---|---|---|
| [Graph_Colored_SA.cu](Simulated%20Annealing/src/Graph_Colored_SA.cu) | **Yes** | Production kernel. Graph-coloured update with dense/sparse bins, adaptive hub threshold, Ben-Ameur start-temp estimation. Uses the `-R/-C/-V/-l/-O` CSR interface that every script expects. |
| [Flip_Single_SA.cu](Simulated%20Annealing/src/Flip_Single_SA.cu) | No | Older single-flip Metropolis on a full J matrix (`-a` flag). Kept for reference. |
| [Flip_All_SA.cu](Simulated%20Annealing/src/Flip_All_SA.cu) | No | Experimental synchronous all-spin update. |

To switch the build target, edit the `set(sources ...)` line in [Simulated Annealing/CMakeLists.txt](Simulated%20Annealing/CMakeLists.txt) and rebuild.

## Acknowledgements

Thanks to Dr. Anil Prabhakar and Dr. Nitin Chandrachoodan (IIT Madras) for guidance, and to Dhruv for the QUBO-to-Ising conversion notes that seeded this work.
