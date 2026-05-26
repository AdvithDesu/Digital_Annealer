# Paper draft

First-iteration draft of the paper *GPU-Accelerated Graph-Colored Simulated Annealing for Integer Factorization*.

## Files

- `paper.tex` — RevTeX 4-2, two-column. Compile with `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`.
- `paper.bib` — bibliography skeleton; expand as needed.

## Figure placeholders

Every figure currently renders as a labelled box (`\figplaceholder` macro at the top of `paper.tex`). To swap in real plots:

1. Drop the PNG into `Paper/figures/` (or anywhere reachable).
2. Replace the `\figplaceholder{...}` or `\figplaceholderwide{...}` call with a standard `\begin{figure} \includegraphics[width=\columnwidth]{...} \caption{...} \label{fig:...} \end{figure}`.

Currently placeholders are placed at:

| Section | What the figure should show | Likely source in `Images/` |
|---|---|---|
| §4 | $J$ sparsity at 162 / 379 / 563 / 902 spins | from presentation (slides 19–20) |
| §4 | Spin count vs hub count vs bit width | `scaling of number of spins with N.png` |
| §5 / §7 | SA time per iteration vs spin count (CUDA / PyTorch GH200 / Mac MPS) | `spins_vs_time_per_iteration_for_mac_gh200.png`, `Execution Time for graph coloring SA.png` |
| §7 | Construction + annealing wall-clock vs bits | `Construction and Annealing Time.png` |
| §7 | Solution quality % gap (three estimators) | `Percentage Gap between real and estimated P.png`, `percentage_difference_predicted_factors.png` |
| §7 | Brute-force time vs raw gap | `Raw gap vs Time taken to find solution.png` |

## Section order

The flow rule (every section motivates the next) is the load-bearing structural assumption. If you re-order anything, check that the bridge sentences at the end of each section still make sense.

## Known gaps for the next pass

- Cite Shor / Metropolis / Kirkpatrick at the right spots (entries already in `paper.bib`).
- Cross-check the $\sim 100$-bit case study numbers against the actual SA log; the values in §7 come from the presentation, not from a fresh re-run.
- The mean-gap numbers (10.33% / 4.23% / 2.98%) are presentation values; regenerate from `Multiseed_Benchmark/results.csv` for the final draft.
- Algorithm boxes use `algpseudocode`; tweak typography (line numbers, indent) once style is settled.
- §1 currently leads with the bullet list of the three novelty pillars. Some venues prefer a contributions paragraph; restructure to taste.
