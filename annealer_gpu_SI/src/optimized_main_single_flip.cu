#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <unistd.h>
#include <getopt.h>

#include <vector>
#include <chrono>

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <cmath>

#define THREADS 1024 //or more threads gpu crashes
#define TCRIT 2.26918531421f

struct FlipCandidate {
    int    spin_id;
    double delta_energy;
};

// Bin thresholds // Spins with row-length >= DENSE_THRESHOLD  → Bin 0: 1 block / 1024 threads
// Spins with row-length <  DENSE_THRESHOLD  → Bin 1: 1 warp  / 32  threads
// packed SPINS_PER_BLOCK_SPARSE at a time into one block of 1024 threads.
#define DENSE_THRESHOLD         128   // tune after inspecting degree histogram
#define SPINS_PER_BLOCK_SPARSE   32   // 32 spins × 32 threads = 1024 threads/block

// Max hubs — used to size shared-memory hub cache in the sparse kernel.
// Raised from 16 because dense spin count scales ~log(N) and we now target N up to ~1M.
#define MAX_HUB_SPINS            256

// High bit tags a sparse neighbor as "this is a hub index into d_hub_vals"
// (low 31 bits = hub index). Otherwise the entry is a plain global spin id.
#define HUB_TAG_BIT              0x80000000
#define HUB_IDX_MASK             0x7FFFFFFF

#include "utils.hpp"

//__constant__ float kd_floats[1000000];


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


// =====================================================================
// Host-side preprocessing: greedy graph coloring + sparse-CSR-with-tagged-hubs
// =====================================================================
//
// Purpose:
//   1. Color the interaction graph (nodes=spins, edges = nonzero J_ij) so
//      that within a color class no two spins interact. We can then flip
//      every accepted spin in a color *in parallel* without breaking the
//      running total-energy accumulator.
//   2. Build a sparse-only CSR for the sparse-bin spins where col_idx is
//      tagged: high bit set => remaining bits are an index into d_hub_vals
//      (hot path), high bit clear => raw global spin id (rare fallback).
//
// Both are O(N + nnz) and run once at setup.

struct ColoringTables {
    int num_colors = 0;

    // Per-color dense spin lists (global spin ids), concatenated.
    std::vector<int> color_dense_flat;
    std::vector<int> color_dense_offsets;   // size num_colors+1

    // Per-color sparse spin lists. These are *indices into the sparse CSR*
    // (0..num_sparse-1), NOT global spin ids. The sparse kernel maps them
    // back to global ids via sparse_global_id[].
    std::vector<int> color_sparse_flat;
    std::vector<int> color_sparse_offsets;  // size num_colors+1
};

struct SparseCSR {
    std::vector<int>     row_ptr;           // size num_sparse+1
    std::vector<int>     col_idx_tagged;    // tagged per HUB_TAG_BIT
    std::vector<double>  J_values;
    std::vector<int>     global_id;         // size num_sparse; sparse-idx -> global spin id
    int                  num_sparse = 0;
    int                  nnz_hub_fallback = 0;
};

// Sampled symmetry check for the J-matrix CSR.
// The energy formula E = ½ s'Js + h's and the ΔE formula
//   dE = -2 s_i (Σ_j J_ij s_j + h_i)
// are correct only when the CSR stores BOTH J_ij and J_ji with equal value.
// We verify this on a random subset of rows: cheap (microseconds even at N=1e6),
// catches accidental upper-triangular-only inputs.
static void verifySymmetrySampled(
    const std::vector<int>&    row_ptr,
    const std::vector<int>&    col_idx,
    const std::vector<double>& J_values,
    int                        num_spins,
    int                        num_sample_rows = 128,
    double                     rel_tol = 1e-9)
{
    if (num_spins <= 0) return;

    // Deterministic sampler so failures are reproducible across runs.
    unsigned int rng_state = 0xC0FFEEu ^ (unsigned int)num_spins;
    auto next_rand = [&]() {
        rng_state = rng_state * 1664525u + 1013904223u;
        return rng_state;
    };

    int rows_to_sample = std::min(num_sample_rows, num_spins);
    int missing = 0, mismatched = 0, checked = 0;

    for (int s = 0; s < rows_to_sample; s++) {
        int i = (int)(next_rand() % (unsigned int)num_spins);
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
            int    j   = col_idx[k];
            double Jij = J_values[k];
            if (j == i) continue;  // diagonals don't need a partner
            checked++;

            // Search row j for column i (linear; rows are short by assumption).
            bool found = false;
            for (int kk = row_ptr[j]; kk < row_ptr[j + 1]; kk++) {
                if (col_idx[kk] == i) {
                    double Jji = J_values[kk];
                    double scale = std::max(1.0, std::max(std::fabs(Jij), std::fabs(Jji)));
                    if (std::fabs(Jij - Jji) > rel_tol * scale) mismatched++;
                    found = true;
                    break;
                }
            }
            if (!found) missing++;
        }
    }

    if (missing || mismatched) {
        fprintf(stderr,
                "FATAL: J-matrix CSR is not symmetric "
                "(sampled %d rows, %d entries; missing partners=%d, value mismatches=%d).\n"
                "Energy and dE assume both J_ij and J_ji are stored with equal value.\n",
                rows_to_sample, checked, missing, mismatched);
        std::exit(1);
    }
    std::cout << "Symmetry check passed (sampled " << rows_to_sample
              << " rows, " << checked << " entries)." << std::endl;
}

// Welsh-Powell style greedy coloring: largest-degree-first, first-fit.
// Returns color_of[num_spins] and num_colors via out-param.
static std::vector<int> buildGreedyColoring(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    int num_spins,
    int& num_colors_out)
{
    std::vector<int> order(num_spins);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        int da = row_ptr[a + 1] - row_ptr[a];
        int db = row_ptr[b + 1] - row_ptr[b];
        return da > db;
    });

    std::vector<int> color_of(num_spins, -1);
    int num_colors = 0;
    std::vector<char> used;   // reused scratch, size tracks num_colors
    used.reserve(64);

    for (int v : order) {
        used.assign(num_colors + 1, 0);
        int start = row_ptr[v];
        int end   = row_ptr[v + 1];
        for (int k = start; k < end; k++) {
            int n = col_idx[k];
            int cn = color_of[n];
            if (cn >= 0) used[cn] = 1;
        }
        int c = 0;
        while (c < (int)used.size() && used[c]) c++;
        color_of[v] = c;
        if (c >= num_colors) num_colors = c + 1;
    }

    // Sanity: no two adjacent spins share a color. Unconditional — asserts
    // get stripped under NDEBUG and we want this to fire even in Release.
    int bad_edges = 0;
    for (int i = 0; i < num_spins; i++) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
            int j = col_idx[k];
            if (i == j) continue;
            if (color_of[i] == color_of[j]) bad_edges++;
        }
    }
    if (bad_edges) {
        fprintf(stderr,
                "FATAL: coloring broken, %d edges connect same-color spins\n",
                bad_edges);
        std::exit(1);
    }

    num_colors_out = num_colors;
    return color_of;
}

// Build the sparse-only CSR with hub-tagged col_idx.
static SparseCSR buildSparseCSR(
    const std::vector<int>&     row_ptr,
    const std::vector<int>&     col_idx,
    const std::vector<double>&  J_values,
    const std::vector<int>&     dense_spins,
    const std::vector<int>&     sparse_spins,
    int                         num_spins)
{
    SparseCSR out;
    int num_sparse = (int)sparse_spins.size();
    out.num_sparse = num_sparse;
    out.row_ptr.resize(num_sparse + 1);
    out.global_id.resize(num_sparse);

    // global_id -> hub_idx, -1 if not a hub
    std::vector<int> hub_map(num_spins, -1);
    for (int i = 0; i < (int)dense_spins.size(); i++)
        hub_map[dense_spins[i]] = i;

    out.row_ptr[0] = 0;
    for (int s = 0; s < num_sparse; s++) {
        int gid = sparse_spins[s];
        out.global_id[s] = gid;
        int start = row_ptr[gid];
        int end   = row_ptr[gid + 1];
        for (int k = start; k < end; k++) {
            int    nb = col_idx[k];
            double J  = J_values[k];
            int h = hub_map[nb];
            if (h >= 0) {
                out.col_idx_tagged.push_back((int)(h | HUB_TAG_BIT));
            } else {
                out.col_idx_tagged.push_back(nb);
                out.nnz_hub_fallback++;
            }
            out.J_values.push_back(J);
        }
        out.row_ptr[s + 1] = (int)out.col_idx_tagged.size();
    }
    return out;
}

// Bucket dense/sparse spins by color.
// dense spins are stored as global ids; sparse spins as indices into the sparse CSR.
static ColoringTables buildColoringTables(
    const std::vector<int>& color_of,
    int                     num_colors,
    const std::vector<int>& dense_spins,     // global ids
    const SparseCSR&        scsr)
{
    ColoringTables ct;
    ct.num_colors = num_colors;
    ct.color_dense_offsets.assign(num_colors + 1, 0);
    ct.color_sparse_offsets.assign(num_colors + 1, 0);

    // Pass 1: count
    for (int g : dense_spins)
        ct.color_dense_offsets[color_of[g] + 1]++;
    for (int s = 0; s < scsr.num_sparse; s++)
        ct.color_sparse_offsets[color_of[scsr.global_id[s]] + 1]++;

    // Prefix sum
    for (int c = 0; c < num_colors; c++) {
        ct.color_dense_offsets[c + 1]  += ct.color_dense_offsets[c];
        ct.color_sparse_offsets[c + 1] += ct.color_sparse_offsets[c];
    }

    ct.color_dense_flat.resize(ct.color_dense_offsets[num_colors]);
    ct.color_sparse_flat.resize(ct.color_sparse_offsets[num_colors]);

    // Pass 2: scatter using per-color cursors
    std::vector<int> cur_dense  = ct.color_dense_offsets;
    std::vector<int> cur_sparse = ct.color_sparse_offsets;
    for (int g : dense_spins) {
        int c = color_of[g];
        ct.color_dense_flat[cur_dense[c]++] = g;
    }
    for (int s = 0; s < scsr.num_sparse; s++) {
        int c = color_of[scsr.global_id[s]];
        ct.color_sparse_flat[cur_sparse[c]++] = s;
    }
    return ct;
}

// Spin-only init (forward decl)
__global__ void init_spins_only(
		const float* __restrict__ randvals,
		signed char*              gpuSpins,
		curandState*              state,
		unsigned long             seed,
		int                       num_spins);

// Compute total energy E = ½ s'Js + h's from the current spin state.
__global__ void compute_total_energy(
		const int*           row_ptr,
		const int*           col_idx,
		const double*        J_values,
		const double*        gpuLinTermsVect,
		signed char*         gpuSpins,
		const unsigned int*  gpu_num_spins,
		double*              total_energy
);


__global__ void collectFlipCandidates_dense(
        const int*           row_ptr,
        const int*           col_idx,
        const double*        J_values,
        const double*        gpuLinTermsVect,
        const float* __restrict__ randvals,
        signed char*         gpuLatSpin,
        const double         beta,
        FlipCandidate*       candidates,
        int*                 num_candidates,
        const int*           dense_spin_ids   // maps blockIdx.x → global spin id (for this color)
);


__global__ void collectFlipCandidates_sparse(
        const int*           sparse_row_ptr,
        const int*           sparse_col_idx_tagged,
        const double*        sparse_J_values,
        const int*           sparse_global_id,
        const double*        gpuLinTermsVect,
        const float* __restrict__ randvals,
        signed char*         gpuLatSpin,
        const signed char*   d_hub_vals,
        int                  num_hubs,
        const double         beta,
        FlipCandidate*       candidates,
        int*                 num_candidates,
        const int*           color_sparse_csr_ids,   // sparse-CSR indices for this color
        int                  num_in_color
);


__global__ void applyAllFlipsInColor(
        signed char*         gpuLatSpin,
        double*              d_total_energy,
        const FlipCandidate* candidates,
        const int*           num_candidates
);


__global__ void refreshHubVals(
        const signed char* gpuLatSpin,
        const int*         d_hub_ids,
        signed char*       d_hub_vals,
        int                num_hubs
);


// Per-spin ΔE estimator for auto start-temperature (Ben-Ameur style).
// Each block handles one spin i, computes ΔE_i = -2 s_i (Σ_j J_ij s_j + h_i)
// via the same shared-mem reduction as compute_total_energy, then atomically
// accumulates the uphill (ΔE > 0) contributions into (sum_pos, count_pos).
__global__ void computeDeltaE_uphill_accum(
        const int*           row_ptr,
        const int*           col_idx,
        const double*        J_values,
        const double*        gpuLinTermsVect,
        const signed char*   gpuSpins,
        double*              d_sum_pos,
        int*                 d_count_pos
);

std::vector<double> create_beta_schedule_geometric(double temp_start, double temp_end, double alpha);
  
static void usage(const char *pname) {

	const char *bname = nullptr;//@R = rindex(pname, '/');

	fprintf(stdout,
		"Usage: %s [options]\n"
		"options:\n"
		"\t-R|--row_ptr_file <FILENAME>\n"
		"\t\t Row pointer vector for J values (indicating start of new row in J matrix)\n"
		"\n"
		"\t-C|--col_idx_file <FILENAME>\n"
		"\t\t Column Index vector for J values (indicating position of non-zero values in specific row)\n"
		"\n"
		"\t-V|--values_file  <FILENAME>\n"
		"\t\t Vector of all non-zero J matrix values arranged in CSR format \n"
		"\n"
		"\t-x|--start temperature <FLOAT>\n"
		"\t\t \n"
		"\n"
		"\t-y|--stop temperature <FLOAT>\n"
		"\t\tnumber of lattice columns\n"
		"\n"
		"\t-c|--alpha <FLOAT>\n"
		"\t\tcooling rate (temperature multiplier, 0 < alpha < 1, default: 0.95)\n"
		"\t\t(number of temperature steps is derived from start/stop/alpha)\n"
		"\n"
		"\t-m|--sweeps_per_beta <INT>\n"
		"\t\tnumber of sweep per temperature\n"
		"\n"
		"\t-s|--seed <SEED>\n"
		"\t\tfix the starting point\n"
		"\n"
		"\t-d|--debug \n"
		"\t\t Print debug output\n"
		"\n"
		"\t-O|--output-dir <DIR>\n"
		"\t\tdirectory for output files (spins, energy history)\n\n",
		bname);
	exit(EXIT_SUCCESS);
}

int main(int argc, char* argv[])
{
	// arguments for files required by the CSR representation
	std::string row_ptr_file = "";
	std::string col_idx_file = "";
	std::string values_file  = "";
	
	std::string linear_file = "";
	
	double start_temp = 20.0;
	double stop_temp = 0.001;
	double alpha = 0.95;
	bool   auto_start_temp  = false;
	double auto_accept_rate = 0.5;
	int    auto_n_config    = 10;
	unsigned long long seed = ((getpid()* rand()) & 0x7FFFFFFFF); // ((GetCurrentProcessId()* rand()) & 0x7FFFFFFFF);
	
	unsigned int num_sweeps_per_beta = 1;//atoi(argv[3]);
	
	std::string output_dir = "";
	bool debug = false;
	
	std::vector<double> energy_history;  // Store best energy at each iteration
	
	std::cout << "Start parsing the file " << std::endl;
	
	while (1) {
		static struct option long_options[] = {
			{ "row_ptr_file", required_argument, 0, 'R' },
			{ "col_idx_file", required_argument, 0, 'C' },
			{ "values_file",  required_argument, 0, 'V' },
			{ "Linear_file", required_argument, 0, 'l' },
			{     "start_temp", required_argument, 0, 'x'},
			{     "stop_temp", required_argument, 0, 'y'},
			{          "seed", required_argument, 0, 's'},
			{ "alpha", required_argument, 0, 'c'},
			{ "sweeps_per_beta", required_argument, 0, 'm'},
			{ "output-dir", required_argument, 0, 'O'},
			{          "debug",       no_argument, 0, 'd'},
			{          "help",       no_argument, 0, 'h'},
			{ "auto-accept-rate", required_argument, 0, 1001 },
			{ "auto-n-config",    required_argument, 0, 1002 },
			{               0,                 0, 0,   0}
		};

		int option_index = 0;
		int ch = getopt_long(argc, argv, "R:C:V:l:x:y:s:c:m:O:dh", long_options, &option_index);
		if (ch == -1) break;

		switch (ch) {
		case 0:
			break;
		case 'R':
		    row_ptr_file = optarg; break;
		case 'C':
		    col_idx_file = optarg; break;
		case 'V':
		    values_file = optarg; break;
		case 'l':
			linear_file = (optarg); break;
		case 'x':
			if (std::string(optarg) == "auto") {
				auto_start_temp = true;
			} else {
				start_temp = atof(optarg);
			}
			break;
		case 1001:
			auto_accept_rate = atof(optarg); break;
		case 1002:
			auto_n_config = atoi(optarg); break;
		case 'y':
			stop_temp = atof(optarg); break;
		case 's':
			seed = atoll(optarg);
			break;
		case 'c': 
			alpha = atof(optarg); 
			break;
		case 'm':
			num_sweeps_per_beta = atoi(optarg); break;
		case 'O':
			output_dir = optarg;
			if (!output_dir.empty() && output_dir.back() != '/')
				output_dir += '/';
			break;
 		case 'd':
			debug = true; break;
		case 'h':
			usage(argv[0]); break;
		case '?':
			exit(EXIT_FAILURE);
		default:
			fprintf(stderr, "unknown option: %c\n", ch);
			exit(EXIT_FAILURE);
		}
	}

	std::cout << "Running sparse SA with:\n"
			<< " start temp " << (auto_start_temp ? std::string("auto") : std::to_string(start_temp))
			<< " stop temp " << stop_temp << "\n"
			<< " seed " << seed << " num sweeps per beta "
			<<  num_sweeps_per_beta << std::endl;

	// --------------------------------------------------
	// Extract run suffix from values_file (e.g. J_values_10403.csv -> 10403)
	// --------------------------------------------------

	std::string run_suffix;
	
	{
	    std::string base = values_file;
	
	    // Remove directory path
	    size_t slash_pos = base.find_last_of("/\\");
	    if (slash_pos != std::string::npos)
	        base = base.substr(slash_pos + 1);
	
	    // Remove extension (.csv)
	    size_t dot_pos = base.find_last_of(".");
	    if (dot_pos != std::string::npos)
	        base = base.substr(0, dot_pos);
	
	    // Extract suffix after last underscore
	    size_t underscore_pos = base.find_last_of("_");
	    if (underscore_pos != std::string::npos)
	        run_suffix = base.substr(underscore_pos + 1);
	    else
	        run_suffix = base;  // fallback (should not happen)
	}

	// ---------------- Sparse J loading ----------------

	if (row_ptr_file.empty() || col_idx_file.empty() || values_file.empty()) {
	    std::cerr << "ERROR: Must provide --row_ptr_file, --col_idx_file, and --values_file\n";
	    exit(1);
	}

	auto t_total_start = std::chrono::high_resolution_clock::now();

 	double starttime = rtclock();

	auto t_load_start = std::chrono::high_resolution_clock::now();

	// Parse data for CSR representation of J matrix
	ParseSparseData parseSparse(row_ptr_file, col_idx_file, values_file);

    std::cout << "ParseData constructed successfully" << std::endl;
	unsigned int num_spins = parseSparse.getNumSpins();

	std::vector<double> linearTermsVect;
	//if (linear_file.empty() == false)
    readLinearValues(linear_file, num_spins, linearTermsVect);

	auto t_load_end = std::chrono::high_resolution_clock::now();
	double t_load = (double)std::chrono::duration_cast<std::chrono::microseconds>(t_load_end - t_load_start).count() * 1e-6;

	double endtime = rtclock();

    if(debug)
  	  printtime("ParseData time: ", starttime, endtime);

	unsigned int nnz = parseSparse.getNNZ();
	
	const std::vector<int>&     row_ptr  = parseSparse.getRowPtr();
	const std::vector<int>&     col_idx  = parseSparse.getColIdx();
	const std::vector<double>&  J_values = parseSparse.getValues();
	
	std::cout << "Sparse J loaded: num_spins = "
	          << num_spins << ", nnz = " << nnz << std::endl;

	// Verify CSR symmetry assumption (cheap sampled check; aborts on failure).
	verifySymmetrySampled(row_ptr, col_idx, J_values, (int)num_spins);

	auto t_setup_start = std::chrono::high_resolution_clock::now();

	// ── Build spin bins based on row degree ──────────────────────────────────
	std::vector<int> dense_spins, sparse_spins;
	dense_spins.reserve(32);
	sparse_spins.reserve(num_spins);

	for (unsigned int i = 0; i < num_spins; i++) {
	    int degree = row_ptr[i + 1] - row_ptr[i];
	    if (degree >= DENSE_THRESHOLD)
	        dense_spins.push_back(i);
	    else
	        sparse_spins.push_back(i);
	}

	int num_dense  = (int)dense_spins.size();
	int num_sparse = (int)sparse_spins.size();

	std::cout << "Bin sizes (dense): " << num_dense
	          << "  sparse: "          << num_sparse << std::endl;

	if (num_dense > MAX_HUB_SPINS) {
	    std::cerr << "ERROR: num_dense (" << num_dense << ") exceeds MAX_HUB_SPINS ("
	              << MAX_HUB_SPINS << "). Raise the compile-time limit.\n";
	    exit(1);
	}
	int h_num_hub = num_dense;

	// ── Graph coloring (greedy, largest-degree-first) ────────────────────────
	int num_colors = 0;
	std::vector<int> color_of = buildGreedyColoring(row_ptr, col_idx, num_spins, num_colors);
	std::cout << "Graph coloring: num_colors = " << num_colors << std::endl;

	// Color-class size histogram for diagnostic output.
	{
	    std::vector<int> sz(num_colors, 0);
	    for (int v = 0; v < (int)num_spins; v++) sz[color_of[v]]++;
	    int mn = sz[0], mx = sz[0];
	    for (int s : sz) { mn = std::min(mn, s); mx = std::max(mx, s); }
	    std::cout << "  color-class sizes: min=" << mn << " max=" << mx
	              << " avg=" << (num_spins / num_colors) << std::endl;
	}

	// ── Build sparse-only CSR with hub-tagged col_idx ────────────────────────
	SparseCSR scsr = buildSparseCSR(row_ptr, col_idx, J_values,
	                                dense_spins, sparse_spins, num_spins);
	std::cout << "Sparse CSR: num_sparse=" << scsr.num_sparse
	          << " nnz=" << scsr.col_idx_tagged.size()
	          << " non-hub fallbacks=" << scsr.nnz_hub_fallback
	          << " ("
	          << (scsr.col_idx_tagged.empty() ? 0.0
	                : 100.0 * scsr.nnz_hub_fallback / scsr.col_idx_tagged.size())
	          << "%)" << std::endl;

	// ── Bucket dense/sparse spin ids by color ────────────────────────────────
	ColoringTables ct = buildColoringTables(color_of, num_colors, dense_spins, scsr);

	// ── GPU uploads: color tables ────────────────────────────────────────────
	int *gpu_color_dense_flat = nullptr,  *gpu_color_dense_offsets  = nullptr;
	int *gpu_color_sparse_flat = nullptr, *gpu_color_sparse_offsets = nullptr;
	if (!ct.color_dense_flat.empty()) {
	    gpuErrchk(cudaMalloc((void**)&gpu_color_dense_flat,
	                         ct.color_dense_flat.size() * sizeof(int)));
	    gpuErrchk(cudaMemcpy(gpu_color_dense_flat, ct.color_dense_flat.data(),
	                         ct.color_dense_flat.size() * sizeof(int),
	                         cudaMemcpyHostToDevice));
	}
	gpuErrchk(cudaMalloc((void**)&gpu_color_dense_offsets,
	                     ct.color_dense_offsets.size() * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpu_color_dense_offsets, ct.color_dense_offsets.data(),
	                     ct.color_dense_offsets.size() * sizeof(int),
	                     cudaMemcpyHostToDevice));

	if (!ct.color_sparse_flat.empty()) {
	    gpuErrchk(cudaMalloc((void**)&gpu_color_sparse_flat,
	                         ct.color_sparse_flat.size() * sizeof(int)));
	    gpuErrchk(cudaMemcpy(gpu_color_sparse_flat, ct.color_sparse_flat.data(),
	                         ct.color_sparse_flat.size() * sizeof(int),
	                         cudaMemcpyHostToDevice));
	}
	gpuErrchk(cudaMalloc((void**)&gpu_color_sparse_offsets,
	                     ct.color_sparse_offsets.size() * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpu_color_sparse_offsets, ct.color_sparse_offsets.data(),
	                     ct.color_sparse_offsets.size() * sizeof(int),
	                     cudaMemcpyHostToDevice));

	// ── GPU uploads: sparse-only CSR ─────────────────────────────────────────
	int     *gpu_sparse_row_ptr        = nullptr;
	int     *gpu_sparse_col_idx_tagged = nullptr;
	double  *gpu_sparse_J_values       = nullptr;
	int     *gpu_sparse_global_id      = nullptr;
	gpuErrchk(cudaMalloc((void**)&gpu_sparse_row_ptr,
	                     (scsr.num_sparse + 1) * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpu_sparse_row_ptr, scsr.row_ptr.data(),
	                     (scsr.num_sparse + 1) * sizeof(int), cudaMemcpyHostToDevice));
	if (!scsr.col_idx_tagged.empty()) {
	    gpuErrchk(cudaMalloc((void**)&gpu_sparse_col_idx_tagged,
	                         scsr.col_idx_tagged.size() * sizeof(int)));
	    gpuErrchk(cudaMemcpy(gpu_sparse_col_idx_tagged, scsr.col_idx_tagged.data(),
	                         scsr.col_idx_tagged.size() * sizeof(int),
	                         cudaMemcpyHostToDevice));
	    gpuErrchk(cudaMalloc((void**)&gpu_sparse_J_values,
	                         scsr.J_values.size() * sizeof(double)));
	    gpuErrchk(cudaMemcpy(gpu_sparse_J_values, scsr.J_values.data(),
	                         scsr.J_values.size() * sizeof(double),
	                         cudaMemcpyHostToDevice));
	}
	if (scsr.num_sparse > 0) {
	    gpuErrchk(cudaMalloc((void**)&gpu_sparse_global_id,
	                         scsr.num_sparse * sizeof(int)));
	    gpuErrchk(cudaMemcpy(gpu_sparse_global_id, scsr.global_id.data(),
	                         scsr.num_sparse * sizeof(int), cudaMemcpyHostToDevice));
	}

	// ── GPU uploads: hub arrays (global, not constant memory) ────────────────
	int         *gpu_hub_ids  = nullptr;
	signed char *gpu_hub_vals = nullptr;
	if (h_num_hub > 0) {
	    gpuErrchk(cudaMalloc((void**)&gpu_hub_ids,  h_num_hub * sizeof(int)));
	    gpuErrchk(cudaMemcpy(gpu_hub_ids, dense_spins.data(),
	                         h_num_hub * sizeof(int), cudaMemcpyHostToDevice));
	    gpuErrchk(cudaMalloc((void**)&gpu_hub_vals, h_num_hub * sizeof(signed char)));
	}

	// Setup cuRAND generator

	curandGenerator_t rng;
	
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandSetPseudoRandomGeneratorSeed(rng, seed);
	float *gpu_randvals;// same as spins
	gpuErrchk(cudaMalloc((void**)&gpu_randvals, (num_spins) * sizeof(float)));

	double *gpuLinTermsVect;
	gpuErrchk(cudaMalloc((void**)&gpuLinTermsVect, (num_spins) * sizeof(double)));

	if (linearTermsVect.size() != num_spins)
		std::cout << "	[ERROR] error in parsing the linear terms from file" << std::endl;

	gpuErrchk(cudaMemcpy(gpuLinTermsVect, linearTermsVect.data(), (num_spins) * sizeof(double), cudaMemcpyHostToDevice));

	// ---- Allocate sparse J (CSR format) on GPU ----

	int*     gpu_row_ptr  = nullptr;
	int*     gpu_col_idx  = nullptr;
	double*  gpu_J_values = nullptr;

	starttime = rtclock();
	
	// row_ptr
	gpuErrchk(cudaMalloc((void**)&gpu_row_ptr, (num_spins + 1) * sizeof(int)));
	
	gpuErrchk(cudaMemcpy(gpu_row_ptr, row_ptr.data(), (num_spins + 1) * sizeof(int), cudaMemcpyHostToDevice));
	
	// col_idx
	gpuErrchk(cudaMalloc((void**)&gpu_col_idx, nnz * sizeof(int)));
	
	gpuErrchk(cudaMemcpy(gpu_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
	
	// J values
	gpuErrchk(cudaMalloc((void**)&gpu_J_values, nnz * sizeof(double)));

	gpuErrchk(cudaMemcpy(gpu_J_values, J_values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

	endtime = rtclock();
   
	if(debug)
		 printtime("J Matrix data transfer time: ", starttime, endtime);

	unsigned int* gpu_num_spins;
	gpuErrchk(cudaMalloc((void**)&gpu_num_spins, sizeof(*gpu_num_spins)));
	gpuErrchk(cudaMemcpy(gpu_num_spins, &num_spins, sizeof(*gpu_num_spins), cudaMemcpyHostToDevice));
	
	double* gpu_total_energy;
	cudaHostAlloc(&gpu_total_energy, sizeof(double), 0);

	double* d_total_energy;
	gpuErrchk(cudaMalloc((void**)&d_total_energy, sizeof(double)));

	double* gpu_best_energy;
	cudaHostAlloc(&gpu_best_energy, sizeof(double), 0);
 
	signed char *gpu_spins_old;
	gpuErrchk(cudaMalloc((void**)&gpu_spins_old, num_spins * sizeof(signed char)));

	// Best-state mirror: snapshot of gpu_spins_old at the lowest running energy
	// observed during annealing. The output spin file is written from this buffer
	// so we report the best state encountered, not whatever state happened to be
	// live at the final temperature step.
	signed char *gpu_spins_best;
	gpuErrchk(cudaMalloc((void**)&gpu_spins_best, num_spins * sizeof(signed char)));

	FlipCandidate* gpu_candidates;
	gpuErrchk(cudaMalloc((void**)&gpu_candidates, num_spins * sizeof(FlipCandidate)));
	
	int* gpu_num_candidates;
	gpuErrchk(cudaMalloc((void**)&gpu_num_candidates, sizeof(int)));

	std::cout << "initialize spin values " << std::endl;
	// int blocks = (num_spins + THREADS - 1) / THREADS;
	curandGenerateUniform(rng, gpu_randvals, num_spins);

	// is a seed for random number generator
	time_t t;
	time(&t);
 
	// create random states    
	curandState* devRanStates;
	cudaMalloc(&devRanStates, num_spins * sizeof(curandState));

	// Dense kernel: maximise shared memory (large hub row reductions)
	cudaFuncSetAttribute(
	    collectFlipCandidates_dense,
	    cudaFuncAttributePreferredSharedMemoryCarveout,
	    cudaSharedmemCarveoutMaxShared
	);
	 
	// Sparse kernel: no shared memory — maximise L1 cache instead
	cudaFuncSetAttribute(
	    collectFlipCandidates_sparse,
	    cudaFuncAttributePreferredSharedMemoryCarveout,
	    cudaSharedmemCarveoutMaxL1
	);
 	
   	starttime = rtclock();

	{
		double zero = 0.0;
		gpuErrchk(cudaMemcpy(d_total_energy, &zero, sizeof(double), cudaMemcpyHostToDevice));
	}

	// Phase 1: initialize all spins. Must finish before any block reads
	// a neighbor's spin, otherwise the energy compute races with the
	// neighbor block's own spin write.
	{
		int blk = (num_spins + 255) / 256;
		init_spins_only<<<blk, 256>>>(
			gpu_randvals, gpu_spins_old, devRanStates,
			(unsigned long)t, (int)num_spins);
	}
	cudaDeviceSynchronize();

	// Phase 2: compute initial total energy from the now-stable spins.
	compute_total_energy<<<num_spins, THREADS>>>(
		gpu_row_ptr,
		gpu_col_idx,
		gpu_J_values,
		gpuLinTermsVect,
		gpu_spins_old,
		gpu_num_spins,
		d_total_energy);

	cudaDeviceSynchronize();

	gpuErrchk(cudaMemcpy(gpu_total_energy, d_total_energy, sizeof(double), cudaMemcpyDeviceToHost));
      
 	endtime = rtclock();

	printtime("init_spins values and calculate total Energy time: ", starttime, endtime);

	// ── Initialize hub values cache from the live spin array ─────────────
	if (h_num_hub > 0) {
	    int hb = (h_num_hub + 63) / 64;
	    refreshHubVals<<<hb, 64>>>(gpu_spins_old, gpu_hub_ids, gpu_hub_vals, h_num_hub);
	    cudaDeviceSynchronize();
	}
 
	gpuErrchk(cudaPeekAtLastError());

	gpu_best_energy[0] = gpu_total_energy[0];
	gpuErrchk(cudaMemcpy(gpu_spins_best, gpu_spins_old,
	                     num_spins * sizeof(signed char),
	                     cudaMemcpyDeviceToDevice));

	// d_total_energy already holds the correct initial value;
	// applyAllFlipsInColor atomicAdds dE onto it incrementally from here.

	// ── Auto start-temperature estimation (Ben-Ameur, simple form) ──────
	// Sample ΔE over uphill single-spin flips across N_config random configs,
	// set T_0 = <ΔE+> / ln(1/chi) so a "typical" uphill move is accepted with
	// probability chi.  Uses the same GPU buffers + hub cache as annealing.
	if (auto_start_temp) {
	    double *d_sum_pos   = nullptr;
	    int    *d_count_pos = nullptr;
	    gpuErrchk(cudaMalloc((void**)&d_sum_pos,   sizeof(double)));
	    gpuErrchk(cudaMalloc((void**)&d_count_pos, sizeof(int)));

	    double total_sum = 0.0;
	    long long total_cnt = 0;

	    for (int cfg = 0; cfg < auto_n_config; cfg++) {
	        // Fresh random spins for this config.
	        curandGenerateUniform(rng, gpu_randvals, num_spins);
	        {
	            int blk = (num_spins + 255) / 256;
	            init_spins_only<<<blk, 256>>>(
	                gpu_randvals, gpu_spins_old, devRanStates,
	                (unsigned long)t + (unsigned long)(cfg + 1), (int)num_spins);
	        }
	        cudaDeviceSynchronize();

	        // Refresh hub cache for the new spins.
	        if (h_num_hub > 0) {
	            int hb = (h_num_hub + 63) / 64;
	            refreshHubVals<<<hb, 64>>>(gpu_spins_old, gpu_hub_ids, gpu_hub_vals, h_num_hub);
	            cudaDeviceSynchronize();
	        }

	        double zero_d = 0.0;
	        int    zero_i = 0;
	        gpuErrchk(cudaMemcpy(d_sum_pos,   &zero_d, sizeof(double), cudaMemcpyHostToDevice));
	        gpuErrchk(cudaMemcpy(d_count_pos, &zero_i, sizeof(int),    cudaMemcpyHostToDevice));

	        computeDeltaE_uphill_accum<<<num_spins, THREADS>>>(
	            gpu_row_ptr, gpu_col_idx, gpu_J_values,
	            gpuLinTermsVect, gpu_spins_old, d_sum_pos, d_count_pos);
	        cudaDeviceSynchronize();

	        double h_sum; int h_cnt;
	        gpuErrchk(cudaMemcpy(&h_sum, d_sum_pos,   sizeof(double), cudaMemcpyDeviceToHost));
	        gpuErrchk(cudaMemcpy(&h_cnt, d_count_pos, sizeof(int),    cudaMemcpyDeviceToHost));
	        total_sum += h_sum;
	        total_cnt += h_cnt;
	    }

	    cudaFree(d_sum_pos);
	    cudaFree(d_count_pos);

	    double mean_dEp = (total_cnt > 0) ? (total_sum / (double)total_cnt) : 1.0;
	    double T0       = mean_dEp / std::log(1.0 / auto_accept_rate);
	    std::cout << "Auto start-temp estimation (Ben-Ameur simple form):\n"
	              << "  N_config = " << auto_n_config
	              << ", uphill samples = " << total_cnt << "\n"
	              << "  mean |dE+|          = " << mean_dEp << "\n"
	              << "  target accept rate  = " << auto_accept_rate << "\n"
	              << "  ==> initial temp T0 = " << T0 << std::endl;
	    start_temp = T0;

	    // Recompute initial energy for the last random config (which is also
	    // the state annealing will start from).
	    {
	        double zero = 0.0;
	        gpuErrchk(cudaMemcpy(d_total_energy, &zero, sizeof(double), cudaMemcpyHostToDevice));
	    }
	    compute_total_energy<<<num_spins, THREADS>>>(
	        gpu_row_ptr, gpu_col_idx, gpu_J_values,
	        gpuLinTermsVect, gpu_spins_old, gpu_num_spins, d_total_energy);
	    cudaDeviceSynchronize();
	    gpuErrchk(cudaMemcpy(gpu_total_energy, d_total_energy,
	                         sizeof(double), cudaMemcpyDeviceToHost));
	    gpu_best_energy[0] = gpu_total_energy[0];
	    gpuErrchk(cudaMemcpy(gpu_spins_best, gpu_spins_old,
	                         num_spins * sizeof(signed char),
	                         cudaMemcpyDeviceToDevice));
	}

	auto t_setup_end = std::chrono::high_resolution_clock::now();
	double t_setup = (double)std::chrono::duration_cast<std::chrono::microseconds>(t_setup_end - t_setup_start).count() * 1e-6;

	std::cout << "start annealing with initial energy: " << gpu_best_energy[0]
	          << "  (start_temp = " << start_temp << ")" << std::endl;
	std::vector<double> beta_schedule = create_beta_schedule_geometric(start_temp, stop_temp, alpha);
	std::cout << "Beta schedule length (derived): " << beta_schedule.size() << std::endl;

	auto t0 = std::chrono::high_resolution_clock::now();
	auto annealing_start = std::chrono::high_resolution_clock::now();

	// Temperature loop
	for (int i = 0; i < (int)beta_schedule.size(); i++) {

		for (int ii = 0; ii < (int)num_sweeps_per_beta; ii++) {

		    // Fresh randoms for this full-sweep's Metropolis tests.
		    curandGenerateUniform(rng, gpu_randvals, num_spins);

		    // One parallel Metropolis pass per color class.
		    for (int c = 0; c < num_colors; c++) {
		        int dense_begin  = ct.color_dense_offsets[c];
		        int dense_end    = ct.color_dense_offsets[c + 1];
		        int nd           = dense_end - dense_begin;

		        int sparse_begin = ct.color_sparse_offsets[c];
		        int sparse_end   = ct.color_sparse_offsets[c + 1];
		        int ns           = sparse_end - sparse_begin;

		        if (nd == 0 && ns == 0) continue;

		        gpuErrchk(cudaMemsetAsync(gpu_num_candidates, 0, sizeof(int)));

		        if (nd > 0) {
		            collectFlipCandidates_dense<<<nd, THREADS>>>(
		                gpu_row_ptr,
		                gpu_col_idx,
		                gpu_J_values,
		                gpuLinTermsVect,
		                gpu_randvals,
		                gpu_spins_old,
		                beta_schedule[i],
		                gpu_candidates,
		                gpu_num_candidates,
		                gpu_color_dense_flat + dense_begin
		            );
		        }

		        if (ns > 0) {
		            int sparse_blocks = (ns + SPINS_PER_BLOCK_SPARSE - 1) / SPINS_PER_BLOCK_SPARSE;
		            collectFlipCandidates_sparse<<<sparse_blocks, THREADS>>>(
		                gpu_sparse_row_ptr,
		                gpu_sparse_col_idx_tagged,
		                gpu_sparse_J_values,
		                gpu_sparse_global_id,
		                gpuLinTermsVect,
		                gpu_randvals,
		                gpu_spins_old,
		                gpu_hub_vals,
		                h_num_hub,
		                beta_schedule[i],
		                gpu_candidates,
		                gpu_num_candidates,
		                gpu_color_sparse_flat + sparse_begin,
		                ns
		            );
		        }

		        // Apply every accepted flip from this color in parallel.
		        // Worst-case num_candidates = nd + ns, so launch that many threads.
		        int color_size = nd + ns;
		        int apply_blocks = (color_size + 255) / 256;
		        applyAllFlipsInColor<<<apply_blocks, 256>>>(
		            gpu_spins_old, d_total_energy, gpu_candidates, gpu_num_candidates);

		        // If any hub could have been flipped this color, refresh cache.
		        if (nd > 0) {
		            int hb = (h_num_hub + 63) / 64;
		            refreshHubVals<<<hb, 64>>>(
		                gpu_spins_old, gpu_hub_ids, gpu_hub_vals, h_num_hub);
		        }
		    }
		}

		// One host sync per temperature iteration.
		cudaDeviceSynchronize();
		gpuErrchk(cudaMemcpy(gpu_total_energy, d_total_energy,
		                     sizeof(double), cudaMemcpyDeviceToHost));
		// Whenever the running accumulator reports a new best, snapshot the
		// live spins into gpu_spins_best. The D2D copy runs on the default
		// stream so it serializes after the annealing kernels of this temp
		// step and before the next one starts.
		if (gpu_total_energy[0] < gpu_best_energy[0]) {
			gpu_best_energy[0] = gpu_total_energy[0];
			gpuErrchk(cudaMemcpyAsync(gpu_spins_best, gpu_spins_old,
			                          num_spins * sizeof(signed char),
			                          cudaMemcpyDeviceToDevice));
		}
		energy_history.push_back(gpu_best_energy[0]);
	}
 
	auto t1 = std::chrono::high_resolution_clock::now();

	double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

	auto annealing_end = std::chrono::high_resolution_clock::now();
	double t_anneal = (double)std::chrono::duration_cast<std::chrono::microseconds>(annealing_end - annealing_start).count() * 1e-6;
	printf("Total annealing time: %.6f seconds\n", t_anneal);

	// Write energy history to file
	std::string energy_filename = output_dir + "energy_history_" + run_suffix;

	FILE* energy_fptr = fopen(energy_filename.c_str(), "w");
	fprintf(energy_fptr, "# Iteration\tBest_Energy\n");
	for (int i = 0; i < (int)energy_history.size(); i++){
	    fprintf(energy_fptr, "%d\t%.6f\n", i, energy_history[i]);
	}
	fclose(energy_fptr);

    // Heap-allocated to avoid stack overflow at large N (was a VLA on the stack).
    std::vector<signed char> cpu_spins(num_spins);

	// Snapshot the running (atomicAdd-tracked) energy. We recompute energy
	// from scratch for two states:
	//   (a) gpu_spins_old   — the live state at the end of annealing; used
	//                         only to validate the running accumulator (drift
	//                         diagnostic).
	//   (b) gpu_spins_best  — the lowest-running-energy state we observed;
	//                         this is what we report and write to the spin file.
	double running_energy = gpu_total_energy[0];

	// (a) Recompute energy of the final running state for drift diagnostic.
	double final_state_energy;
	{
		double zero = 0.0;
		gpuErrchk(cudaMemcpy(d_total_energy, &zero, sizeof(double), cudaMemcpyHostToDevice));
		compute_total_energy<<<num_spins, THREADS>>>(
			gpu_row_ptr, gpu_col_idx, gpu_J_values,
			gpuLinTermsVect, gpu_spins_old, gpu_num_spins, d_total_energy);
		cudaDeviceSynchronize();
		gpuErrchk(cudaMemcpy(&final_state_energy, d_total_energy,
		                     sizeof(double), cudaMemcpyDeviceToHost));
	}

	// (b) Recompute energy of the best state and copy it back to host.
	double best_state_energy;
	{
		double zero = 0.0;
		gpuErrchk(cudaMemcpy(d_total_energy, &zero, sizeof(double), cudaMemcpyHostToDevice));
		compute_total_energy<<<num_spins, THREADS>>>(
			gpu_row_ptr, gpu_col_idx, gpu_J_values,
			gpuLinTermsVect, gpu_spins_best, gpu_num_spins, d_total_energy);
		cudaDeviceSynchronize();
		gpuErrchk(cudaMemcpy(&best_state_energy, d_total_energy,
		                     sizeof(double), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(cpu_spins.data(), gpu_spins_best,
		                     num_spins * sizeof(signed char),
		                     cudaMemcpyDeviceToHost));
	}

	{
		std::string spins_filename = output_dir + "spins_" + run_suffix;

		FILE* fptr1 = fopen(spins_filename.c_str() , "w");
		for(int i = 0; i < num_spins; i++){
			fprintf(fptr1, "%d\t",  (int)cpu_spins[i]);
		}
		fprintf(fptr1,"\n");
		fclose(fptr1);
	}

	printf("\t final-state energy (recomputed): %.6f\n", final_state_energy);
	printf("\t best-state  energy (recomputed): %.6f\n", best_state_energy);
	printf("best engy (running tracker): %.6f\n", gpu_best_energy[0]);

	{
		double diff = running_energy - final_state_energy;
		double denom = std::max(1.0, std::fabs(final_state_energy));
		printf("energy check: running=%.6f recomputed=%.6f"
		       " diff=%.6f (rel=%.3e)\n",
		       running_energy, final_state_energy, diff,
		       diff / denom);
	}

	auto t_total_end = std::chrono::high_resolution_clock::now();
	double t_total = (double)std::chrono::duration_cast<std::chrono::microseconds>(t_total_end - t_total_start).count() * 1e-6;

	printf("\n--- Timing ---\n");
	printf("  Load data:   %.6f s\n", t_load);
	printf("  Setup (GPU): %.6f s\n", t_setup);
	printf("  Annealing:   %.6f s\n", t_anneal);
	printf("  Total:       %.6f s\n", t_total);

	// --------------------------------------------------
	// Free host-pinned memory
	// --------------------------------------------------
	cudaFreeHost(gpu_total_energy);
	cudaFreeHost(gpu_best_energy);
	
	// --------------------------------------------------
	// Free device memory
	// --------------------------------------------------
	cudaFree(gpu_row_ptr);
	cudaFree(gpu_col_idx);
	cudaFree(gpu_J_values);
	cudaFree(gpu_num_spins);
	cudaFree(gpu_spins_old);
	cudaFree(gpu_spins_best);
	cudaFree(d_total_energy);
	cudaFree(devRanStates);
	cudaFree(gpu_candidates);
	cudaFree(gpu_num_candidates);
	cudaFree(gpu_color_dense_flat);
	cudaFree(gpu_color_dense_offsets);
	cudaFree(gpu_color_sparse_flat);
	cudaFree(gpu_color_sparse_offsets);
	cudaFree(gpu_sparse_row_ptr);
	cudaFree(gpu_sparse_col_idx_tagged);
	cudaFree(gpu_sparse_J_values);
	cudaFree(gpu_sparse_global_id);
	cudaFree(gpu_hub_ids);
	cudaFree(gpu_hub_vals);

	return 0;
}


#define CORRECT 1

#if CORRECT

__global__ void collectFlipCandidates_dense(
        const int*     row_ptr,
        const int*     col_idx,
        const double*  J_values,
        const double*  gpuLinTermsVect,
        const float* __restrict__ randvals,
        signed char*   gpuLatSpin,
        const double   beta,
        FlipCandidate* candidates,
        int*           num_candidates,
        const int*     dense_spin_ids   // maps blockIdx.x → global spin id
){
    int vertice_Id = dense_spin_ids[blockIdx.x];   // global spin index
    int p_Id       = threadIdx.x;

    __shared__ double sh_mem[THREADS];
    sh_mem[p_Id] = 0.0;
    __syncthreads();

    int current_spin = (int)gpuLatSpin[vertice_Id];

    int start = row_ptr[vertice_Id];
    int end   = row_ptr[vertice_Id + 1];

    for (int k = start + p_Id; k < end; k += blockDim.x)
        sh_mem[p_Id] += J_values[k] * (double)gpuLatSpin[col_idx[k]];
    __syncthreads();

    // Standard shared-memory tree reduction
    for (int off = blockDim.x / 2; off; off /= 2) {
        if (p_Id < off) sh_mem[p_Id] += sh_mem[p_Id + off];
        __syncthreads();
    }

    if (p_Id == 0) {
        double dE = -2.0 * (double)current_spin
                    * (sh_mem[0] + gpuLinTermsVect[vertice_Id]);
        float acceptance = fminf(1.0f, (float)exp(-beta * dE));

        if (randvals[vertice_Id] < acceptance) {
            int idx = atomicAdd(num_candidates, 1);
            candidates[idx].spin_id      = vertice_Id;
            candidates[idx].delta_energy = dE;
        }
    }
}

#endif

// Sparse-bin Metropolis candidate collection for a single color class.
//
// Layout: SPINS_PER_BLOCK_SPARSE warps per block, one warp per sparse spin.
// Hub values for the whole problem are copied into shared memory at block
// start (≤256 bytes total) so the inner neighbor-sum loop becomes a shared
// load for the hot "neighbor is a hub" case and a global load for the rare
// fallback. Neighbor tagging is resolved branchlessly via the sign bit.
__global__ void collectFlipCandidates_sparse(
        const int*          sparse_row_ptr,
        const int*          sparse_col_idx_tagged,
        const double*       sparse_J_values,
        const int*          sparse_global_id,
        const double*       gpuLinTermsVect,
        const float* __restrict__ randvals,
        signed char*        gpuLatSpin,
        const signed char*  d_hub_vals,
        int                 num_hubs,
        const double        beta,
        FlipCandidate*      candidates,
        int*                num_candidates,
        const int*          color_sparse_csr_ids,
        int                 num_in_color
){
    __shared__ signed char s_hub_vals[MAX_HUB_SPINS];

    // Cooperative load of hub cache (≤ MAX_HUB_SPINS bytes).
    for (int t = threadIdx.x; t < num_hubs; t += blockDim.x)
        s_hub_vals[t] = d_hub_vals[t];
    __syncthreads();

    int warp_in_block = threadIdx.x / 32;
    int lane          = threadIdx.x & 31;

    int slot = (int)blockIdx.x * SPINS_PER_BLOCK_SPARSE + warp_in_block;
    if (slot >= num_in_color) return;

    int sparse_idx = color_sparse_csr_ids[slot];       // index into sparse CSR
    int vertice_Id = sparse_global_id[sparse_idx];     // global spin id

    int current_spin = (int)gpuLatSpin[vertice_Id];

    int start = sparse_row_ptr[sparse_idx];
    int end   = sparse_row_ptr[sparse_idx + 1];
    int len   = end - start;

    double local_sum = 0.0;
    for (int k = lane; k < len; k += 32) {
        int e = sparse_col_idx_tagged[start + k];
        int neighbor_spin;
        if (e < 0) {
            // Hot path: high bit set => hub index into shared cache.
            neighbor_spin = (int)s_hub_vals[e & HUB_IDX_MASK];
        } else {
            // Rare fallback: non-hub neighbor, read straight from global.
            neighbor_spin = (int)gpuLatSpin[e];
        }
        local_sum += sparse_J_values[start + k] * (double)neighbor_spin;
    }

    // Warp-shuffle reduction (double).
    unsigned mask = 0xFFFFFFFFu;
    local_sum += __shfl_down_sync(mask, local_sum, 16);
    local_sum += __shfl_down_sync(mask, local_sum,  8);
    local_sum += __shfl_down_sync(mask, local_sum,  4);
    local_sum += __shfl_down_sync(mask, local_sum,  2);
    local_sum += __shfl_down_sync(mask, local_sum,  1);

    if (lane == 0) {
        double dE = -2.0 * (double)current_spin
                    * (local_sum + gpuLinTermsVect[vertice_Id]);
        float acceptance = fminf(1.0f, (float)exp(-beta * dE));
        if (randvals[vertice_Id] < acceptance) {
            int idx = atomicAdd(num_candidates, 1);
            candidates[idx].spin_id      = vertice_Id;
            candidates[idx].delta_energy = dE;
        }
    }
}


// Apply every accepted flip collected during a single color pass.
// Because all candidates come from one color class, no two share an edge,
// so parallel flips cannot invalidate each other's dE — and the running
// total-energy accumulator stays exact.
__global__ void applyAllFlipsInColor(
        signed char*         gpuLatSpin,
        double*              d_total_energy,
        const FlipCandidate* candidates,
        const int*           num_candidates
){
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int n = *num_candidates;
    if (t >= n) return;
    int    sid = candidates[t].spin_id;
    double dE  = candidates[t].delta_energy;
    gpuLatSpin[sid] = -gpuLatSpin[sid];
    atomicAdd(d_total_energy, dE);
}


// Refresh the hub-values cache in global memory from the live spin array.
// Runs after any color pass that may have flipped a hub.
__global__ void refreshHubVals(
        const signed char* gpuLatSpin,
        const int*         d_hub_ids,
        signed char*       d_hub_vals,
        int                num_hubs
){
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_hubs)
        d_hub_vals[t] = gpuLatSpin[d_hub_ids[t]];
}

// Spin-only init. Must complete (kernel boundary) before any energy
// kernel reads neighbors, otherwise gpuSpins[col_idx[k]] races with the
// neighbor block's own spin write inside a fused init+energy kernel.
__global__ void init_spins_only(
        const float* __restrict__ randvals,
        signed char*              gpuSpins,
        curandState*              state,
        unsigned long             seed,
        int                       num_spins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_spins) return;
    gpuSpins[i] = (randvals[i] < 0.5f) ? -1 : 1;
    curand_init(seed, i, 0, &state[i]);
}

// Compute total energy E = ½ s'Js + h's from the current spin state.
// Each block handles one spin: computes field_i = Σ_j J_ij s_j via shared
// reduction, then atomicAdds ½*s_i*field_i + s_i*h_i into total_energy.
__global__ void compute_total_energy(
		const int*           row_ptr,
	    const int*           col_idx,
	    const double*        J_values,
		const double*        gpuLinTermsVect,
		signed char*         gpuSpins,
		const unsigned int*  gpu_num_spins,
		double*              total_energy){

	unsigned int vertice_Id = blockIdx.x;
	unsigned int p_Id = threadIdx.x;

	__shared__ double sh_mem[THREADS];
	sh_mem[p_Id] = 0.0;
	__syncthreads();

	int start = row_ptr[vertice_Id];
	int end   = row_ptr[vertice_Id + 1];

	for (int k = start + p_Id; k < end; k += blockDim.x){
	    sh_mem[p_Id] += J_values[k] * (double)gpuSpins[col_idx[k]];
	}
	__syncthreads();

	for (int off = blockDim.x / 2; off; off /= 2){
		if (threadIdx.x < off){
			sh_mem[threadIdx.x] += sh_mem[threadIdx.x + off];
		}
		__syncthreads();
	}

	if (p_Id == 0){
		double s_i = (double)gpuSpins[vertice_Id];
		// E_i = ½ * s_i * field_i + s_i * h_i
		double vertice_energy = 0.5 * s_i * sh_mem[0]
		                      + s_i * gpuLinTermsVect[vertice_Id];
		atomicAdd(total_energy, vertice_energy);
	}
}

__global__ void computeDeltaE_uphill_accum(
        const int*           row_ptr,
        const int*           col_idx,
        const double*        J_values,
        const double*        gpuLinTermsVect,
        const signed char*   gpuSpins,
        double*              d_sum_pos,
        int*                 d_count_pos)
{
    unsigned int vertice_Id = blockIdx.x;
    unsigned int p_Id       = threadIdx.x;

    __shared__ double sh_mem[THREADS];
    sh_mem[p_Id] = 0.0;
    __syncthreads();

    int start = row_ptr[vertice_Id];
    int end   = row_ptr[vertice_Id + 1];
    for (int k = start + p_Id; k < end; k += blockDim.x) {
        sh_mem[p_Id] += J_values[k] * (double)gpuSpins[col_idx[k]];
    }
    __syncthreads();

    for (int off = blockDim.x / 2; off; off /= 2) {
        if (p_Id < off) sh_mem[p_Id] += sh_mem[p_Id + off];
        __syncthreads();
    }

    if (p_Id == 0) {
        double dE = -2.0 * (double)gpuSpins[vertice_Id]
                    * (sh_mem[0] + gpuLinTermsVect[vertice_Id]);
        if (dE > 0.0) {
            atomicAdd(d_sum_pos, dE);
            atomicAdd(d_count_pos, 1);
        }
    }
}

std::vector<double> create_beta_schedule_geometric(
		double temp_start,
		double temp_end,
		double alpha){

	std::vector<double> beta_schedule;
	double current_temp = temp_start;

	// Number of temperature steps is fully determined by start/stop/alpha:
	// N such that temp_start * alpha^N <= temp_end  =>  N = ceil(log(temp_end/temp_start)/log(alpha)).
	uint32_t num_sweeps = (uint32_t)std::ceil(std::log(temp_end / temp_start) / std::log(alpha));

	for (uint32_t i = 0; i < num_sweeps; i++){

		beta_schedule.push_back(1.0 / current_temp);
		current_temp *= alpha;
		
		if (current_temp < temp_end){
			current_temp = temp_end;  // Floor at minimum temperature
		}
	}

	return beta_schedule;
}
