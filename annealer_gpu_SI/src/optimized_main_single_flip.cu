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

#define THREADS 1024 //or more threads gpu crashes
#define BREAK_UPDATE_VAL 2//1000
#define TCRIT 2.26918531421f

struct FlipCandidate {
    int    spin_id;
    float  delta_energy;
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

#define CHANGE_MAX_ENERGY 0.0f
#define BREAK_AFTER_ITERATION 1.0f
//__constant__ float kd_floats[1000000];


// float atomicMin
__device__ __forceinline__ float mAtomicMin(float *address, float val)
{
	int ret = __float_as_int(*address);
	while (val < __int_as_float(ret))
	{
		int old = ret;
		if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
			break;
	}
	return __int_as_float(ret);
}


__device__ __forceinline__ float mAtomicMax(float *address, float val)
{
	int ret = __float_as_int(*address);
	while (val > __int_as_float(ret))
	{
		int old = ret;
		if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
			break;
	}
	return __int_as_float(ret);
}


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
    std::vector<int>   row_ptr;           // size num_sparse+1
    std::vector<int>   col_idx_tagged;    // tagged per HUB_TAG_BIT
    std::vector<float> J_values;
    std::vector<int>   global_id;         // size num_sparse; sparse-idx -> global spin id
    int                num_sparse = 0;
    int                nnz_hub_fallback = 0;
};

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

    // Sanity: no two adjacent spins share a color.
    for (int i = 0; i < num_spins; i++) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
            int j = col_idx[k];
            if (i == j) continue;
            assert(color_of[i] != color_of[j] && "coloring broken: adjacent spins share a color");
        }
    }

    num_colors_out = num_colors;
    return color_of;
}

// Build the sparse-only CSR with hub-tagged col_idx.
static SparseCSR buildSparseCSR(
    const std::vector<int>&   row_ptr,
    const std::vector<int>&   col_idx,
    const std::vector<float>& J_values,
    const std::vector<int>&   dense_spins,
    const std::vector<int>&   sparse_spins,
    int                       num_spins)
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
            int nb = col_idx[k];
            float J = J_values[k];
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

__global__ void init_best_energy(float* total_energy, float* best_energy, bool init = false)
{
	if (init)
	{
		best_energy[0] = total_energy[0];
		printf("initial energy %.6f \n", total_energy[0]);
	}
	else
	{
		mAtomicMin(best_energy, total_energy[0]);
		printf(" best_energy, total_energy %.6f %.6f \n", best_energy[0], total_energy[0]);
	}
}


// Initialize lattice spins
__global__ void init_spins_total_energy(
		const int* row_ptr, 
		const int* col_idx, 
		const float* J_values,
		float* gpuLinTermsVect,
		const float* __restrict__ randvals,
		signed char* gpuSpins,
		const unsigned int* gpu_num_spins,
		float* total_energy,
		curandState * state,
		unsigned long seed
);


// Final lattice spins
__global__ void final_spins_total_energy(
		const int* row_ptr, 
		const int* col_idx, 
		const float* J_values,
		float* gpuLinTermsVect,
		signed char* gpuSpins,
		const unsigned int* gpu_num_spins,
		float* total_energy
);


__global__ void collectFlipCandidates_dense(
        const int*     row_ptr,
        const int*     col_idx,
        const float*   J_values,
        float*         gpuLinTermsVect,
        const float* __restrict__ randvals,
        signed char*   gpuLatSpin,
        const float    beta,
        FlipCandidate* candidates,
        int*           num_candidates,
        const int*     dense_spin_ids   // maps blockIdx.x → global spin id (for this color)
);


__global__ void collectFlipCandidates_sparse(
        const int*          sparse_row_ptr,
        const int*          sparse_col_idx_tagged,
        const float*        sparse_J_values,
        const int*          sparse_global_id,
        const float*        gpuLinTermsVect,
        const float* __restrict__ randvals,
        signed char*        gpuLatSpin,
        const signed char*  d_hub_vals,
        int                 num_hubs,
        const float         beta,
        FlipCandidate*      candidates,
        int*                num_candidates,
        const int*          color_sparse_csr_ids,   // sparse-CSR indices for this color
        int                 num_in_color
);


__global__ void applyAllFlipsInColor(
        signed char*         gpuLatSpin,
        float*               d_total_energy,
        const FlipCandidate* candidates,
        const int*           num_candidates
);


__global__ void refreshHubVals(
        const signed char* gpuLatSpin,
        const int*         d_hub_ids,
        signed char*       d_hub_vals,
        int                num_hubs
);

std::vector<double> create_beta_schedule_geometric(uint32_t num_sweeps, double temp_start, double temp_end, double alpha);
  
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
		"\t-n|--niters <INT>\n"
		"\t\tnumber of iterations\n"
		"\n"
		"\t-c|--alpha <FLOAT>\n"
		"\t\tcooling rate (temperature multiplier, 0 < alpha < 1, default: 0.95)"
		"\n"
		"\t-n|--sweeps_per_beta <INT>\n"
		"\t\tnumber of sweep per temperature\n"
		"\n"
		"\t-s|--seed <SEED>\n"
		"\t\tfix the starting point\n"
		"\n"
		"\t-s|--debug \n"
		"\t\t Print the final lattice value at every temperature\n"
		"\n"
		"\t-e|--no-early-stop\n"
		"\t\tDisable early stopping - run full annealing schedule\n"
		"\n"
		"\t-o|--write-lattice\n"
		"\t\twrite final lattice configuration to file\n\n",
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
	
	float start_temp = 20.f;
	float stop_temp = 0.001f;
	float alpha = 0.95f;
	unsigned long long seed = ((getpid()* rand()) & 0x7FFFFFFFF); // ((GetCurrentProcessId()* rand()) & 0x7FFFFFFFF);
	
	unsigned int num_temps = 1000; //atoi(argv[2]);
	unsigned int num_sweeps_per_beta = 1;//atoi(argv[3]);
	
	// bool do_write = false;
	bool debug = false;
	bool disable_early_stop = false;
	
	std::vector<float> energy_history;  // Store best energy at each iteration
	
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
			{        "niters", required_argument, 0, 'n'},
			{ "sweeps_per_beta", required_argument, 0, 'm'},
			{ "write-lattice",       no_argument, 0, 'o'},
			{          "debug",       no_argument, 0, 'd'},
			{ "no-early-stop",       no_argument, 0, 'e'},
			{          "help",       no_argument, 0, 'h'},
			{               0,                 0, 0,   0}
		};

		int option_index = 0;
		int ch = getopt_long(argc, argv, "R:C:V:l:x:y:s:c:n:m:odeh", long_options, &option_index);
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
			start_temp = atof(optarg); break;
		case 'y':
			stop_temp = atof(optarg); break;
		case 's':
			seed = atoll(optarg);
			break;
		case 'c': 
			alpha = atof(optarg); 
			break;
		case 'n':
			num_temps = atoi(optarg); break;
		case 'm':
			num_sweeps_per_beta = atoi(optarg); break;
        //does not seem to be used
		//case 'o':
			//do_write = true; break;
 		case 'd':
			debug = true; break;
		case 'e':
		    disable_early_stop = true; break;
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
			<< " start temp " << start_temp << " stop temp " << stop_temp << "\n" 
			<< " seed " << seed << " num temp " << num_temps << " num sweeps " 
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

	std::vector<float> linearTermsVect;
	//if (linear_file.empty() == false)
    readLinearValues(linear_file, num_spins, linearTermsVect);

	auto t_load_end = std::chrono::high_resolution_clock::now();
	double t_load = (double)std::chrono::duration_cast<std::chrono::microseconds>(t_load_end - t_load_start).count() * 1e-6;

	double endtime = rtclock();

    if(debug)
  	  printtime("ParseData time: ", starttime, endtime);

	unsigned int nnz = parseSparse.getNNZ();
	
	const std::vector<int>&row_ptr = parseSparse.getRowPtr();
	const std::vector<int>&col_idx = parseSparse.getColIdx();
	const std::vector<float>&J_values = parseSparse.getValues();
	
	std::cout << "Sparse J loaded: num_spins = "
	          << num_spins << ", nnz = " << nnz << std::endl;

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
	int   *gpu_sparse_row_ptr = nullptr;
	int   *gpu_sparse_col_idx_tagged = nullptr;
	float *gpu_sparse_J_values = nullptr;
	int   *gpu_sparse_global_id = nullptr;
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
	                         scsr.J_values.size() * sizeof(float)));
	    gpuErrchk(cudaMemcpy(gpu_sparse_J_values, scsr.J_values.data(),
	                         scsr.J_values.size() * sizeof(float),
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

	float *gpuLinTermsVect;
	gpuErrchk(cudaMalloc((void**)&gpuLinTermsVect, (num_spins) * sizeof(float)));

	if (linearTermsVect.size() != num_spins)
		std::cout << "	[ERROR] error in parsing the linear terms from file" << std::endl;

	gpuErrchk(cudaMemcpy(gpuLinTermsVect, linearTermsVect.data(), (num_spins) * sizeof(float), cudaMemcpyHostToDevice));

	// ---- Allocate sparse J (CSR format) on GPU ----

	int* gpu_row_ptr = nullptr;
	int* gpu_col_idx = nullptr;
	float* gpu_J_values = nullptr;

	starttime = rtclock();
	
	// row_ptr
	gpuErrchk(cudaMalloc((void**)&gpu_row_ptr, (num_spins + 1) * sizeof(int)));
	
	gpuErrchk(cudaMemcpy(gpu_row_ptr, row_ptr.data(), (num_spins + 1) * sizeof(int), cudaMemcpyHostToDevice));
	
	// col_idx
	gpuErrchk(cudaMalloc((void**)&gpu_col_idx, nnz * sizeof(int)));
	
	gpuErrchk(cudaMemcpy(gpu_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
	
	// J values
	gpuErrchk(cudaMalloc((void**)&gpu_J_values, nnz * sizeof(float)));
	
	gpuErrchk(cudaMemcpy(gpu_J_values, J_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

	endtime = rtclock();
   
	if(debug)
		 printtime("J Matrix data transfer time: ", starttime, endtime);

	unsigned int* gpu_num_spins;
	gpuErrchk(cudaMalloc((void**)&gpu_num_spins, sizeof(*gpu_num_spins)));
	gpuErrchk(cudaMemcpy(gpu_num_spins, &num_spins, sizeof(*gpu_num_spins), cudaMemcpyHostToDevice));
	
	float* gpu_total_energy;
	cudaHostAlloc(&gpu_total_energy, sizeof(float), 0);

	float* d_total_energy;
	gpuErrchk(cudaMalloc((void**)&d_total_energy, sizeof(float)));

	float* gpu_best_energy;
	cudaHostAlloc(&gpu_best_energy, sizeof(float), 0);
 
	signed char *gpu_spins_old;
	
	gpuErrchk(cudaMalloc((void**)&gpu_spins_old, num_spins * sizeof(signed char)));

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

	gpuErrchk(cudaMemset(d_total_energy, 0, sizeof(float)));

	init_spins_total_energy << < num_spins, THREADS >> > (gpu_row_ptr,
    	gpu_col_idx,
    	gpu_J_values,
		gpuLinTermsVect,
		gpu_randvals,
		gpu_spins_old,
		gpu_num_spins,
		d_total_energy,
		devRanStates,
		(unsigned long)t
	);
  
  	cudaDeviceSynchronize();

	gpuErrchk(cudaMemcpy(gpu_total_energy, d_total_energy, sizeof(float), cudaMemcpyDeviceToHost));
      
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

	// d_total_energy already holds the correct initial value from init_spins_total_energy.
	// No reset needed here — applyAllFlipsInColor atomicAdds dE onto it incrementally.
	// We just confirm the device value is in sync before the loop starts.
	gpuErrchk(cudaMemcpy(d_total_energy, gpu_total_energy, sizeof(float), cudaMemcpyHostToDevice));

	auto t_setup_end = std::chrono::high_resolution_clock::now();
	double t_setup = (double)std::chrono::duration_cast<std::chrono::microseconds>(t_setup_end - t_setup_start).count() * 1e-6;

	std::cout << "start annealing with initial energy: " << gpu_best_energy[0] << std::endl;
	std::vector<double> beta_schedule = create_beta_schedule_geometric(num_temps, start_temp, stop_temp, alpha);

	auto t0 = std::chrono::high_resolution_clock::now();
	auto annealing_start = std::chrono::high_resolution_clock::now(); 

	// Previous-iteration best energy used for coarse early-stop tracking.
	float prev_best = gpu_best_energy[0];
	int no_update = 0;

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
		                (float)beta_schedule[i],
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
		                (float)beta_schedule[i],
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
		                     sizeof(float), cudaMemcpyDeviceToHost));
		gpu_best_energy[0] = std::min(gpu_total_energy[0], gpu_best_energy[0]);
		energy_history.push_back(gpu_best_energy[0]);

		// Coarse early stopping: count temperatures with no best-energy improvement.
		if (gpu_best_energy[0] < prev_best - CHANGE_MAX_ENERGY) {
		    no_update = 0;
		    prev_best = gpu_best_energy[0];
		} else {
		    no_update++;
		}
		if (!disable_early_stop && no_update > (int)BREAK_AFTER_ITERATION * 10) {
		    printf("Breaking early at temperature iteration %d\n", i);
		    break;
		}
	}
 
	auto t1 = std::chrono::high_resolution_clock::now();

	double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

	auto annealing_end = std::chrono::high_resolution_clock::now();
	double t_anneal = (double)std::chrono::duration_cast<std::chrono::microseconds>(annealing_end - annealing_start).count() * 1e-6;
	printf("Total annealing time: %.6f seconds\n", t_anneal);

	// Write energy history to file for plotting
	std::string energy_filename = "energy_history_" + run_suffix;
	
	FILE* energy_fptr = fopen(energy_filename.c_str(), "w");
	fprintf(energy_fptr, "# Iteration\tBest_Energy\n");
	for (int i = 0; i < energy_history.size(); i++){
	    fprintf(energy_fptr, "%d\t%.6f\n", i, energy_history[i]);
	}
	fclose(energy_fptr);
	// printf("Energy history written to: %s\n", energy_filename.c_str());

    signed char cpu_spins[num_spins];

	gpuErrchk(cudaMemset(d_total_energy, 0, sizeof(float)));
	{	
        final_spins_total_energy << < num_spins, THREADS >> > (gpu_row_ptr,
				gpu_col_idx,
				gpu_J_values,
				gpuLinTermsVect,
				gpu_spins_old,
				gpu_num_spins,
				d_total_energy
		);

  		cudaDeviceSynchronize();

	    gpuErrchk(cudaMemcpy(gpu_total_energy, d_total_energy, sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(cpu_spins, gpu_spins_old, num_spins * sizeof(signed char), cudaMemcpyDeviceToHost));
	}     
        			
	if(debug){
		std::string spins_filename = "spins_" + run_suffix;
		
		FILE* fptr1 = fopen(spins_filename.c_str() , "w");
		for(int i = 0; i < num_spins; i++){
			fprintf(fptr1, "%d\t",  (int)cpu_spins[i]);
		}  

		fprintf(fptr1,"\n\n\n");
		// fprintf(fptr1,"\tbest energy value: %.6f\n", gpu_best_energy[0] );
		fprintf(fptr1,"\ttotal energy value: %.6f\n", gpu_total_energy[0] );
		// fprintf(fptr1," \t elapsed time in sec: %.6f\n", duration * 1e-6 );
		fclose(fptr1);
	}

	std::cout << "\t total energy value: " << gpu_total_energy[0] << std::endl;
	printf("best engy %.1f \n", gpu_best_energy[0]);

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
        const float*   J_values,
        float*         gpuLinTermsVect,
        const float* __restrict__ randvals,
        signed char*   gpuLatSpin,
        const float    beta,
        FlipCandidate* candidates,
        int*           num_candidates,
        const int*     dense_spin_ids   // maps blockIdx.x → global spin id
){
    int vertice_Id = dense_spin_ids[blockIdx.x];   // global spin index
    int p_Id       = threadIdx.x;
 
    __shared__ float sh_mem[THREADS];
    sh_mem[p_Id] = 0.0f;
    __syncthreads();
 
    float current_spin = (float)gpuLatSpin[vertice_Id];
 
    int start = row_ptr[vertice_Id];
    int end   = row_ptr[vertice_Id + 1];
 
    for (int k = start + p_Id; k < end; k += blockDim.x)
        sh_mem[p_Id] += J_values[k] * (float)gpuLatSpin[col_idx[k]];
    __syncthreads();
 
    // Standard shared-memory tree reduction
    for (int off = blockDim.x / 2; off; off /= 2) {
        if (p_Id < off) sh_mem[p_Id] += sh_mem[p_Id + off];
        __syncthreads();
    }
 
    if (p_Id == 0) {
        float dE = -2.0f * (sh_mem[0] + gpuLinTermsVect[vertice_Id]) * current_spin;
        float acceptance = fminf(1.0f, expf(-beta * dE));

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
        const float*        sparse_J_values,
        const int*          sparse_global_id,
        const float*        gpuLinTermsVect,
        const float* __restrict__ randvals,
        signed char*        gpuLatSpin,
        const signed char*  d_hub_vals,
        int                 num_hubs,
        const float         beta,
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

    float current_spin = (float)gpuLatSpin[vertice_Id];

    int start = sparse_row_ptr[sparse_idx];
    int end   = sparse_row_ptr[sparse_idx + 1];
    int len   = end - start;

    float local_sum = 0.0f;
    for (int k = lane; k < len; k += 32) {
        int e = sparse_col_idx_tagged[start + k];
        float neighbor_spin;
        if (e < 0) {
            // Hot path: high bit set => hub index into shared cache.
            neighbor_spin = (float)s_hub_vals[e & HUB_IDX_MASK];
        } else {
            // Rare fallback: non-hub neighbor, read straight from global.
            neighbor_spin = (float)gpuLatSpin[e];
        }
        local_sum += sparse_J_values[start + k] * neighbor_spin;
    }

    // Warp-shuffle reduction.
    unsigned mask = 0xFFFFFFFFu;
    local_sum += __shfl_down_sync(mask, local_sum, 16);
    local_sum += __shfl_down_sync(mask, local_sum,  8);
    local_sum += __shfl_down_sync(mask, local_sum,  4);
    local_sum += __shfl_down_sync(mask, local_sum,  2);
    local_sum += __shfl_down_sync(mask, local_sum,  1);

    if (lane == 0) {
        float dE = -2.0f * (local_sum + gpuLinTermsVect[vertice_Id]) * current_spin;
        float acceptance = fminf(1.0f, expf(-beta * dE));
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
        float*               d_total_energy,
        const FlipCandidate* candidates,
        const int*           num_candidates
){
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int n = *num_candidates;
    if (t >= n) return;
    int   sid = candidates[t].spin_id;
    float dE  = candidates[t].delta_energy;
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

// Initialize lattice spins
__global__ void init_spins_total_energy(
		const int* row_ptr,
		const int* col_idx,
		const float* J_values,
		float* gpuLinTermsVect,
		const float* __restrict__ randvals,
		signed char* gpuSpins,
		const unsigned int* gpu_num_spins,
		float* total_energy,
		curandState * state,
		unsigned long seed){

	unsigned int vertice_Id = blockIdx.x; // actual spin id in this threadBlock
	unsigned int p_Id = threadIdx.x;// which worker id

	if (p_Id == 0){
		float randval = randvals[vertice_Id];
		signed char val = (randval < 0.5f) ? -1 : 1;
		gpuSpins[vertice_Id] = val;// random spin init.
		curand_init(seed, blockIdx.x, 0, &state[blockIdx.x]);
	}
	__syncthreads();

	__shared__ float sh_mem_spins_Energy[THREADS];
    sh_mem_spins_Energy[p_Id] = 0;
    __syncthreads();
  
	// --- Sparse adjacency traversal ---
	int start = row_ptr[vertice_Id];
	int end   = row_ptr[vertice_Id + 1];
	
	for (int k = start + p_Id; k < end; k += blockDim.x){
	    int j = col_idx[k];
	    float Jij = J_values[k];
	    sh_mem_spins_Energy[p_Id] += Jij * (float)gpuSpins[j];
	}
	__syncthreads();

	for (int off = blockDim.x/2; off; off /= 2){
		if (threadIdx.x < off){
		 sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];
		}
		__syncthreads();
	}

	if (p_Id == 0){

 		// Original vertice_energy
		// float vertice_energy = ((float)gpuSpins[vertice_Id]) * ( sh_mem_spins_Energy[0] - gpuLinTermsVect[vertice_Id] );
		// float vertice_energy = ((float)gpuSpins[vertice_Id]) * ( sh_mem_spins_Energy[0] + gpuLinTermsVect[vertice_Id] );
		// hamiltonian_per_spin[vertice_Id] = vertice_energy;// each threadblock updates its own memory location

		float current_spin = (float)gpuSpins[vertice_Id];
		float J_term = 0.5f * current_spin * sh_mem_spins_Energy[0];
		float h_term = current_spin * gpuLinTermsVect[vertice_Id];
		float vertice_energy = J_term + h_term;

		// printf("vertice_energy  %f \n", vertice_energy);
		atomicAdd(total_energy, vertice_energy);
	}

	// printf("%d total %.1f",blockIdx.x, total_energy);
}

// fINAL lattice spins
__global__ void final_spins_total_energy(
		const int* row_ptr,
	    const int* col_idx,
	    const float* J_values,
		float* gpuLinTermsVect,
		signed char* gpuSpins,
		const unsigned int* gpu_num_spins,
		float* total_energy){

	unsigned int vertice_Id = blockIdx.x;  // actual spin id in this threadBlock
	unsigned int p_Id = threadIdx.x;  // which worker id

	__shared__ float sh_mem_spins_Energy[THREADS];
	sh_mem_spins_Energy[p_Id] = 0;
	__syncthreads();

	// --- Sparse adjacency traversal ---
	int start = row_ptr[vertice_Id];
	int end   = row_ptr[vertice_Id + 1];
	
	for (int k = start + p_Id; k < end; k += blockDim.x){
	    int j = col_idx[k];
	    float Jij = J_values[k];
	    sh_mem_spins_Energy[p_Id] += Jij * (float)gpuSpins[j];
	}
	__syncthreads();

	for (int off = blockDim.x / 2; off; off /= 2){
		if (threadIdx.x < off){
			sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];
		}
		__syncthreads();
	}

	if (p_Id == 0){

        // Original vertice energy
		// float vertice_energy = ((float)gpuSpins[vertice_Id]) * ( sh_mem_spins_Energy[0] - gpuLinTermsVect[vertice_Id] );
		// float vertice_energy = ((float)gpuSpins[vertice_Id]) * ( sh_mem_spins_Energy[0] + gpuLinTermsVect[vertice_Id] );
		// hamiltonian_per_spin[vertice_Id] = vertice_energy;// each threadblock updates its own memory location

		float current_spin = (float)gpuSpins[vertice_Id];
		float J_term = 0.5f * current_spin * sh_mem_spins_Energy[0];
		float h_term = current_spin * gpuLinTermsVect[vertice_Id];
		float vertice_energy = J_term + h_term;

		//printf("vertice_energy  %d %f \n",vertice_Id, vertice_energy);
		atomicAdd(total_energy, vertice_energy);
	}
	// printf("%d total %.1f",blockIdx.x, total_energy);
}

std::vector<double> create_beta_schedule_geometric(
		uint32_t num_sweeps, 
		double temp_start,
		double temp_end, 
		double alpha){

	std::vector<double> beta_schedule;
	double current_temp = temp_start;
	
	// Calculate required iterations to reach temp_end (if num_sweeps not specified)
	num_sweeps = log(temp_end / temp_start) / log(alpha);
	
	for (int i = 0; i < num_sweeps; i++){

		beta_schedule.push_back(1.0 / current_temp);
		current_temp *= alpha;
		
		if (current_temp < temp_end){
			current_temp = temp_end;  // Floor at minimum temperature
		}
	}

	return beta_schedule;
}
