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

#define THREADS 1024 //or more threads gpu crashes
#define BREAK_UPDATE_VAL 2//1000 
#define TCRIT 2.26918531421f

struct FlipCandidate {
    int    spin_id;
    float  delta_energy;
};

// ── Bin thresholds ────────────────────────────────────────────────────────
// Spins with row-length >= DENSE_THRESHOLD  → Bin 0: 1 block / 1024 threads
// Spins with row-length <  DENSE_THRESHOLD  → Bin 1: 1 warp  / 32  threads
// packed SPINS_PER_BLOCK_SPARSE at a time into one block of 1024 threads.
#define DENSE_THRESHOLD         128   // tune after inspecting degree histogram
#define SPINS_PER_BLOCK_SPARSE   32   // 32 spins × 32 threads = 1024 threads/block
 
// Max hub spins cached in __constant__ memory (compile-time literal)
#define MAX_HUB_SPINS            16   // safe upper bound given your pattern
 
// Constant-memory cache for hub spin VALUES (refreshed per flip if needed)
__constant__ signed char c_hub_spin_vals[MAX_HUB_SPINS];
// Constant-memory: global indices of hub spins (set once at startup)
__constant__ int 	c_hub_spin_ids[MAX_HUB_SPINS];
__constant__ int 	c_num_hub_spins;  // actual count (<= MAX_HUB_SPINS)

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
        const int*     dense_spin_ids   // maps blockIdx.x → global spin id
);


__global__ void collectFlipCandidates_sparse(
        const int*     row_ptr,
        const int*     col_idx,
        const float*   J_values,
        float*         gpuLinTermsVect,
        const float* __restrict__ randvals,
        signed char*   gpuLatSpin,
        const float    beta,
        FlipCandidate* candidates,
        int*           num_candidates,
        const int*     sparse_spin_ids,  // logical sparse index → global spin id
        int            num_sparse
);


__global__ void applySingleFlip(
        signed char*   gpuLatSpin,
        float*         d_total_energy,
        FlipCandidate* candidates,
        int            chosen_idx
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

	int h_num_hub = std::min(num_dense, (int)MAX_HUB_SPINS);
	 
	if (num_dense > MAX_HUB_SPINS)
	    std::cerr << "WARNING: more dense spins than MAX_HUB_SPINS ("
	              << MAX_HUB_SPINS << "). Increase the constant.\n";
	 
	// Upload hub indices to __constant__ memory (done once, never changes)
	cudaMemcpyToSymbol(c_hub_spin_ids,  dense_spins.data(), h_num_hub * sizeof(int));
	cudaMemcpyToSymbol(c_num_hub_spins, &h_num_hub,         sizeof(int));
	 
	// GPU arrays for bin membership

	int *gpu_dense_ids, *gpu_sparse_ids;
	gpuErrchk(cudaMalloc((void**)&gpu_dense_ids, num_dense  * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&gpu_sparse_ids, num_sparse * sizeof(int)));

	gpuErrchk(cudaMemcpy(gpu_dense_ids,  dense_spins.data(), num_dense  * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(gpu_sparse_ids, sparse_spins.data(), num_sparse * sizeof(int), cudaMemcpyHostToDevice));

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

	// ── Initialize hub values in constant memory before annealing ────────
	{
	    signed char h_hub_vals[MAX_HUB_SPINS];
	    for (int hh = 0; hh < h_num_hub; hh++) {
	        gpuErrchk(cudaMemcpy(&h_hub_vals[hh], 
	                             gpu_spins_old + dense_spins[hh],
	                             sizeof(signed char),
	                             cudaMemcpyDeviceToHost));
	    }
	    cudaMemcpyToSymbol(c_hub_spin_vals, h_hub_vals, h_num_hub * sizeof(signed char));
	}
 
	gpuErrchk(cudaPeekAtLastError());

	gpu_best_energy[0] = gpu_total_energy[0];

	// d_total_energy already holds the correct initial value from init_spins_total_energy.
	// No reset needed here — applySingleFlip will atomicAdd dE onto it incrementally.
	// We just confirm the device value is in sync before the loop starts.
	gpuErrchk(cudaMemcpy(d_total_energy, gpu_total_energy, sizeof(float), cudaMemcpyHostToDevice));

	auto t_setup_end = std::chrono::high_resolution_clock::now();
	double t_setup = (double)std::chrono::duration_cast<std::chrono::microseconds>(t_setup_end - t_setup_start).count() * 1e-6;

	std::cout << "start annealing with initial energy: " << gpu_best_energy[0] << std::endl;
	std::vector<double> beta_schedule = create_beta_schedule_geometric(num_temps, start_temp, stop_temp, alpha);

	auto t0 = std::chrono::high_resolution_clock::now();
	auto annealing_start = std::chrono::high_resolution_clock::now(); 

	// Temperature loop
	for (int i = 0; i < beta_schedule.size(); i++){

		int no_update = 0;
		cudaEvent_t start, stop;
		if(debug){   
			// @ Debugging
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
		}         
	         
		for (int ii = 0; ii < num_sweeps_per_beta; ii++) {
		
		    // 1. Generate fresh random values for Metropolis acceptance test
		    curandGenerateUniform(rng, gpu_randvals, num_spins);
		
		    // 2. Reset candidate counter
		    gpuErrchk(cudaMemset(gpu_num_candidates, 0, sizeof(int)));

			// 3a. Dense bin — one block per hub spin, 1024 threads each
			if (num_dense > 0) {
			    collectFlipCandidates_dense<<<num_dense, THREADS>>>(
			        gpu_row_ptr,
			        gpu_col_idx,
			        gpu_J_values,
			        gpuLinTermsVect,
			        gpu_randvals,
			        gpu_spins_old,
			        (float)beta_schedule[i],
			        gpu_candidates,
			        gpu_num_candidates,
			        gpu_dense_ids
			    );
			}
			 
			// 3b. Sparse bin — 32 spins per block, one warp per spin
			if (num_sparse > 0) {
			    int sparse_blocks = (num_sparse + SPINS_PER_BLOCK_SPARSE - 1) / SPINS_PER_BLOCK_SPARSE;
			    collectFlipCandidates_sparse<<<sparse_blocks, THREADS>>>(
			        gpu_row_ptr,
			        gpu_col_idx,
			        gpu_J_values,
			        gpuLinTermsVect,
			        gpu_randvals,
			        gpu_spins_old,
			        (float)beta_schedule[i],
			        gpu_candidates,
			        gpu_num_candidates,
			        gpu_sparse_ids,
			        num_sparse
			    );
			}
			 
			cudaDeviceSynchronize();   // single sync covers both kernels

		    // 4. Read candidate count back to host
		    int h_num_candidates = 0;
		    gpuErrchk(cudaMemcpy(&h_num_candidates, gpu_num_candidates, sizeof(int), cudaMemcpyDeviceToHost));
		
		    if (h_num_candidates > 0) {
		
		        // 5. Pick one candidate uniformly at random on the host
		        int chosen = (int)(((double)rand() / RAND_MAX) * h_num_candidates);
		        if (chosen >= h_num_candidates) chosen = h_num_candidates - 1;
		
		        // 6. Apply that single flip and update running energy via atomicAdd
		        applySingleFlip<<<1, 1>>>(
		            gpu_spins_old,
		            d_total_energy,
		            gpu_candidates,
		            chosen
		        );
		        cudaDeviceSynchronize();

				// Refresh hub values in __constant__ memory if a hub was flipped
				{
				    FlipCandidate h_chosen;
				    gpuErrchk(cudaMemcpy(&h_chosen, gpu_candidates + chosen, sizeof(FlipCandidate),
				                         cudaMemcpyDeviceToHost));
				 
				    bool is_hub = std::find(dense_spins.begin(), dense_spins.end(),
				                            h_chosen.spin_id) != dense_spins.end();
				    if (is_hub) {
				        // Pull updated values for hub slots only (MAX_HUB_SPINS bytes — negligible)
						signed char h_hub_vals[MAX_HUB_SPINS];
						int h_dense_ids[MAX_HUB_SPINS];

						gpuErrchk(cudaMemcpy(h_dense_ids, gpu_dense_ids, h_num_hub * sizeof(int), 
												cudaMemcpyDeviceToHost));

						for (int hh = 0; hh < h_num_hub; hh++) {
						    gpuErrchk(cudaMemcpy(&h_hub_vals[hh], gpu_spins_old + h_dense_ids[hh],
						                         sizeof(signed char),
						                         cudaMemcpyDeviceToHost));
						}
						cudaMemcpyToSymbol(c_hub_spin_vals, h_hub_vals, h_num_hub * sizeof(signed char));
				    }
				}

		        // 7. Read updated energy
		        gpuErrchk(cudaMemcpy(gpu_total_energy, d_total_energy, sizeof(float), cudaMemcpyDeviceToHost));
		
		        gpu_best_energy[0] = std::min(gpu_total_energy[0], gpu_best_energy[0]);
		
		        // 8. Early stopping
		        if ((gpu_best_energy[0] - gpu_total_energy[0]) < CHANGE_MAX_ENERGY)
		            no_update = 0;
		        else
		            no_update++;
		
		        if (!disable_early_stop && no_update > (int)(BREAK_AFTER_ITERATION * num_sweeps_per_beta)) {
		            printf("Breaking early at temperature iteration %d\n", i);
		            break;
		        }
		    }
			// Always sync energy to host at end of each sweep regardless of whether a flip occurred
			gpuErrchk(cudaMemcpy(gpu_total_energy, d_total_energy, sizeof(float), cudaMemcpyDeviceToHost));
		}
		gpuErrchk(cudaMemcpy(gpu_total_energy, d_total_energy, sizeof(float), cudaMemcpyDeviceToHost));
		gpu_best_energy[0] = std::min(gpu_total_energy[0], gpu_best_energy[0]);
		energy_history.push_back(gpu_best_energy[0]);
	    if(debug){
	        cudaEventDestroy(start);
	        cudaEventDestroy(stop);
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
	cudaFree(gpu_dense_ids);
	cudaFree(gpu_sparse_ids);

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

// Binary search to check if a spin_id is a hub and get its constant-memory index
__device__ __forceinline__ int find_hub_index(int spin_id) {
    // Linear search is faster than binary for small MAX_HUB_SPINS (16)
    // If c_num_hub_spins grows large, switch to binary search
    for (int i = 0; i < c_num_hub_spins; i++) {
        if (c_hub_spin_ids[i] == spin_id)
            return i;  // found — return index into c_hub_spin_vals[]
    }
    return -1;  // not a hub
}

__global__ void collectFlipCandidates_sparse(
        const int*     row_ptr,
        const int*     col_idx,
        const float*   J_values,
        float*         gpuLinTermsVect,
        const float* __restrict__ randvals,
        signed char*   gpuLatSpin,
        const float    beta,
        FlipCandidate* candidates,
        int*           num_candidates,
        const int*     sparse_spin_ids,  // logical sparse index → global spin id
        int            num_sparse
){
    // Which sparse spin does this warp own?
    int warp_in_block = threadIdx.x / 32;     // 0 .. SPINS_PER_BLOCK_SPARSE-1
    int lane          = threadIdx.x & 31;     // 0 .. 31
 
    int sparse_idx = (int)blockIdx.x * SPINS_PER_BLOCK_SPARSE + warp_in_block;
    if (sparse_idx >= num_sparse) return;
 
    int vertice_Id = sparse_spin_ids[sparse_idx];   // global spin index
 
    float current_spin = (float)gpuLatSpin[vertice_Id];
 
    int start = row_ptr[vertice_Id];
    int end   = row_ptr[vertice_Id + 1];
    int len   = end - start;
 
	// ── Warp-strided accumulation with hub constant-memory optimization ──
	float local_sum = 0.0f;
	for (int k = lane; k < len; k += 32) {
	    int neighbor_id = col_idx[start + k];
	    float neighbor_spin;
	    
	    // Check if neighbor is a hub — read from constant memory if so
	    int hub_idx = find_hub_index(neighbor_id);
	    if (hub_idx >= 0) {
	        neighbor_spin = (float)c_hub_spin_vals[hub_idx];  // constant memory (L1 cached)
	    } else {
	        neighbor_spin = (float)gpuLatSpin[neighbor_id];   // global memory
	    }
	    
	    local_sum += J_values[start + k] * neighbor_spin;
	}
 
    // ── Warp-shuffle reduction (no shared mem, no __syncthreads)
    // GH100: full warp mask, single-cycle shuffle
    unsigned mask = 0xFFFFFFFFu;
    local_sum += __shfl_down_sync(mask, local_sum, 16);
    local_sum += __shfl_down_sync(mask, local_sum,  8);
    local_sum += __shfl_down_sync(mask, local_sum,  4);
    local_sum += __shfl_down_sync(mask, local_sum,  2);
    local_sum += __shfl_down_sync(mask, local_sum,  1);
    // lane 0 now holds the complete neighbour sum
 
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


__global__ void applySingleFlip(
        signed char*   gpuLatSpin,
        float*         d_total_energy,
        FlipCandidate* candidates,
        int            chosen_idx          // which candidate was selected
){
    // single thread is sufficient
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int   sid = candidates[chosen_idx].spin_id;
        float dE  = candidates[chosen_idx].delta_energy;
        gpuLatSpin[sid] = -gpuLatSpin[sid];
        atomicAdd(d_total_energy, dE);
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
