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

#include "annealer_gpu_SI/utils.hpp"

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

//
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

__global__ void d_debug_kernel(const int* row_ptr, const int* col_idx, const float* J_values, signed char* gpu_spins, signed char* gpu_spins_1, const unsigned int* gpu_num_spins);


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
__global__ void init_spins_total_energy(const int* row_ptr, const int* col_idx, const float* J_values,
	float* gpuLinTermsVect,
	const float* __restrict__ randvals,
	signed char* gpuSpins,
	const unsigned int* gpu_num_spins,
	float* total_energy,
	curandState * state,
	unsigned long seed);

// fINAL lattice spins
__global__ void final_spins_total_energy(const int* row_ptr, const int* col_idx, const float* J_values,
       float* gpuLinTermsVect,
       signed char* gpuSpins,
       const unsigned int* gpu_num_spins,
       float* total_energy);

__global__ void changeInLocalEnePerSpin(const int* row_ptr, const int* col_idx, const float* J_values,
	float* gpuLinTermsVect,
	const float* __restrict__ randvals,
	signed char* gpuLatSpin,
	const unsigned int* gpu_num_spins,
	const float beta,
	float* total_energy,
	curandState* globalState);
  
__global__ void d_avg_magnetism(signed char* gpuSpins, const unsigned int* gpu_num_spins, float* avg_magnetism);

// Initialize lattice spins
__global__ void preprocess_max_cut_from_ising(const int* row_ptr, const int* col_idx, const float* J_values,
	signed char* gpuSpins,
	const unsigned int* gpu_num_spins,
	float* max_cut_value,
	int* plus_one_spin,
	int* minus_one_spin);

std::vector<double> create_beta_schedule_geometric(uint32_t num_sweeps, double temp_start, double temp_end, double alpha);

__device__ volatile int sem = 0;

__global__ void initSemaphore() {
	sem = 0;
}

__device__ void acquire_semaphore(volatile int *lock){
  while (atomicCAS((int *)lock, 0, 1) != 0);
}

__device__ void release_semaphore(volatile int *lock){
  *lock = 0;
  __threadfence();
}
  
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
		"\t\t Print the final lattice value and shows avg magnetization at every temperature\n"
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
  unsigned long long seed = ((getpid()* rand()) & 0x7FFFFFFFF); //((GetCurrentProcessId()* rand()) & 0x7FFFFFFFF);
  
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
		int ch = getopt_long(argc, argv, "a:l:x:y:s:c:n:m:odeh", long_options, &option_index);
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

    std::cout << "filename " << filename << " linear filename " << linear_file << " start temp " << start_temp << " stop temp " << stop_temp << " seed " << seed << " num temp " << num_temps << " num sweeps " <<  num_sweeps_per_beta << std::endl;
	
	// ---------------- Sparse J loading ----------------
	if (row_ptr_file.empty() || col_idx_file.empty() || values_file.empty()) {
	    std::cerr << "ERROR: Must provide --row_ptr_file, --col_idx_file, and --values_file\n";
	    exit(1);
	}

 	double starttime = rtclock();

	// Parse data for CSR representation of J matrix
	ParseSparseData parseSparse(row_ptr_file, col_idx_file, values_file);

    std::cout << "ParseData constructed successfully" << std::endl;

	std::vector<float> linearTermsVect;
	//if (linear_file.empty() == false)
	parseData.readLinearValues(linear_file, linearTermsVect);

	double endtime = rtclock();
  
    if(debug)
  	  printtime("ParseData time: ", starttime, endtime);

	unsigned int num_spins = parseSparse.getNumSpins();
	unsigned int nnz       = parseSparse.getNNZ();
	
	const std::vector<int>&    row_ptr = parseSparse.getRowPtr();
	const std::vector<int>&    col_idx = parseSparse.getColIdx();
	const std::vector<float>&  J_values = parseSparse.getValues();
	
	std::cout << "Sparse J loaded: num_spins = "
	          << num_spins << ", nnz = " << nnz << std::endl;

    // CPU_THREADS does not seem to be used
	// unsigned int CPU_THREADS = THREADS;//(num_spins < 32) ? num_spins : 32; 

//	cudaMemcpyToSymbol( &THREADS, &CPU_THREADS, sizeof(unsigned int));
	// Setup cuRAND generator
	
	std::cout << "num_spins: " << num_spins
          << " nnz: " << nnz
          << " num temps: " << num_temps
          << " sweeps per beta: " << num_sweeps_per_beta
          << std::endl;

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
	
	size_t num_spins = row_ptr.size() - 1;
	size_t nnz = J_values.size();

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
	//printVecOfVec(adjMat);

	unsigned int* gpu_num_spins;
	gpuErrchk(cudaMalloc((void**)&gpu_num_spins, sizeof(*gpu_num_spins)));
	gpuErrchk(cudaMemcpy(gpu_num_spins, &num_spins, sizeof(*gpu_num_spins), cudaMemcpyHostToDevice));

	int* gpu_plus_one_spin;
	cudaHostAlloc(&gpu_plus_one_spin, sizeof(int), 0);

	int* gpu_minus_one_spin;
	cudaHostAlloc(&gpu_minus_one_spin, sizeof(int), 0);

	int* gpu_best_plus_one_spin;
	cudaHostAlloc(&gpu_best_plus_one_spin, sizeof(int), 0);
	gpu_best_plus_one_spin[0] = 0;

	int* gpu_best_minus_one_spin;
	cudaHostAlloc(&gpu_best_minus_one_spin, sizeof(int), 0);
	gpu_best_minus_one_spin[0] = 0;

	
	float* gpu_total_energy;
	cudaHostAlloc(&gpu_total_energy, sizeof(float), 0);

	float* gpu_best_energy;
	cudaHostAlloc(&gpu_best_energy, sizeof(float), 0);

	float* gpu_max_cut_value;
	cudaHostAlloc(&gpu_max_cut_value, sizeof(float), 0);

	float* gpu_best_max_cut_value;
	cudaHostAlloc(&gpu_best_max_cut_value, sizeof(float), 0);
	gpu_best_max_cut_value[0] = -1000.f;

	float* gpu_avg_magnetism;	
	cudaHostAlloc(&gpu_avg_magnetism, sizeof(*gpu_avg_magnetism), 0);	
	gpu_avg_magnetism[0] = 0.f;
 
	// Setup spin values
	signed char *gpu_spins;
	gpuErrchk(cudaMalloc((void**)&gpu_spins, num_spins * sizeof(*gpu_spins)));

	std::cout << "initialize spin values " << std::endl;
	// int blocks = (num_spins + THREADS - 1) / THREADS;
	curandGenerateUniform(rng, gpu_randvals, num_spins);
	
  //d_debug_kernel<<< 1, 1>>>(gpuAdjMat, gpu_adj_mat_size, gpu_num_spins);

// is a seed for random number generator
	time_t t;
	time(&t);
 
	//  create random states    
	curandState* devRanStates;
	cudaMalloc(&devRanStates, num_spins * sizeof(curandState));
 	
   starttime = rtclock();

	init_spins_total_energy << < num_spins, THREADS >> > (row_ptr, 
		col_idx, 
		J_values,
		gpuLinTermsVect,
		gpu_randvals,
		gpu_spins,
		gpu_num_spins,
		gpu_total_energy,
		devRanStates,
		(unsigned long)t);
  
  cudaDeviceSynchronize();
      
 	 endtime = rtclock();

	printtime("init_spins values and calculate total Energy time: ", starttime, endtime);
 

	gpuErrchk(cudaPeekAtLastError());

	gpu_best_energy[0] = gpu_total_energy[0];

	std::cout << "start annealing with initial energy: " << gpu_best_energy[0] << std::endl;
	std::vector<double> beta_schedule = create_beta_schedule_geometric(num_temps, start_temp, stop_temp, alpha);


  std::string out_filename = "avgmagnet_";  
  std::string in_adjmat = filename;
  {
    // Find position of '_' using find()
    int pos = in_adjmat.find_last_of("_");
    // Copy substring after pos
    std::string sub = in_adjmat.substr(pos + 1);
    out_filename += sub;
  }

 	FILE* fptr = fopen(out_filename.c_str() , "w");

	auto t0 = std::chrono::high_resolution_clock::now();
	auto annealing_start = std::chrono::high_resolution_clock::now(); 

// temperature 
	for (int i = 0; i < beta_schedule.size(); i++)
	{
	 int no_update = 0;
	 cudaEvent_t start, stop;
   if(debug)
   {   
     // @ Debugging
     
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
    }         
         
      for(int ii = 0; ii < num_sweeps_per_beta; ii++)
	    {   
        //int prev_energy = gpu_total_energy[0];
        initSemaphore<<<1, 1>>>();
        curandGenerateUniform(rng, gpu_randvals, num_spins);
   if(debug)
   {         
        cudaEventRecord(start); 
   }
      	changeInLocalEnePerSpin << < num_spins, THREADS >> > (row_ptr, 
				col_idx, 
				J_values,
				gpuLinTermsVect,
      			gpu_randvals,
      			gpu_spins,
      			gpu_num_spins,
      			beta_schedule.at(i),
      			gpu_total_energy,
      			devRanStates);
                        
    if(debug)
    {
         cudaEventRecord(stop);   
         cudaEventSynchronize(stop);
         float milliseconds = 0;
         cudaEventElapsedTime(&milliseconds, start, stop);
         printf("Elapse time : %f ms \n", milliseconds);
    }     
       cudaDeviceSynchronize();
       
       if(gpu_total_energy[0] > gpu_best_energy[0])
           no_update = 0;
       
       gpu_best_energy[0] = std::min(gpu_total_energy[0], gpu_best_energy[0]);
    	 
       if ((gpu_best_energy[0] - gpu_total_energy[0]) < CHANGE_MAX_ENERGY)
  		  	no_update = 0;
  		 else
  		  	no_update++;
  		// printf("cur engy %.1f best engy %.1f \n", gpu_total_energy[0], gpu_best_energy[0]);

		// Only check early stopping if not disabled
		if (!disable_early_stop && no_update > (BREAK_AFTER_ITERATION) * num_sweeps_per_beta)
		{
		    printf("Breaking early at temperature iteration %d due to convergence\n", i);
		    break;
		}
              
// @R Debugging
if(debug)
{
   {	
       d_avg_magnetism << < 1, THREADS >> >(gpu_spins, gpu_num_spins, gpu_avg_magnetism);   	
   }
}     	
         cudaDeviceSynchronize();      

		 gpuErrchk(cudaPeekAtLastError());         		 
 	  }
          
  if(debug)
    fprintf(fptr, "Temperature %.6f magnet %.6f \n", 1.f/beta_schedule.at(i),  gpu_avg_magnetism[0]); 

  energy_history.push_back(gpu_best_energy[0]);

	}
 
	auto t1 = std::chrono::high_resolution_clock::now();

	double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

	auto annealing_end = std::chrono::high_resolution_clock::now();
	double annealing_duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(annealing_end - annealing_start).count();
	printf("Total annealing time: %.6f seconds\n", annealing_duration * 1e-6);

	// Write energy history to file for plotting
	std::string energy_filename = "energy_history_";
	{
	    int pos = filename.find_last_of("_");
	    std::string sub = filename.substr(pos + 1);
	    energy_filename += sub;
	}
	
	FILE* energy_fptr = fopen(energy_filename.c_str(), "w");
	fprintf(energy_fptr, "# Iteration\tBest_Energy\n");
	for (int i = 0; i < energy_history.size(); i++)
	{
	    fprintf(energy_fptr, "%d\t%.6f\n", i, energy_history[i]);
	}
	fclose(energy_fptr);
	// printf("Energy history written to: %s\n", energy_filename.c_str());

	fprintf(fptr, "duration %.3f \n", (duration * 1e-6) );
	fclose(fptr);

 
 // @R Debugging 
/*	d_debug_kernel << < 1, 1 >> > (row_ptr, col_idx, J_values,
		gpu_spins,
		gpu_spins_1,
		gpu_num_spins);
*/   


  
  signed char cpu_spins[num_spins];

	gpu_max_cut_value[0] = 0.f;
	gpu_plus_one_spin[0] = 0;
	gpu_minus_one_spin[0] = 0;
  gpu_total_energy[0] = 0.f;

   {

       final_spins_total_energy << < num_spins, THREADS >> > (row_ptr, 
						col_idx, 
						J_values,
                        gpuLinTermsVect,
                        gpu_spins,
                        gpu_num_spins,
                        gpu_total_energy);

       preprocess_max_cut_from_ising << < num_spins, THREADS >> > (row_ptr, 
				col_idx, 
				J_values,
  				gpu_spins,
  				gpu_num_spins,
  				gpu_max_cut_value,
  				gpu_plus_one_spin,
  				gpu_minus_one_spin);
  
  			cudaDeviceSynchronize();
       gpuErrchk(cudaMemcpy(cpu_spins, gpu_spins, num_spins * sizeof(*gpu_spins), cudaMemcpyDeviceToHost));
       gpu_max_cut_value[0] *= -0.5f; 
   }     
        
			gpu_best_max_cut_value[0] = std::max(gpu_best_max_cut_value[0], gpu_max_cut_value[0]);
			gpu_best_plus_one_spin[0] = std::max(gpu_best_plus_one_spin[0], gpu_plus_one_spin[0]);
			gpu_best_minus_one_spin[0] = std::max(gpu_best_minus_one_spin[0], gpu_minus_one_spin[0]);
			printf("cur engy %.1f curr cut %.1f best cut %.1f with best +1 %d and -1 %d \n", gpu_total_energy[0], gpu_max_cut_value[0], gpu_best_max_cut_value[0], gpu_best_plus_one_spin[0], gpu_best_minus_one_spin[0]);

 if(debug)
 {
 

 
  std::string spins_filename = "spins_";  
  
  std::string adjmat = filename;

  {
    // Find position of '_' using find()
    int pos = adjmat.find_last_of("_");
    // Copy substring after pos
    std::string sub = adjmat.substr(pos + 1);
    spins_filename += sub;
  }

 	FILE* fptr1 = fopen(spins_filename.c_str() , "w");
  for(int i = 0; i < num_spins; i++)
  {
        fprintf(fptr1, "%d\t",  (int)cpu_spins[i]);

  }  
  fprintf(fptr1,"\n\n\n");
  //fprintf(fptr1,"\tbest energy value: %.6f\n", gpu_best_energy[0] );
  fprintf(fptr1,"\ttotal energy value: %.6f\n", gpu_total_energy[0] );
  fprintf(fptr1,"\tbest max cut value: %.6f\n", gpu_best_max_cut_value[0]);
	// fprintf(fptr1," \telapsed time in sec: %.6f\n", duration * 1e-6 );
  fclose(fptr1);
  
 }
	std::cout << "\ttotal energy value: " << gpu_total_energy[0] << std::endl;
	std::cout << "\tbest max cut value: " << gpu_best_max_cut_value[0] << std::endl;
	// std::cout << "\telapsed time in sec: " << duration * 1e-6 << std::endl;
 
	cudaFree(gpu_randvals);
	cudaFree(row_ptr);
	cudaFree(col_idx);
	cudaFree(J_values);
	cudaFree(gpu_num_spins);
	cudaFree(gpu_spins);
	return 0;
}




#define CORRECT 1

#if CORRECT

__global__ void changeInLocalEnePerSpin(const int* row_ptr,
    const int* col_idx,
    const float* J_values,
	float* gpuLinTermsVect,
	const float* __restrict__ randvals,
	signed char* gpuLatSpin,
	const unsigned int* gpu_num_spins,
	const float beta,
	float* total_energy,
	curandState* globalState) {

	unsigned int vertice_Id = blockIdx.x;
	unsigned int p_Id = threadIdx.x;    //32 worker threads 
	// for each neighour of vertex id pull the gpucurrentupdate[i] and place it in the shared memory

  __syncthreads();
  if (threadIdx.x == 0)
     acquire_semaphore(&sem);
  __syncthreads();

	// shared  spin_v0|spin_v1|.......|J_spin0| J_spin1| J_spin2|..
	__shared__ float sh_mem_spins_Energy[THREADS];
    sh_mem_spins_Energy[p_Id] = 0;
    __syncthreads();

	float current_spin_shared_mem;

	current_spin_shared_mem = (float)gpuLatSpin[vertice_Id];

	// --- Sparse adjacency traversal ---
	// For vertex vertice_Id, neighbors are in
	// [row_ptr[vertice_Id], row_ptr[vertice_Id + 1])
	
	int start = gpu_row_ptr[vertice_Id];
	int end   = gpu_row_ptr[vertice_Id + 1];
	
	// Each thread accumulates over a strided subset of neighbors
	for (int k = start + p_Id; k < end; k += blockDim.x)
	{
	    int j = gpu_col_idx[k];                // neighbor index
	    float Jij = gpu_J_values[k];           // coupling value
	    sh_mem_spins_Energy[p_Id] += Jij * (float)gpuLatSpin[j];
	}
	__syncthreads();


  for (int off = blockDim.x/2; off; off /= 2) {
     if (threadIdx.x < off) {
         sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];
       }
   __syncthreads();
   }
   
  __syncthreads();
	
	if (p_Id == 0)
	{
      	  // Original delta energy expression
      	  // float local_ham_per_spin =  - 2.f * ( (-1.f * sh_mem_spins_Energy[0]) - gpuLinTermsVect[vertice_Id] ) * current_spin_shared_mem;
      	  float local_ham_per_spin =  - 2.f * ( (sh_mem_spins_Energy[0]) + gpuLinTermsVect[vertice_Id] ) * current_spin_shared_mem; //  final energy - current energy
	
	  float prob_ratio = exp(-1.f * beta * (local_ham_per_spin)); // exp(- (E_f - E_i) / T)
        
	  float acceptance_probability = min((float)1.f, prob_ratio);

	  if (randvals[vertice_Id] < acceptance_probability)
      	  {
		gpuLatSpin[vertice_Id] = (signed char)(-1.f * current_spin_shared_mem); 

		atomicAdd(total_energy, local_ham_per_spin);
      	  }
	}
 
 __threadfence(); 
 __syncthreads();
 if (threadIdx.x == 0)
   release_semaphore(&sem);
 __syncthreads();

}


#endif

// Initialize lattice spins
__global__ void init_spins_total_energy(const int* row_ptr,
    const int* col_idx,
    const float* J_values,
	float* gpuLinTermsVect,
	const float* __restrict__ randvals,
	signed char* gpuSpins,
	const unsigned int* gpu_num_spins,
	float* total_energy,
	curandState * state,
	unsigned long seed) {

	unsigned int vertice_Id = blockIdx.x; // actual spin id in this threadBlock
	unsigned int p_Id = threadIdx.x;// which worker id

	if (p_Id == 0)
	{
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
	int start = gpu_row_ptr[vertice_Id];
	int end   = gpu_row_ptr[vertice_Id + 1];
	
	for (int k = start + p_Id; k < end; k += blockDim.x)
	{
	    int j = gpu_col_idx[k];
	    float Jij = gpu_J_values[k];
	    sh_mem_spins_Energy[p_Id] += Jij * (float)gpuSpins[j];
	}
	__syncthreads();


  for (int off = blockDim.x/2; off; off /= 2) {
     if (threadIdx.x < off) {
         sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];
       }
   __syncthreads();
   }
 

	if (p_Id == 0)
	{
 		// Original vertice_energy
		// float vertice_energy = ((float)gpuSpins[vertice_Id]) * ( sh_mem_spins_Energy[0] - gpuLinTermsVect[vertice_Id] );
		// float vertice_energy = ((float)gpuSpins[vertice_Id]) * ( sh_mem_spins_Energy[0] + gpuLinTermsVect[vertice_Id] );
		// hamiltonian_per_spin[vertice_Id] = vertice_energy;// each threadblock updates its own memory location

		float current_spin = (float)gpuSpins[vertice_Id];
		float J_term = 0.5f * current_spin * sh_mem_spins_Energy[0];
		float h_term = current_spin * gpuLinTermsVect[vertice_Id];
		float vertice_energy = J_term + h_term;

//		printf("vertice_energy  %f \n", vertice_energy);
		atomicAdd(total_energy, vertice_energy);
	}

	//        printf("%d total %.1f",blockIdx.x, total_energy);
}

// fINAL lattice spins
__global__ void final_spins_total_energy(const int* row_ptr,
    const int* col_idx,
    const float* J_values,
	float* gpuLinTermsVect,
	signed char* gpuSpins,
	const unsigned int* gpu_num_spins,
	float* total_energy) {

	unsigned int vertice_Id = blockIdx.x; // actual spin id in this threadBlock
	unsigned int p_Id = threadIdx.x;// which worker id

	__shared__ float sh_mem_spins_Energy[THREADS];
	sh_mem_spins_Energy[p_Id] = 0;
	__syncthreads();

	// --- Sparse adjacency traversal ---
	int start = gpu_row_ptr[vertice_Id];
	int end   = gpu_row_ptr[vertice_Id + 1];
	
	for (int k = start + p_Id; k < end; k += blockDim.x)
	{
	    int j = gpu_col_idx[k];
	    float Jij = gpu_J_values[k];
	    sh_mem_spins_Energy[p_Id] += Jij * (float)gpuSpins[j];
	}
	__syncthreads();


	for (int off = blockDim.x / 2; off; off /= 2) {
		if (threadIdx.x < off) {
			sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];
		}
		__syncthreads();
	}


	if (p_Id == 0)
	{
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

	//        printf("%d total %.1f",blockIdx.x, total_energy);
}

// Initialize lattice spins
__global__ void preprocess_max_cut_from_ising(const int* row_ptr,
    const int* col_idx,
    const float* J_values,
	signed char* gpuSpins,
	const unsigned int* gpu_num_spins,
	float* max_cut_value,
	int* plus_one_spin,
	int* minus_one_spin) {

	unsigned int vertice_Id = blockIdx.x; // actual spin id in this threadBlock
	unsigned int p_Id = threadIdx.x;// which worker id
	float current_spin_row = (float)gpuSpins[vertice_Id];

	__shared__ float sh_mem_spins_Energy[THREADS];
    sh_mem_spins_Energy[p_Id] = 0;
    __syncthreads();

	// --- Sparse adjacency traversal ---
	int start = gpu_row_ptr[vertice_Id];
	int end   = gpu_row_ptr[vertice_Id + 1];
	
	for (int k = start + p_Id; k < end; k += blockDim.x)
	{
	    int j = gpu_col_idx[k];
	    float Jij = gpu_J_values[k];
	    sh_mem_spins_Energy[p_Id] +=
	        Jij * (1.f - current_spin_row * (float)gpuSpins[j]);
	}
	__syncthreads();

  for (int off = blockDim.x/2; off; off /= 2) {
     if (threadIdx.x < off) {
         sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];
       }
   __syncthreads();
   }
   
	if (p_Id == 0)
	{

		float vertice_energy;
		// Origial vertice_energy
		vertice_energy = (0.5f) * sh_mem_spins_Energy[0];
		// vertice_energy = sh_mem_spins_Energy[0];

		atomicAdd(max_cut_value, vertice_energy);

		if (current_spin_row == 1.f)
			atomicAdd(plus_one_spin, 1);
		else
			atomicAdd(minus_one_spin, 1);
	}

	//       
}


std::vector<double> create_beta_schedule_geometric(uint32_t num_sweeps, double temp_start, double temp_end, double alpha)
{
	std::vector<double> beta_schedule;
	double current_temp = temp_start;
	
	// Calculate required iterations to reach temp_end (if num_sweeps not specified)
	num_sweeps = log(temp_end / temp_start) / log(alpha);
	
	for (int i = 0; i < num_sweeps; i++)
	{
		beta_schedule.push_back(1.0 / current_temp);
		current_temp *= alpha;
		
		if (current_temp < temp_end) {
			current_temp = temp_end;  // Floor at minimum temperature
		}
	}
	
	return beta_schedule;
}

__global__ void d_debug_kernel(const int* row_ptr, const int* col_idx, const float* J_values, signed char* gpu_spins, signed char* gpu_spins_1, const unsigned int* gpu_num_spins)
{
	int ones = 0;
	int ones_1 = 0;
	for (int i = 0; i < gpu_num_spins[0]; i++)
	{
		printf("%d %.1f ", i, (float)gpu_spins[i]);
		if ((float)gpu_spins[i] == 1.f)
			ones++;
		if ((float)gpu_spins_1[i] == -1.f)
			ones_1++;
	}

	printf("\n");
	printf("\n");
	printf("%d %d \n", ones, ones_1);
	int m_ones = 0;
	int m_ones_1 = 0;
	for (int i = 0; i < gpu_num_spins[0]; i++)
	{
		printf("%d %.1f ", i, (float)gpu_spins_1[i]);
		if ((float)gpu_spins[i] == 1.f)
			m_ones++;
		if ((float)gpu_spins_1[i] == -1.f)
			m_ones_1++;
	}
	printf("\n");
	printf("\n");
	printf("%d %d\n", m_ones, m_ones_1);
}

__global__ void d_avg_magnetism(signed char* gpuSpins, const unsigned int* gpu_num_spins, float* avg_magnetism)	
{	
  unsigned int p_Id = threadIdx.x;	
  	
	__shared__ float sh_mem_spins_Energy[THREADS];	
  sh_mem_spins_Energy[p_Id] = 0;	
  __syncthreads();	

    // num_iter does not seem to be used here
 	// int num_iter = (gpu_num_spins[0] + THREADS - 1) / THREADS;
   	 	
	for (int i = 0; i < gpu_num_spins[0]; i++)	
	{	
		// p_Id (worker group)	
		if (p_Id + i * THREADS < gpu_num_spins[0])	
		{		
			sh_mem_spins_Energy[p_Id] += ((float)gpuSpins[p_Id + i * THREADS]); 	
		}	
	}	
	__syncthreads();	
 	
   for (int off = blockDim.x/2; off; off /= 2) {	
     if (threadIdx.x < off) {	
         sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];	
       }	
   __syncthreads();	
   }	
   	
	if (p_Id == 0)	
	{	
      avg_magnetism[0] = sh_mem_spins_Energy[0]/gpu_num_spins[0];		
  }	
}
