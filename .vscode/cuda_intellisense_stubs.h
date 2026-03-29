// CUDA IntelliSense stubs — for IDE support only, never compiled by nvcc.
// Declares device built-ins that are hidden without __CUDACC__.
#pragma once
#include <cstdint>

// ── Thread/block built-ins ───────────────────────────────────────────────────
// dim3 / uint3 come from vector_types.h (included by cuda_runtime.h),
// so only declare the built-in variables.
extern uint3 threadIdx;
extern uint3 blockIdx;
extern dim3  blockDim;
extern dim3  gridDim;
extern int   warpSize;

// ── Synchronization ──────────────────────────────────────────────────────────
void __syncthreads();
void __syncwarp(unsigned int mask = 0xffffffff);

// ── Type punning ─────────────────────────────────────────────────────────────
int          __float_as_int(float x);
float        __int_as_float(int x);
unsigned int __float_as_uint(float x);
float        __uint_as_float(unsigned int x);

// ── Atomics ──────────────────────────────────────────────────────────────────
int          atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address, unsigned int val);
float        atomicAdd(float* address, float val);
int          atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val);
int          atomicMin(int* address, int val);
int          atomicMax(int* address, int val);

// ── Warp shuffle ─────────────────────────────────────────────────────────────
float __shfl_down_sync(unsigned int mask, float var, unsigned int delta, int width = 32);
int   __shfl_down_sync(unsigned int mask, int   var, unsigned int delta, int width = 32);

// ── cuRAND device API ────────────────────────────────────────────────────────
// curand_kernel.h is __CUDACC__-gated, so define the essentials here.
struct curandStateXORWOW { unsigned int d, v[5]; unsigned int boxmuller_flag; float boxmuller_extra; };
typedef curandStateXORWOW curandState;

void  curand_init(unsigned long long seed, unsigned long long sequence,
                  unsigned long long offset, curandState* state);
float curand_uniform(curandState* state);
