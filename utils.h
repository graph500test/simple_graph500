#ifndef __UTILS_H
#define __UTILS_H

#include <iostream>
#include <random>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sys/time.h>
#include <string>

#include <cuda.h>

#define MAX_32S 2147483647 
#define MAX_32U 4294967295
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

void set_device(int x){
  cudaSetDevice(x);
  printf("Using Device %d\n", x);
}

void query_device_prop(){
  int nDevices;
	cudaGetDeviceCount(&nDevices);
	for(int i = 0; i < nDevices; i++){
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf(" Device name: %s\n", prop.name);
    printf(" Device Capability: %d.%d\n", prop.major, prop.minor);
    printf(" Device Overlap: %s\n", (prop.deviceOverlap ? "yes":"no"));
    printf(" Device canMapHostMemory: %s\n", (prop.canMapHostMemory ? "yes":"no"));
    printf(" Memory Detils\n");
    printf("  - Share Memory per Block (KB): %.2f\n", (prop.sharedMemPerBlock+.0)/(1<<10));
    printf("  - Total Global Memory (GB): %.2f\n", (prop.totalGlobalMem+.0)/(1<<30));
		printf("  - Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  - Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  - Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf(" Thread Detils\n");
    printf("  - max threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  - processor Count: %d\n", prop.multiProcessorCount);
    printf("\n");
	}
}

// HandleError
static void 
HandleError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", \
    cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
  }
}

#define H_ERR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define ASSERT( Predicate, Err_msg ) \
if(true){                            \
  if( !(Predicate) ){                \
    std::cerr << "CHECK failed :"    \
      << Err_msg  << " at ("         \
      << __FILE__ << ", "            \
      << __LINE__ << ")"             \
      << std::endl;                  \
		exit(1);						             \
  }                                  \
}


// timming
#define RDTSC(val) do {			                          \
    uint64_t __a,__d;					                        \
    asm volatile("rdtsc" : "=a" (__a), "=d" (__d));		\
    (val) = ((uint64_t)__a) | (((uint64_t)__d)<<32);	\
  } while(0)

static inline uint64_t rdtsc() {
  uint64_t val;
  RDTSC(val);
  return val;
}

inline double wtime()
{
	double time[2];	
	struct timeval time1;
	gettimeofday(&time1, NULL);

	time[0]=time1.tv_sec;
	time[1]=time1.tv_usec;

	return time[0]+time[1]*1.0e-6;
}

// data type 
typedef int     index_t;
typedef int64_t packed_t;

// const data 
const int BIN_SZ = 512;
const int Q_NUM  = 3;
const int CTANUM = 256;
const int THDNUM = 256;

// cuda
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

__device__ __forceinline__
int warpReduceSum(int val){
  for(int offset = 32>>1; offset>0; offset>>=1)
    val += __shfl_down(val, offset);
  return val;
}

__device__ __forceinline__
int blockReduceSum(int val){
  static __shared__ int shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);
  if(lane==0) shared[wid]=val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x/32) ? shared[lane]:0;
  if(wid==0) val=warpReduceSum(val);
  return val;
}

// N = blocksize = 256
__global__ void reduce(int *dg_in, int* dg_ans, int N){
  const int gtid = blockIdx.x*blockDim.x + threadIdx.x;
  const int STRIDE = blockDim.x*gridDim.x;
  int sum = 0;
  for(int idx=gtid; idx<N; idx+=STRIDE) sum += dg_in[idx];
  sum = blockReduceSum(sum);
  if(threadIdx.x==0) *dg_ans = sum;
}


template<typename data_t>
__global__ void __excudaMemset(data_t* dg_in, data_t dft, size_t N){
  const int gtid = blockIdx.x*blockDim.x + threadIdx.x;
  const int STRIDE = blockDim.x*gridDim.x;
  for(int idx=gtid; idx<N; idx+=STRIDE) dg_in[idx] = dft;
}

template<typename data_t>
__host__ void excudaMemset(data_t* dg_in, data_t dft, size_t N){
  __excudaMemset<<<CTANUM,THDNUM>>>(dg_in, dft, N);
  cudaThreadSynchronize();
}

// data_t is packed with 4bytes
template<typename data_t>
__device__ __forceinline__
data_t __exshfl_down(data_t data, int delta){
  int N = sizeof(data_t)/sizeof(int);
  int* x = (int*)&data;
  for(int i = 0; i < N; ++i){
    x[i] = __shfl_down(x[i], delta);
  }
  return *((data_t*)x);
}

// data_t is packed with 4bytes
template<typename data_t>
__device__ __forceinline__
bool __equals(data_t v, data_t u){
  int N = sizeof(data_t)/sizeof(int);
  int* x = (int*)&v;
  int* y = (int*)&u;
  bool flag=true;
  for(int i = 0; i < N; ++i){
    flag &= (x[i] == y[i]);
  }
  return flag;
}

__device__ float atomicMin(float* address, float val){
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
      __float_as_int(::fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

#endif
