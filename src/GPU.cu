#include "check.h"
#include "GPU.h"

#include <iostream>
#include <mutex>

using namespace std;

__device__ int maxConcurrentBlocksVar;
__device__ volatile int maxConcurrentBlockEvalDoneVar;

__device__ int* maxConcurrentBlocks()
{
	return &maxConcurrentBlocksVar;
}

__device__ int* maxConcurrentBlockEvalDone()
{
	return (int*)&maxConcurrentBlockEvalDoneVar;
}

__device__ volatile float BigData_[1024 * 1024];

__device__ volatile float* BigData()
{
	return ::BigData_;
}

template<int ITS, int REGS>
class DelayFMADS
{
public:

	__device__ __inline__
	static void delay()
	{
		float values[REGS];

		#pragma unroll
		for(int r = 0; r < REGS; ++r)
			values[r] = BigData()[threadIdx.x + r * 32];

		#pragma unroll
		for(int i = 0; i < (ITS + REGS - 1) / REGS; ++i)
		{
			#pragma unroll
			for(int r = 0; r < REGS; ++r)
				values[r] += values[r] * values[r];
			__threadfence_block();
		}

		#pragma unroll
		for(int r = 0; r < REGS; ++r)
			BigData()[threadIdx.x + r * 32] = values[r];
	}
};

__global__ void maxConcurrentBlockEval()
{
	if (*maxConcurrentBlockEvalDone() != 0)
		return;

	if (threadIdx.x == 0)
		atomicAdd(maxConcurrentBlocks(), 1);

	DelayFMADS<10000, 32>::delay();
	__syncthreads();

	*maxConcurrentBlockEvalDone() = 1;
	__threadfence();
}

std::mutex gpuMutex;

void GPU::initGPU()
{
	if (!gpu.get())
	{
		gpuMutex.lock();
		if (!gpu.get())	gpu.reset(new GPU());
		gpuMutex.unlock();
	}
}

bool GPU::isAvailable()
{
	initGPU();

	return (gpu->ngpus > 0);
}

int GPU::getBlockSize()
{
	return 128;
}

int GPU::getBlockCount()
{
	initGPU();

	return gpu->nblocks;
}

int GPU::getSMCount()
{
	initGPU();

	return gpu->nsms;
}

int GPU::getConstMemSize()
{
	initGPU();

	return gpu->szcmem;
}

int GPU::getSharedMemSizePerSM()
{
	initGPU();

	return gpu->szshmem;
}

void* GPU::malloc(size_t size)
{
#define MALLOC_ALIGNMENT 256

	initGPU();

	if (!gpu->gmem) return NULL;

	if (gpu->ptr + size + MALLOC_ALIGNMENT > gpu->gmem + gpu->szgmem)
		return NULL;
	
	void* result = gpu->ptr;
	gpu->ptr += size;
	
	ptrdiff_t alignment = (ptrdiff_t)gpu->ptr % MALLOC_ALIGNMENT;
	if (alignment)
		gpu->ptr += MALLOC_ALIGNMENT - alignment;
	
	return result;
}

// Reset free memory pointer to the beginning of preallocated buffer.
void GPU::mfree()
{
	initGPU();

	gpu->ptr = gpu->gmem;
}

// Check whether the specified memory address belongs to GPU memory allocation.
bool GPU::isAllocatedOnGPU(void* ptr)
{
	initGPU();
	
	if (!gpu->gmem) return false;

	if ((ptr >= gpu->gmem) && (ptr <= gpu->gmem + gpu->szgmem))
		return true;
	
	return false;
}

GPU::GPU() : ngpus(0), gmem(NULL)
{
	cudaError_t cudaError = cudaGetDeviceCount(&ngpus);
	if (cudaError != cudaErrorNoDevice)
		CUDA_ERR_CHECK(cudaError);

	if (!ngpus) return;

	maxConcurrentBlockEval<<<1024, getBlockSize()>>>();
	
	CUDA_ERR_CHECK(cudaGetLastError());
	CUDA_ERR_CHECK(cudaDeviceSynchronize());

	CUDA_ERR_CHECK(cudaMemcpyFromSymbol(&nblocks, maxConcurrentBlocksVar, sizeof(int)));
	
	cudaDeviceProp props;
	CUDA_ERR_CHECK(cudaGetDeviceProperties(&props, 0));

	nsms = props.multiProcessorCount;

	szcmem = props.totalConstMem;
	
	szshmem = props.sharedMemPerMultiprocessor;

	cout << "Using GPU " << props.name << " : max concurrent blocks = " << nblocks <<
		" : " << min(szshmem, (nsms * szshmem) / nblocks) << "B of shmem per block" << endl;

	// Preallocate 85% of GPU memory to save on costly subsequent allocations.
	size_t available, total;
    cudaMemGetInfo(&available, &total);

	szgmem = 0.85 * available;

	CUDA_ERR_CHECK(cudaMalloc(&gmem, szgmem));
	ptr = gmem;
}

unique_ptr<GPU> GPU::gpu;

