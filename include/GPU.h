#ifndef GPU_H
#define GPU_H

#include <memory>

#if !defined(__device__)
#define __device__
#endif
#if !defined(__host__)
#define __host__
#endif

class GPU
{
	static std::unique_ptr<GPU> gpu;

	int ngpus;
	int nblocks;
	int nsms;
	int szcmem;
	int szshmem;
	size_t szgmem;
	
	char *gmem, *ptr;

	GPU();

	static void initGPU();

public :

	static bool isAvailable();
	
	static int getBlockSize();
	
	static int getBlockCount();
	
	static int getSMCount();
	
	static int getConstMemSize();
	
	static int getSharedMemSizePerSM();
	
	// Allocate global memory from the preallocated buffer.
	static void* malloc(size_t size);
	
	// Reset free memory pointer to the beginning of preallocated buffer.
	static void mfree();
	
	// Check whether the specified memory address belongs to GPU memory allocation.
	static bool isAllocatedOnGPU(void* ptr);
};

#endif // GPU_H

