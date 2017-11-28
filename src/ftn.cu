#include <cstdio>
#include <thrust/transform_reduce.h>
#include <cmath>

#include "check.h"
#include "GPU.h"

struct SquareTransform
{
	__attribute__((always_inline))
	__host__ __device__
	double operator()(const double& x) const
	{ 
		return x * x;
	}
};

struct ScaleTransform
{
	const double c;
	
	ScaleTransform(const double& c_) : c(c_) { }
	
	__attribute__((always_inline))
	__host__ __device__
	double operator()(const double& t) const
	{
		return c * t;
	}
};

struct LinearTransform
{
	const double c0, c1, c2, c3, c4, c5;
	
	LinearTransform(
		const double& c0_, const double& c1_, const double& c2_,
		const double& c3_, const double& c4_, const double& c5_) :
	
	c0(c0_), c1(c1_), c2(c2_), c3(c3_), c4(c4_), c5(c5_) { }

	__attribute__((always_inline))
	__host__ __device__
	double operator()(const thrust::tuple<double, int> pair) const
	{
		double t;
		int s;
		
		thrust::tie(t, s) = pair;
		
		// t^2序列
		double t2 = t * t;

		// t^3序列
		double t3 = t2 * t;

		// 线性操作1 & 线性操作2 & 线性操作3
		double c = c0 + c1 * t + c2 * t2 + c3 * t3;

		// 三角函数
		double s0 = cos(c);

		// 幅度函数
		double amp = c5 * t + c4;

		// 形成函数
		s0 *= amp;

		// O-C
		return (-s0 + s) * (-s0 + s);
	}
};

class ReductionScratchSpace
{
	char* ptr;

public :
    // just allocate bytes
    typedef char value_type;

	ReductionScratchSpace() { }

	template<typename T>
	ReductionScratchSpace(T* ptr_) : ptr((char*)ptr_) { }
 
    char* allocate(std::ptrdiff_t num_bytes)
    {
    	return ptr;
    } 

    void deallocate(char* ptr, size_t n) { }
};

namespace ftn
{
	long long length = 0;
	
	// CPU vectors.
	double* tt = NULL;
	int* s0 = NULL;

	// GPU vectors. Wrap into smart pointer, in order to
	// avoid possible initialization before GPU driver.
	double* d_tt = NULL;
	int* d_s0 = NULL;
	
	// Scratch space for Thrust reduction.
	double* d_scratch = NULL;

	// Use preallocated vector as a scratch space for reduction.
	// This way we same time on additional call to cudaMalloc,
	// which Thrust uses to allocate scratch space by default
	// each time it does reduction.
	ReductionScratchSpace scratchSpace;
}

extern "C" void ftn_set_length(long long* length_)
{
	ftn::length = *length_;
	
	if (GPU::isAvailable())
	{
		// Initialize GPU vectors.
		GPU::mfree();
		ftn::d_tt = (double*)GPU::malloc(ftn::length * sizeof(double));
		ftn::d_scratch = (double*)GPU::malloc(ftn::length * sizeof(double));
		ftn::d_s0 = (int*)GPU::malloc(ftn::length * sizeof(int));
		
		ftn::scratchSpace = ReductionScratchSpace(ftn::d_scratch);
	}
}

extern "C" void ftn_set_tt(double* tt_)
{
	ftn::tt = tt_;
	
	if (GPU::isAvailable())
	{
		// Fill GPU counterpart.
		CUDA_ERR_CHECK(cudaMemcpy(ftn::d_tt, ftn::tt,
			ftn::length * sizeof(double), cudaMemcpyHostToDevice));
	}
}

extern "C" void ftn_set_s0(int* s0_)
{
	ftn::s0 = s0_;

	if (GPU::isAvailable())
	{
		// Fill GPU counterpart.
		CUDA_ERR_CHECK(cudaMemcpy(ftn::d_s0, ftn::s0,
			ftn::length * sizeof(int), cudaMemcpyHostToDevice));
	}
}

extern "C" void ftn_eval(const double* x, double* result_)
{
	clock_t t1 = clock();

	if (!ftn::length)
	{
		fprintf(stderr, "ftn_eval: t and s0 vectors length was not properly set (ftn_set_length must be called first)\n");
		exit(-1);
	}
	if (!ftn::tt)
	{
		fprintf(stderr, "ftn_eval: t vector was not properly set (ftn_set_tt must be called first)\n");
		exit(-1);
	}
	if (!ftn::s0)
	{
		fprintf(stderr, "ftn_eval: s0 vector was not properly set (ftn_set_s0 must be called first)\n");
		exit(-1);
	}

	double result = 0;
	
	if (!GPU::isAvailable())
	{
		// Process on multicore CPU, if GPU is not available.
		#pragma omp parallel for reduction(+:result) schedule(dynamic, 100)
		for (long long i = 0; i < ftn::length; i++)
		{
			const double t = ftn::tt[i];
			double val = ((x[4] + x[5] * t) * cos(x[0] + x[1] * t + x[2] * t * t + x[3] * t * t * t) - ftn::s0[i]);
			result += val * val;
		}
	}
	else
	{
		result = thrust::transform_reduce(thrust::cuda::par(ftn::scratchSpace),
			thrust::make_zip_iterator(thrust::make_tuple(ftn::d_tt, ftn::d_s0)),
			thrust::make_zip_iterator(thrust::make_tuple(ftn::d_tt + ftn::length, ftn::d_s0 + ftn::length)),
			LinearTransform(x[0], x[1], x[2], x[3], x[4], x[5]), 0.0, thrust::plus<double>());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());
	}

	result = sqrt(result / ftn::length);
	*result_ = result;

	clock_t t2 = clock();

#if 0
	printf("平方 + 求和 (linear opration + reducion) : %f ms\n",
		(double)(t2 - t1) * 1000.0 / CLOCKS_PER_SEC);
#endif
}

