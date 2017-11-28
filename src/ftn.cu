#define CUDA_ERR_CHECK(x)                                  \
    do { cudaError_t err = x; if (err != cudaSuccess) {    \
        fprintf(stderr, "CUDA error %d \"%s\" at %s:%d\n", \
            (int)err, cudaGetErrorString(err),             \
            __FILE__, __LINE__);                           \
        abort();                                           \
    }} while (0)

#include <limits>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <algorithm>
#include <iostream>
#include <cmath>

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
	double operator()(const double& t, const double& s) const
	{
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
		return -s0 + s;
	}
};

class AllocationReuse
{
	char* ptr;

public:
    // just allocate bytes
    typedef char value_type;

	template<typename T>
	AllocationReuse(thrust::device_vector<T>& v) : ptr((char*)thrust::raw_pointer_cast(v.data())) { }
 
    char* allocate(std::ptrdiff_t num_bytes)
    {
    	return ptr;
    } 

    void deallocate(char* ptr, size_t n) { }
};

namespace ftn
{
	long long length = 0;
	double* tt = NULL;
	int* s0 = NULL;
}

extern "C" void ftn_set_length(long long* length_)
{
	ftn::length = *length_;
}

extern "C" void ftn_set_tt(double* tt_)
{
	ftn::tt = tt_;
}

extern "C" void ftn_set_s0(int* s0_)
{
	ftn::s0 = s0_;
}

extern "C" void ftn_eval(const double* x, double* result_)
{
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
	
	#pragma omp parallel for reduction(+:result) schedule(dynamic, 100)
	for (long long i = 0; i < ftn::length; i++)
	{
		const double t = ftn::tt[i];
		double val = ((x[4] + x[5] * t) * cos(x[0] + x[1] * t + x[2] * t * t + x[3] * t * t * t) - ftn::s0[i]);
		result += val * val;
	}

	result = sqrt(result / ftn::length);
	*result_ = result;
}

using namespace std;

int main_(void)
{
	int n = 1000000;
	double t_factor = 1.0 / n;
	double c0 = 1.0, c1 = 2.0, c2 = 3.0, c3 = 4.0, c4 = 1.0, c5 = 6.0;

	thrust::host_vector<double> h_t(n);
	thrust::device_vector<double> d_t(n);

	// 序列生成 (time series)
	thrust::sequence(d_t.begin(), d_t.end());
	thrust::transform(d_t.begin(), d_t.end(), d_t.begin(), ScaleTransform(t_factor));

	// 仿真序列(simulation signal)
	for (int i = 0; i < n; ++i)
		h_t[i] = rand() / (double)RAND_MAX;

	thrust::device_vector<double> d_s = h_t;
	
	clock_t t1 = clock();
	{
		thrust::transform(d_t.begin(), d_t.end(), d_s.begin(), d_s.begin(),
			LinearTransform(c0, c1, c2, c3, c4, c5));
		CUDA_ERR_CHECK(cudaDeviceSynchronize());
	}

	double norm;
	clock_t t2 = clock();
	{
		// Use free d_t vector as a scratch space for reduction.
		// This way we same time on additional call to cudaMalloc,
		// which Thrust uses to allocate scratch space by default.
		AllocationReuse reuse(d_t);

		// 平方 (这里把平方和求和分开了)
		norm = sqrt(thrust::transform_reduce(thrust::cuda::par(reuse),
			d_s.begin(), d_s.end(), SquareTransform(), 0.0, thrust::plus<double>()));
	}
	clock_t t3 = clock();

	cout << "平方 (linear opration) : " << (double)(t2 - t1) * 1000.0 / CLOCKS_PER_SEC << " ms" << endl;
	cout << "求和 (reduction) : " << (double)(t3 - t2) * 1000.0 / CLOCKS_PER_SEC << " ms" << endl;
	cout.precision(std::numeric_limits<double>::max_digits10 + 1);
	cout << "范数 (norm) : " << norm << endl;

	return 0;
}
