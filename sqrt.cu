#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <algorithm>
#include <iostream>
#include <cmath>
// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
	    __host__ __device__
			        T operator()(const T& x) const { 
						            return x * x;
									        }
};
//线性操作
struct saxpy_functor : public thrust::binary_function<double,double,double>
{
    const double a;

    saxpy_functor(double _a) : a(_a) {}

    __host__ __device__
        double operator()(const double& x, const double& y) const { 
            return a * x + y;
        }
};
void saxpy_fast(double A, thrust::device_vector<double>& X, thrust::device_vector<double>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}
//序列相乘
struct xy_functor : public thrust::binary_function<double,double,double>
{
    __host__ __device__
        double operator()(const double& x, const double& y) const { 
            return x * y;
        }
};
void xy_fast(thrust::device_vector<double>& X, thrust::device_vector<double>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), xy_functor());
}

//cos函数
struct cos_func {

__host__ __device__
  double operator()(double x){
    return cos(x);
  }
};
//序列平方
struct t2_func {
__host__ __device__
  double operator()(double x){
    return x*x;
  }
};
//序列立方
struct t3_func {

__host__ __device__
  double operator()(double x){
    return x*x*x;
  }
};



int main(void)
{
	int n=1000000;
	double t_factor=1.0/(double)n;
	double c0=1.0,c1=2.0,c2=3.0,c3=4.0,c4=1.0, c5=6.0;
	thrust::host_vector<double> h_t(n);
	thrust::device_vector<double> d_c0(n);
	thrust::device_vector<double> d_amp(n);
	thrust::device_vector<double> d_s0(n);
	thrust::device_vector<double> d_t(n);
	thrust::device_vector<double> d_t2(n);
	thrust::device_vector<double> d_t3(n);

//序列生成(time series)
	thrust::sequence(d_t.begin(), d_t.end());
	saxpy_fast(t_factor, d_t, d_t2);
	d_t=d_t2;
//	std::cout << d_t[10000] << std::endl;
//	exit (0);

//仿真序列(simulation signal)
    for (int i = 0; i < n; ++i)
    {
        h_t[i] = rand()/(double)RAND_MAX;
//		printf("%18.12f\n",h_t[i]);
    }
	thrust::device_vector<double> d_s = h_t;
	
	clock_t t1=clock();
//幅度序列
	thrust::fill(d_amp.begin(), d_amp.end(), c4);

//数传模拟(host到device，测试用)
//	d_t2=h_t;
//c0序列
	thrust::fill(d_c0.begin(), d_c0.end(), c0);
//t^2序列
	thrust::transform(d_t.begin(), d_t.end(), d_t2.begin(),t2_func());
//t^3序列
	thrust::transform(d_t.begin(), d_t.end(), d_t3.begin(),t3_func());
//线性操作1
	saxpy_fast(c1, d_t, d_c0);
//线性操作2
	saxpy_fast(c2, d_t2, d_c0);
//线性操作3
	saxpy_fast(c3, d_t3, d_c0);
//三角函数	
    thrust::transform(d_c0.begin(), d_c0.end(), d_s0.begin(),cos_func());
//幅度函数
	saxpy_fast(c5, d_t, d_amp);
//形成函数
	xy_fast(d_amp, d_s0);
//	thrust::copy(d_s.begin(), d_s.end(), h_t.begin());
//    for (int i = 0; i < n; ++i)
//    {
//		printf("%18.12f\n",h_t[i]);
//	}



//O-C
	saxpy_fast(-1.0, d_s0, d_s);

	clock_t t2=clock();
//	thrust::copy(d_s.begin(), d_s.end(), h_t.begin());
//    for (int i = 0; i < n; ++i)
//    {
//		printf("%18.12f\n",h_t[i]);
//	}

//平方(这里把平方和求和分开了)
//	xy_fast(d_s, d_s);
//	double norm = std::sqrt( thrust::reduce(d_s.begin(), d_s.end(), (double)0.0, thrust::plus<double>()) );

//norm
// setup arguments
    square<double>        unary_op;
    thrust::plus<double> binary_op;
    double init = 0.E0;
	double norm = std::sqrt( thrust::transform_reduce(d_s.begin(), d_s.end(), unary_op, init, binary_op) );
	clock_t t3=clock();
	std::cout << "平方(linear opration) : " << (double)(t2 - t1)*1000.0 / CLOCKS_PER_SEC << "ms" << std::endl;
	std::cout << "求和(reduction) : " << (double)(t3 - t2)*1000.0 / CLOCKS_PER_SEC << "ms" << std::endl;



//	thrust::copy(d_s0.begin(), d_s0.end(), h_s0.begin());
	printf("%18.12f\n",norm);
//    return 0;
}
