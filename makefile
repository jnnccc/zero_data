all: acc_data_regress

acc_data_regress: acc_data_regress.cu
	nvcc -std=c++11 -arch=sm_61 -g -O3 $< -o $@

clean:
	rm -rf acc_data_regress

