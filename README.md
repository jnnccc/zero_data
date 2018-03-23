# Data fitting Doppler data processor with OpenMP and CUDA GPU backends

kernel algrithm : http://www1.icsi.berkeley.edu/~storn/code.html#csou

## Author
Nianchuan Jian and Dmitry Mikushin

## Prerequisites

```
$ sudo apt-get install pgplot5
```

## Install and run 

$ make -j24
$ cd bin
$ ./phase_track

## Release notes

This version uses the new makefile system, which is not bound to Intel compiler. The package could be compiled with any modern set of GNU C/C++11 and Fortran compilers.

This version uses Fortran version of libspice for data I/O: https://sourceforge.net/projects/ngspice/files/ng-spice-rework/27/
The source code of necessary function is shipped together with this package, no external libraries are needed.

Function evaluation can use either CPU (OpenMP) implementation or GPU (CUDA/Thrust). On systems with NVIDIA GPU available, evaulation will use GPU version by default. In order to enforce the use of CPU (OpenMP) version on GPU-enabled machine, disable GPUs visibility:

```
$ CUDA_VISIBLE_DEVICES= ./phase_check
```

GPU version is set to build for SM 3.5 (Kepler) and 6.1 (Pascal) GPUs. In order to add any other compute capability, locate the following lines in the `Makefile` and adjust them accordingly:

```
GPUARCH += -gencode=arch=compute_35,code=sm_35 # Kepler GK110 CC 3.5
GPUARCH += -gencode=arch=compute_61,code=sm_61 # Pascal GP106 CC 6.1
```

