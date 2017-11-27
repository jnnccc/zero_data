DEBUG =

ifeq (,$(DEBUG))
CXXFLAGS = -std=c++11 -g -O3 -ffast-math -fopenmp -fdata-sections -ffunction-sections
F90FLAGS = -g -O3 -ffast-math -fopenmp -Jobj/ -ffree-line-length-none -fdata-sections -ffunction-sections
FCFLAGS = -g -O3 -ffast-math -Iinclude/ -fdata-sections -ffunction-sections
LDFLAGS = -Xcompiler=-fopenmp -Xlinker=--gc-sections
NVCCFLAGS = -std=c++11 -g -O3 -Xcompiler=-ffast-math -Xcompiler=-fopenmp -Xcompiler=-fdata-sections -Xcompiler=-ffunction-sections
else
CXXFLAGS = -std=c++11 -g -O0
F90FLAGS = -g -O0 -Jobj/ -ffree-line-length-none
FCFLAGS = -g -O0 -Iinclude/
LDFLAGS =
NVCCFLAGS = -std=c++11 -g -O0 -G
endif

GPUARCH = 35 # Kepler CC 3.5

.SUFFIXES: .cpp .f .f90 .cu .cpp.o .f.o .f90.o .cu.o

all: bin/phase_track

SRCS := $(wildcard src/*.cpp) $(wildcard src/*.f) $(wildcard src/*.f90) $(wildcard src/*.cu)

OBJS := $(addsuffix .o, $(patsubst src/%,obj/%,$(SRCS)))

bin/phase_track: $(OBJS)
	mkdir -p bin && nvcc $(LDFLAGS) $^ -o $@ -lgfortran -lpgplot

obj/DE_Fortran90.f90.o: obj/param.f90.o 

obj/math_lib.f90.o: obj/lib_array.f90.o

obj/module10.f90.o: obj/lib_array.f90.o obj/math_lib.f90.o obj/param.f90.o obj/plot_module.f90.o obj/utilmod.f90.o 

obj/phase.f90.o: obj/module10.f90.o 

obj/read_driv.f90.o: obj/param.f90.o 

obj/utilmod.f90.o: obj/param.f90.o 

obj/%.cpp.o: src/%.cpp
	mkdir -p obj && g++ $(CXXFLAGS) -c $< -o $@

obj/%.f90.o: src/%.f90
	mkdir -p obj && gfortran $(F90FLAGS) -c $< -o $@

obj/%.f.o: src/%.f
	mkdir -p obj && gfortran $(FCFLAGS) -c $< -o $@

obj/%.cu.o: src/%.cu
	mkdir -p obj && nvcc -arch=sm_$(GPUARCH) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf bin obj

