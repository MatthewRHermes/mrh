
# Definition of MACROS

BINROOT=./
EXE=libgpu.so
SHELL=/bin/sh
CXX = g++
CXXFLAGS=
FC=gfortran
FCFLAGS=
LD = $(CXX)
LDFLAGS = -fPIC -shared
AR = ar rcs
CPP = cpp -P -traditional
INSTALL=../

CUDA_CXX = $(CXX)
CUDA_CXXFLAGS = 

ARCH ?= polaris-gnu-nvcc
include arch/$(ARCH)

# couple more flags to build shared library (allows arch files to be used for general app building)
CXXFLAGS += -fPIC
CUDA_CXXFLAGS += -shared -Xcompiler -fPIC
LDFLAGS += -fPIC -shared

# -- subset of src files with cuda kernels (need nvcc compiler)
CUDA_SRC = pm/pm_cuda.cpp pm/device_cuda.cpp
CUDA_INC = $(CUDA_SRC:.cpp=.h)
TMP0 = $(CUDA_SRC:pm/%=%)
CUDA_OBJ = $(TMP0:.cpp=.o)

# -- subset of src files in src directory
C_SRC = $(wildcard *.cpp)
C_INC = $(wildcard *.h)
C_OBJ = $(C_SRC:.cpp=.o)

# -- subset of src files in src/pm directory
PM_SRC = $(filter-out $(CUDA_SRC), $(wildcard pm/*.cpp))
PM_INC = $(filter-out $(CUDA_INC), $(wildcard pm/*.h))
TMP1 = $(PM_SRC:pm/%=%)
PM_OBJ = $(TMP1:.cpp=.o)

# -- subset of src files in src/mathlib directory
ML_SRC = $(wildcard mathlib/*.cpp)
ML_INC = $(wildcard mathlib/*.h)
TMP2 = $(ML_SRC:mathlib/%=%)
ML_OBJ = $(TMP2:.cpp=.o)

#$(info CUDA_SRC is [${CUDA_SRC}])
#$(info CUDA_INC is [${CUDA_INC}])
#$(info CUDA_OBJ is [${CUDA_OBJ}])

#$(info C_SRC is [${C_SRC}])
#$(info C_INC is [${C_INC}])
#$(info C_OBJ is [${C_OBJ}])

#$(info PM_SRC is [${PM_SRC}])
#$(info PM_INC is [${PM_INC}])
#$(info PM_OBJ is [${PM_OBJ}])

#$(info ML_SRC is [${ML_SRC}])
#$(info ML_INC is [${ML_INC}])
#$(info ML_OBJ is [${ML_OBJ}])

# -- target : 	Dependencies
# --		Rule to create target

$(EXE): $(C_OBJ) $(CUDA_OBJ) $(PM_OBJ) $(ML_OBJ)
	$(LD) $(LDFLAGS) -o $@ $(C_OBJ) $(CUDA_OBJ) $(PM_OBJ) $(ML_OBJ) $(LIB)

install: $(EXE)
	 cp $(EXE) $(INSTALL)

####################################################################

$(C_OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(PM_OBJ): %.o: pm/%.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(ML_OBJ): %.o: mathlib/%.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(CUDA_OBJ): %.o: pm/%.cpp
	$(CUDA_CXX) $(CUDA_CXXFLAGS) -c $< -o $@
	#$(CUDA_CXX) -x cu $(CUDA_CXXFLAGS) -c $< -o $@

#
# -- Remove *.o and *~ from the directory
clean:
	rm -f *.o *~ ./$(EXE)
	rm -f $(INSTALL)/$(EXE)
	rm -rf $(EXE).dSYM
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f *.o *~ ./$(EXE)
	rm -f $(INSTALL)/$(EXE)
	rm -rf $(EXE).dSYM
	rm -f *.optrpt

#
# -- Simple dependencies

libgpu.o : libgpu.cpp libgpu.h

pm_cuda.o : pm/pm_cuda.cpp pm/pm_cuda.h pm/pm.h
pm_sycl.o : pm/pm_sycl.cpp pm/pm_sycl.h pm/pm.h
pm_hip.o : pm/pm_hip.cpp pm/pm_hip.h pm/pm.h
pm_openmp.o : pm/pm_openmp.cpp pm/pm_openmp.h pm/pm.h
pm_host.o : pm/pm_host.cpp pm/pm_host.h pm/pm.h

device.o : device.cpp device.h pm/device_cuda.cpp pm/device_sycl.cpp pm/device_hip.cpp pm/device_openmp.cpp pm/device_host.cpp
device_cuda.o : pm/device_cuda.cpp device.h pm/pm_cuda.o
device_sycl.o : pm/device_sycl.cpp device.h pm/pm_sycl.o
device_hip.o : pm/device_hip.cpp device.h pm/pm_hip.o
device_openmp.o : pm/device_openmp.cpp device.h pm/pm_openmp.o
device_host.o : pm/device_host.cpp device.h pm/pm_host.o

mathlib_cublas.o : mathlib/mathlib_cublas.cpp mathlib/mathlib_cublas.h mathlib/mathlib.h pm/pm_cuda.o
mathlib_mkl.o : mathlib/mathlib_mkl.cpp mathlib/mathlib_mkl.h mathlib/mathlib.h pm/pm_cuda.o pm/pm_sycl.o
mathlib_rocm.o : mathlib/mathlib_rocm.cpp mathlib/mathlib_rocm.h mathlib/mathlib.h pm/pm_hip.o
mathlib_host.o : mathlib/mathlib_host.cpp mathlib/mathlib_host.h mathlib/mathlib.h pm/pm_host.o
