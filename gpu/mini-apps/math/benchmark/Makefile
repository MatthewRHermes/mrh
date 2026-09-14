
# Definition of MACROS

BASE_LIBGPU=../../../src

BINROOT=./
EXE=a.out
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
include $(BASE_LIBGPU)/arch/$(ARCH)

# one-off flags, e.g.  make ARCH=mac EXTRA_CXXFLAGS=-D_PROFILE_ML
# (`make CXXFLAGS+=...` would replace the arch flags, not append to them)
# plain '=' (not '?=') so a stray EXTRA_CXXFLAGS exported in the caller's
# environment cannot leak into the build; the command line still overrides.
EXTRA_CXXFLAGS      =
EXTRA_CUDA_CXXFLAGS =

CXXFLAGS      += $(EXTRA_CXXFLAGS)
CUDA_CXXFLAGS += $(EXTRA_CUDA_CXXFLAGS)

# -- subset of src files with cuda kernels (need nvcc compiler)
# -- since the device refactor the platform managers live in pm/<backend>/
CUDA_SRC = pm/cuda/pm_cuda.cpp
CUDA_INC = $(CUDA_SRC:.cpp=.h)
TMP0 = $(CUDA_SRC:pm/%=%)
CUDA_OBJ = $(TMP0:.cpp=.o)

# -- subset of src files in src directory
C_SRC = $(wildcard *.cpp)
C_INC = $(wildcard *.h)
C_OBJ = $(C_SRC:.cpp=.o)

# -- subset of src files in src/pm directory
# -- only the pm_<backend> platform managers: this mini-app links PM + MATHLIB and
# -- has no Device, so pm/<backend>/{jk,ao2mo,h2eff,impham,pdft,fci,utilities}.cpp
# -- and pm/mono/ are deliberately excluded (see the update target).
PM_SRC = $(filter-out $(CUDA_SRC), $(wildcard pm/*.cpp) $(wildcard pm/*/pm_*.cpp))
PM_INC = $(filter-out $(CUDA_INC), $(wildcard pm/*.h) $(wildcard pm/*/pm_*.h))
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

update:
	rm -rf ./pm ./mathlib
	cp -r $(BASE_LIBGPU)/pm ./
	cp -r $(BASE_LIBGPU)/mathlib ./
	rm -rf ./pm/mono
	find ./pm -name '*.cpp' ! -name 'pm_*.cpp' -delete

####################################################################

$(C_OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(CUDA_OBJ): %.o: pm/%.cpp
	@mkdir -p $(dir $@)
	$(CUDA_CXX) $(CUDA_CXXFLAGS) -c $< -o $@

$(PM_OBJ): %.o: pm/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ML_OBJ): %.o: mathlib/%.cpp
	$(CXX) $(CXXFLAGS) -c $<

#
# -- Remove *.o and *~ from the directory
clean:
	rm -f *.o *~ ./$(EXE)
	rm -f cuda/*.o hip/*.o sycl/*.o openmp/*.o host/*.o
	rm -f $(EXE)
	rm -rf $(EXE).dSYM
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f *.o *~ ./$(EXE)
	rm -rf cuda hip sycl openmp host
	rm -f $(EXE)
	rm -rf $(EXE).dSYM
	rm -f *.optrpt
	rm -rf pm
	rm -rf mathlib

#
# -- Simple dependencies

cuda/pm_cuda.o     : pm/cuda/pm_cuda.cpp     pm/cuda/pm_cuda.h     pm/pm.h
sycl/pm_sycl.o     : pm/sycl/pm_sycl.cpp     pm/sycl/pm_sycl.h     pm/pm.h
hip/pm_hip.o       : pm/hip/pm_hip.cpp       pm/hip/pm_hip.h       pm/pm.h
openmp/pm_openmp.o : pm/openmp/pm_openmp.cpp pm/openmp/pm_openmp.h pm/pm.h
host/pm_host.o     : pm/host/pm_host.cpp     pm/host/pm_host.h     pm/pm.h

mathlib_cublas.o  : mathlib/mathlib_cublas.cpp  mathlib/mathlib_cublas.h  mathlib/mathlib.h cuda/pm_cuda.o
mathlib_mkl.o     : mathlib/mathlib_mkl.cpp     mathlib/mathlib_mkl.h     mathlib/mathlib.h cuda/pm_cuda.o sycl/pm_sycl.o
mathlib_hipblas.o : mathlib/mathlib_hipblas.cpp mathlib/mathlib_hipblas.h mathlib/mathlib.h hip/pm_hip.o
mathlib_host.o    : mathlib/mathlib_host.cpp    mathlib/mathlib_host.h    mathlib/mathlib.h host/pm_host.o
