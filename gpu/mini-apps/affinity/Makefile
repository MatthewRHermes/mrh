
# Definition of MACROS

BASE_LIBGPU=../../src

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

ARCH ?= polaris
include $(BASE_LIBGPU)/arch/$(ARCH)

# -- subset of src files with cuda kernels (need nvcc compiler)
CUDA_SRC = pm/cuda/pm_cuda.cpp
CUDA_INC = $(CUDA_SRC:.cpp=.h)
TMP0 = $(CUDA_SRC:pm/%=%)
CUDA_OBJ = $(TMP0:.cpp=.o)

# -- subset of src files in src directory
C_SRC = $(wildcard *.cpp)
C_INC = $(wildcard *.h)
C_OBJ = $(C_SRC:.cpp=.o)

# -- subset of src files in src/pm directory (and per-platform subdirs)
PM_SRC = $(filter-out $(CUDA_SRC), $(wildcard pm/sycl/pm_sycl.cpp) $(wildcard pm/hip/pm_hip.cpp) $(wildcard pm/openmp/pm_openmp.cpp) $(wildcard pm/host/pm_host.cpp))
PM_INC = $(filter-out $(CUDA_INC), $(wildcard pm/sycl/pm_sycl.h) $(wildcard pm/hip/pm_hip.h) $(wildcard pm/openmp/pm_openmp.h) $(wildcard pm/host/pm_host.h))
TMP1 = $(PM_SRC:pm/%=%)
PM_OBJ = $(TMP1:.cpp=.o)

#$(info CUDA_SRC is [${CUDA_SRC}])
#$(info CUDA_INC is [${CUDA_INC}])
#$(info CUDA_OBJ is [${CUDA_OBJ}])

#$(info C_SRC is [${C_SRC}])
#$(info C_INC is [${C_INC}])
#$(info C_OBJ is [${C_OBJ}])

#$(info PM_SRC is [${PM_SRC}])
#$(info PM_INC is [${PM_INC}])
#$(info PM_OBJ is [${PM_OBJ}])

# -- target : 	Dependencies
# --		Rule to create target
$(EXE): $(C_OBJ) $(CUDA_OBJ) $(PM_OBJ) 
	$(LD) $(LDFLAGS) -o $@ $(C_OBJ) $(CUDA_OBJ) $(PM_OBJ) $(LIB)

update:
	cp -r ../../src/pm ./
	rm -f ./pm/device_*

####################################################################

$(C_OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(CUDA_OBJ): %.o: pm/%.cpp
	@mkdir -p $(dir $@)
	$(CUDA_CXX) $(CUDA_CXXFLAGS) -c $< -o $@
	#$(CUDA_CXX) -x cu $(CUDA_CXXFLAGS) -c $< -o $@

$(PM_OBJ): %.o: pm/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

#
# -- Remove *.o and *~ from the directory
clean:
	rm -f *.o *~ ./$(EXE)
	rm -f $(EXE)
	rm -rf $(EXE).dSYM
	find cuda sycl hip openmp host -name '*.o' -delete 2>/dev/null || true
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f *.o *~ ./$(EXE)
	rm -f $(EXE)
	rm -rf $(EXE).dSYM
	rm -f *.optrpt
	rm -rf pm
	rm -rf cuda sycl hip openmp host

#
# -- Simple dependencies

cuda/pm_cuda.o : pm/cuda/pm_cuda.cpp pm/cuda/pm_cuda.h pm/pm.h
sycl/pm_sycl.o : pm/sycl/pm_sycl.cpp pm/sycl/pm_sycl.h pm/pm.h
hip/pm_hip.o : pm/hip/pm_hip.cpp pm/hip/pm_hip.h pm/pm.h
openmp/pm_openmp.o : pm/openmp/pm_openmp.cpp pm/openmp/pm_openmp.h pm/pm.h
host/pm_host.o : pm/host/pm_host.cpp pm/host/pm_host.h pm/pm.h
