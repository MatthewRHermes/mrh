
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

ARCH ?= polaris-gnu-nvcc
include $(BASE_LIBGPU)/arch/$(ARCH)

# -- subset of src files with cuda kernels (need nvcc compiler)
CUDA_SRC = pm/pm_cuda.cpp
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
	$(CUDA_CXX) $(CUDA_CXXFLAGS) -c $< -o $@
	#$(CUDA_CXX) -x cu $(CUDA_CXXFLAGS) -c $< -o $@

$(PM_OBJ): %.o: pm/%.cpp
	$(CXX) $(CXXFLAGS) -c $<

#
# -- Remove *.o and *~ from the directory
clean:
	rm -f *.o *~ ./$(EXE)
	rm -f $(EXE)
	rm -rf $(EXE).dSYM
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f *.o *~ ./$(EXE)
	rm -f $(EXE)
	rm -rf $(EXE).dSYM
	rm -f *.optrpt
	rm -r pm

#
# -- Simple dependencies

pm_cuda.o : pm/pm_cuda.cpp pm/pm_cuda.h pm/pm.h
pm_sycl.o : pm/pm_sycl.cpp pm/pm_sycl.h pm/pm.h
pm_hip.o : pm/pm_hip.cpp pm/pm_hip.h pm/pm.h
pm_openmp.o : pm/pm_openmp.cpp pm/pm_openmp.h pm/pm.h
pm_host.o : pm/pm_host.cpp pm/pm_host.h pm/pm.h
