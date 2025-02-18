#
# Definition of MACROS

BINROOT=./
EXE=libgpu.so
SHELL=/bin/sh
CXX = g++
CXXFLAGS=
FC=gfortran
FCFLAGS=
LD = g++
LDFLAGS = -fPIC -shared
AR = ar rcs
CPP=cpp -P -traditional
INSTALL=../

ARCH ?= polaris-cpu-gnu
include arch/$(ARCH)

# couple more flags to build shared library (allows arch files to be used for general app building)
CXXFLAGS += -fPIC
CUDA_CXXFLAGS += -shared -Xcompiler -fPIC
LDFLAGS += -fPIC -shared

$(info ARCH is [${ARCH}])

CSRC = $(wildcard *.cpp) $(wildcard pm/*.cpp) $(wildcard mathlib/*.cpp)
INC = $(wildcard *.h) $(wildcard pm/*.h) $(wildcard mathlib/*.h)
COBJ = $(CSRC:.cpp=.o)

#$(info CSRC is [${CSRC}])
#$(info INC is [${INC}])
#$(info COBJ is [${COBJ}])

#
# -- target : 	Dependencies
# --		Rule to create target

$(EXE):	$(COBJ) 
	$(LD) $(LDFLAGS) -o $@ $(COBJ) $(LIB)

install: $(EXE)
	cp $(EXE) $(INSTALL)

####################################################################

$(COBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

#
# -- Remove *.o and *~ from the directory
clean:
	rm -f *.o *.mod *~ $(EXE)
	rm -f $(INSTALL)/$(EXE)
	rm -rf $(EXE).dSYM
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f *.o *.mod *~ $(EXE)
	rm -f $(INSTALL)/$(EXE)
	rm -rf $(EXE).dSYM
	rm -f *.optrpt

#
# -- Simple dependencies

libgpu.o : libgpu.cpp libgpu.h

pm_cuda.o : pm_cuda.cpp pm_cuda.h pm.h
pm_sycl.o : pm_sycl.cpp pm_sycl.h pm.h
pm_hip.o : pm_hip.cpp pm_hip.h pm.h
pm_openmp.o : pm_openmp.cpp pm_openmp.h pm.h
pm_host.o : pm_host.cpp pm_host.h pm.h

device.o : device.cpp device_* pm*
device_cuda.o : device_cuda.cpp device.h pm*
device_sycl.o : device_sycl.cpp device.h pm*
device_hip.o : devie_hip.cpp device.h pm*
device_openmp.o : device_openmp.cpp device.h pm*
device_host.o : device_host.cpp device.h pm*

mathlib_cublas.o : mathlib_cublas.cpp mathlib_cublas.h mathlib.h pm*
mathlib_mkl.o : mathlib_mkl.cpp mathlib_mkl.h mathlib.h pm*
mathlib_rocm.o : mathlib_rocm.cpp mathlib_rocm.h mathlib.h pm*
mathlib_host.o : mathlib_host.cpp mathlib_host.h mathlib.h pm*
