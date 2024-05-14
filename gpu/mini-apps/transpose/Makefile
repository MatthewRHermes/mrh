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

ARCH ?= polaris
include ../../src/arch/$(ARCH)

CXXFLAGS += -I../../src

$(info ARCH is [${ARCH}])

CSRC = $(wildcard *.cpp)
INC = $(wildcard *.h)
COBJ = $(CSRC:.cpp=.o)

FSRC = $(wildcard *.F90)
MOD = $(FSRC:.F90=.mod)
FOBJ = $(FSRC:.F90=.o)

#$(info CSRC is [${CSRC}])
#$(info INC is [${INC}])
#$(info COBJ is [${COBJ}])

#$(info FSRC is [${FSRC}])
#$(info MOD is [${MOD}])
#$(info FOBJ is [${FOBJ}])


#
# -- target : 	Dependencies
# --		Rule to create target

$(EXE):	$(COBJ) $(FOBJ) $(MOD)
	$(LD) $(LDFLAGS) -o $@ $(COBJ) $(LIB)

install: $(EXE)
	cp $(EXE) $(INSTALL)
#	cp $(MOD) $(FOBJ) $(INSTALL)/include

####################################################################

$(COBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(FOBJ): %.o: %.F90
	$(FC) $(FCFLAGS) -c $<

$(MOD): %.mod: %.F90
	$(FC) $(FCFLAGS) -c $<

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
pm_host.o : pm_host.cpp pm_host.h pm.h
pm_openmp.o : pm_openmp.cpp pm_openmp.h pm.h

device_cuda.o : device_cuda.cpp device.h
device_host.o : device_host.cpp device.h
device_openmp.o : device_openmp.cpp device.h
device.o : device.cpp device.h
