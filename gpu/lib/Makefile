#
# Definition of MACROS

BINROOT=./
EXE=libgpu.so
SHELL=/bin/sh
CXX = clang++-mp-13
CXXFLAGS=
FC=gfortran-mp-11
FCFLAGS=
LD = g++
LDFLAGS = -fPIC -shared
AR = ar rcs
CPP=cpp -P -traditional
INSTALL=../

ARCH ?= polaris
include arch/$(ARCH)

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
	cp $(EXE) $(INSTALL)/lib
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
	rm -f *.o *.mod *~ ./$(EXE)
	rm -rf $(EXE).dSYM
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f *.o *.mod *~ ./$(EXE)
	rm -rf $(EXE).dSYM
	rm -f *.optrpt

#
# -- Simple dependencies

#libgpu_fc_interface.mod : libgpu_fc_interface.F90
#libgpu_fc_interface.o : libgpu_fc_interface.F90

libgpu.o : libgpu.cpp
libgpu.o : libgpu.h

pm_cuda.o : pm_cuda.cpp
pm_cuda.o : pm_cuda.h

device_cuda.o : device_cuda.cpp
device_cuda.o : device.h
device.o : device.h
