#
# Definition of MACROS

BASE_LIBGPU=../../src

BINROOT=./
EXE=a.out
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
include $(BASE_LIBGPU)/arch/$(ARCH)

CXXFLAGS += -I$(BASE_LIBGPU)

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

# -- only copy source files; headers referenced with compiler flag
$(shell cp ../../src/pm*.cpp ./)

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
	rm -f pm*.cpp
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f *.o *.mod *~ $(EXE)
	rm -f $(INSTALL)/$(EXE)
	rm -rf $(EXE).dSYM
	rm -f *.optrpt
	rm -f pm*.h pm*.cpp

#
# -- Simple dependencies
