
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
ifneq ($(findstring _USE_CPU,$(CXXFLAGS)),_USE_CPU)
CUDA_CXXFLAGS += -shared -Xcompiler -fPIC
endif
LDFLAGS += -fPIC -shared

# compiler-generated header dependencies (see -include $(DEPS) below)
# note: nvcc accepts -MMD but not -MP, so CUDA gets -MMD only
DEPFLAGS = -MMD -MP
CXXFLAGS += $(DEPFLAGS)

# one-off flags without respelling the arch file, e.g. to enable the MATHLIB call
# profile for a single build:
#   make ARCH=aurora EXTRA_CXXFLAGS=-D_PROFILE_ML install
# NB: `make CXXFLAGS+=-D_PROFILE_ML` does NOT work -- a command-line assignment
# overrides every makefile assignment to that variable, so the arch flags and the
# -fPIC/-MMD added here would all be dropped.
# plain '=' (not '?=') so a stray EXTRA_CXXFLAGS exported in the caller's
# environment cannot leak into the build; the command line still overrides.
EXTRA_CXXFLAGS      =
EXTRA_CUDA_CXXFLAGS =

CXXFLAGS      += $(EXTRA_CXXFLAGS)
CUDA_CXXFLAGS += $(EXTRA_CUDA_CXXFLAGS)
CUDA_CXXFLAGS += -MMD

# -- subset of src files with cuda kernels (need nvcc compiler)
CUDA_SRC = pm/cuda/pm_cuda.cpp pm/cuda/jk.cpp pm/cuda/ao2mo.cpp pm/cuda/h2eff.cpp pm/cuda/impham.cpp pm/cuda/pdft.cpp pm/cuda/fci.cpp pm/cuda/utilities.cpp
CUDA_INC = $(CUDA_SRC:.cpp=.h)
TMP0 = $(CUDA_SRC:pm/%=%)
CUDA_OBJ = $(TMP0:.cpp=.o)

# -- subset of src files in src/device directory
DEV_SRC = $(wildcard device/*.cpp)
DEV_INC = $(wildcard device/*.h)
TMPD = $(DEV_SRC:device/%=%)
DEV_OBJ = $(TMPD:.cpp=.o)

# -- subset of src files in src directory
C_SRC = $(wildcard *.cpp)
C_INC = $(wildcard *.h)
C_OBJ = $(C_SRC:.cpp=.o)

# -- subset of src files in src/pm directory (and per-platform subdirs)
PM_SRC = $(filter-out $(CUDA_SRC), $(wildcard pm/*.cpp) $(wildcard pm/sycl/*.cpp) $(wildcard pm/hip/*.cpp) $(wildcard pm/openmp/*.cpp) $(wildcard pm/host/*.cpp))
PM_INC = $(filter-out $(CUDA_INC), $(wildcard pm/*.h) $(wildcard pm/sycl/*.h) $(wildcard pm/hip/*.h) $(wildcard pm/openmp/*.h) $(wildcard pm/host/*.h))
TMP1 = $(PM_SRC:pm/%=%)
PM_OBJ = $(TMP1:.cpp=.o)

# -- subset of src files in src/mathlib directory
ML_SRC = $(wildcard mathlib/*.cpp)
ML_INC = $(wildcard mathlib/*.h)
TMP2 = $(ML_SRC:mathlib/%=%)
ML_OBJ = $(TMP2:.cpp=.o)

# -- aggregate of all objects; DEPS are the -MMD files the compiler emits next to each .o
OBJS = $(C_OBJ) $(DEV_OBJ) $(CUDA_OBJ) $(PM_OBJ) $(ML_OBJ)
DEPS = $(OBJS:.o=.d)

# -- target : 	Dependencies
# --		Rule to create target

$(EXE): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $(OBJS) $(LIB)

install: $(EXE)
	 cp $(EXE) $(INSTALL)

####################################################################

$(C_OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(DEV_OBJ): %.o: device/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(PM_OBJ): %.o: pm/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ML_OBJ): %.o: mathlib/%.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(CUDA_OBJ): %.o: pm/%.cpp
	@mkdir -p $(dir $@)
	$(CUDA_CXX) $(CUDA_CXXFLAGS) -c $< -o $@

# -- compiler-generated dependencies (replaces the hand-maintained list below)
-include $(DEPS)

#
# -- Remove *.o and *~ from the directory
clean:
	rm -f $(OBJS) $(DEPS) *~ ./$(EXE)
	find device cuda sycl hip openmp host -name '*.o' -delete 2>/dev/null || true
	rm -f $(INSTALL)/$(EXE)
	rm -rf $(EXE).dSYM
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f $(OBJS) $(DEPS) *~ ./$(EXE)
	find device cuda sycl hip openmp host -name '*.o' -delete 2>/dev/null || true
	rm -f $(INSTALL)/$(EXE)
	rm -rf $(EXE).dSYM
	rm -f *.optrpt

