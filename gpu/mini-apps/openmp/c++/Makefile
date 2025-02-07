# Use NVIDIA compilers w/ ALCF provided OpenMPI
#
# module load nvhpc-nompi
#
# Definition of MACROS

PATH_TO_NVCC = $(shell which nvcc)
PATH_TO_NVHPC = $(shell echo ${PATH_TO_NVCC} | rev | cut -d '/' -f 4- | rev)

$(info PATH_TO_NVHPC= [${PATH_TO_NVHPC}])

CUDA = ${PATH_TO_NVHPC}/cuda

PATH_TO_PYTHON=$(shell readlink -f `which python` | rev | cut -d '/' -f 2- | rev)

$(info PATH_TO_PYTHON= [$(PATH_TO_PYTHON)])

PYTHON_INC=$(shell python -m pybind11 --includes)
PYTHON_LIB=$(shell $(PATH_TO_PYTHON)/python3-config --ldflags)
PYTHON_LIB+=-lpython3

CXX = CC
CXXFLAGS = -g -O3 -std=c++0x
CXXFLAGS += -D_SINGLE_PRECISION
CXXFLAGS += -mp=gpu -gpu=cc80,cuda11.0

CXXFLAGS += $(PYTHON_INC)
CXXFLAGS += -I../../src -D_USE_GPU -D_GPU_OPENMP

CUDA_CXX = $(CXX)
CUDA_CXXFLAGS = $(CXXFLAGS)

CPP = cpp -P -traditional
CPPFLAGS =

LD = $(CXX)
LIB = ../../src/libgpu.so $(PYTHON_LIB)
LIB += -L/home/knight/soft/polaris/lapack/lib -llapack -lrefblas -lgfortran  

BINROOT=./
EX=vecadd
SHELL=/bin/sh

# -- subset of src files with cuda kernels
CUDA_SRC = 
CUDA_OBJ = $(CUDA_SRC:.cpp=.o)

SRC = $(filter-out $(CUDA_SRC), $(wildcard *.cpp))
INC = $(wildcard *.h)
OBJ = $(SRC:.cpp=.o)

#
# -- target : 	Dependencies
# --		Rule to create target

$(EX): 	$(OBJ) $(CUDA_OBJ)
	$(LD) -o $@ $(CXXFLAGS) $(OBJ) $(CUDA_OBJ) $(LIB)

####################################################################

$(OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(CUDA_OBJ): %.o: %.cpp
	$(CUDA_CXX) -x cu $(CUDA_CXXFLAGS) -c $< -o $@

#
# -- Remove *.o and *~ from the directory
clean:
	rm -f *.o *~
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f *.o *~ ./$(EX)

#
# -- Simple dependencies

$(OBJ) : $(INC)
$(CUDA_OBJ) : $(INC)
