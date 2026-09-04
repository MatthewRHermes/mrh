cmake_minimum_required(VERSION 3.19)

# Use the MPI compiler driver as the C++ compiler, mirroring the Makefile
# (mpicxx), which brings in the right runtime + search paths for this project.
# Must be set before project() so CMake detects the compiler correctly.
find_program(_LIBGPU_CXX NAMES mpicxx-mpich-gcc14 mpicxx)
if(NOT _LIBGPU_CXX)
  set(_LIBGPU_CXX "$ENV{LIBGPU_CXX}")
endif()
if(_LIBGPU_CXX)
  set(CMAKE_CXX_COMPILER "${_LIBGPU_CXX}" CACHE FILEPATH "C++ compiler driver")
endif()

project(libgpu LANGUAGES CXX)

# ---------------------------------------------------------------------------
# libgpu: CMake build for the CPU backends (pm/openmp + pm/host).
# Mirrors `make ARCH=mac` in this directory (see arch/mac). Configure:
#     cmake -S . -B build
#     cmake --build build -j
#     cmake --install build
# Useful overrides:
#     -DCMAKE_CXX_COMPILER=/path/to/mpicxx      compiler driver
#     -DCMAKE_BUILD_TYPE=Release|Debug          optimization
#     -DLIBGPU_INSTALL_DIR=/some/dir            where `cmake --install` copies libgpu.so
# ---------------------------------------------------------------------------

# Install destination: same location the Makefile's INSTALL macro points to.
set(LIBGPU_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../my_pyscf/gpu"
    CACHE PATH "Directory to install libgpu.so into")

# --- MPI / OpenMP ----------------------------------------------------------
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# --- Python + pybind11 -----------------------------------------------------
# The Makefile discovers pybind11 via `python3 -m pybind11 --includes`; the
# pybind11 pip package may live in a user site-packages, so ask the interpreter
# directly instead of find_package(pybind11). Default to python3.12 (matching
# the Makefile) but respect -DPython_EXECUTABLE=... overrides.
find_program(_LIBGPU_PYTHON3_12 NAMES python3.12)
if(_LIBGPU_PYTHON3_12 AND NOT Python_EXECUTABLE)
  set(Python_EXECUTABLE "${_LIBGPU_PYTHON3_12}" CACHE FILEPATH "Python interpreter")
endif()
find_package(Python COMPONENTS Interpreter)
if(NOT Python_Interpreter_FOUND)
  message(FATAL_ERROR
    "Python interpreter not found; set -DPython_EXECUTABLE=/path/to/python3.12")
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --includes
  OUTPUT_VARIABLE _LIBGPU_PYBIND11_INCLUDES
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE _LIBGPU_PYBIND11_ERR
  RESULT_VARIABLE _LIBGPU_PYBIND11_RC)
if(NOT _LIBGPU_PYBIND11_RC EQUAL 0)
  message(FATAL_ERROR
    "`${Python_EXECUTABLE} -m pybind11` failed: ${_LIBGPU_PYBIND11_ERR}\n"
    "Set -DPython_EXECUTABLE=/path/to/python (with pybind11 installed)")
endif()
separate_arguments(_LIBGPU_PYBIND11_INCLUDES NATIVE_COMMAND "${_LIBGPU_PYBIND11_INCLUDES}")
list(TRANSFORM _LIBGPU_PYBIND11_INCLUDES REPLACE "^(-I|-isystem)" "")
message(STATUS "pybind11 via ${Python_EXECUTABLE}: ${_LIBGPU_PYBIND11_INCLUDES}")

# --- Sources (CPU backends only; pm/cuda, pm/sycl, pm/hip are GPU-only) ------
file(GLOB LIBGPU_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/libgpu.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/device/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/pm/openmp/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/pm/host/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/mathlib/mathlib_host.cpp")

add_library(libgpu SHARED ${LIBGPU_SOURCES})

target_compile_definitions(libgpu PRIVATE _USE_CPU)
target_compile_options(libgpu PRIVATE -g -O3)
target_include_directories(libgpu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${_LIBGPU_PYBIND11_INCLUDES})

target_link_libraries(libgpu PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX)

set_target_properties(libgpu PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX ""
    SUFFIX ".so")

if(APPLE)
  # pybind11 shared libs must keep undefined symbols (resolved by Python at load)
  target_link_options(libgpu PRIVATE "-undefined" "dynamic_lookup")
  find_library(_LIBGPU_ACCELERATE Accelerate)
  if(_LIBGPU_ACCELERATE)
    target_link_libraries(libgpu PRIVATE "${_LIBGPU_ACCELERATE}")
  endif()
endif()

install(TARGETS libgpu DESTINATION "${LIBGPU_INSTALL_DIR}")
