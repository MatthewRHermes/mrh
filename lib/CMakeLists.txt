# Modified from QC-DMET; see below
#project (clib_qcdmet)
#set (QC_DMET_VERSION      "0.1")
#set (QC_DMET_AUTHORS      "Sebastian Wouters")
#set (QC_DMET_DESCRIPTION  "QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry")
#set (QC_DMET_EMAIL        "sebastianwouters@gmail.com")
#set (QC_DMET_URL          "https://github.com/sebwouters/qc-dmet")
#set (QC_DMET_LICENSE      "GNU General Public License, version 2")

project (clib_mrh)
set (MRH_AUTHORS        "Matthew R. Hermes")
set (MRH_DESCRIPTION    "mrh: GPL research code of Matthew R. Hermes")
set (MRH_EMAIL          "MatthewRobertHermes@gmail.com")
set (MRH_URL            "https://github.com/MatthewRHermes/mrh")
set (MRH_LICENSE        "GNU General Public License, version 2")

cmake_minimum_required (VERSION 2.8.11)
set (CMAKE_VERBOSE_MAKEFILE OFF)

# Whether or not to compile with MKL
option (MKL "Compile using the MKL" OFF)
if (MKL)
    set ( ENV{BLA_VENDOR} "Intel10_64lp" )
endif (MKL)
link_directories ($ENV{LD_LIBRARY_PATH})
find_package (LAPACK REQUIRED)

# OpenMP
find_package (OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

include (CheckCXXCompilerFlag)

# Compile to run of this machine
check_cxx_compiler_flag (-xHost HAS_XHOST)
check_cxx_compiler_flag (-march=native HAS_MARCH_NATIVE)
if (HAS_XHOST)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")
elseif (HAS_MARCH_NATIVE)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

# Do interparts optimization
check_cxx_compiler_flag (-ipo HAS_IPO)
if (HAS_IPO)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
endif (HAS_IPO)

# With debug symbols and highly optimized
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3")

# Find the PySCF "library" directory
set (PYTHON_EXECUTABLE "python")
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import os
    from pyscf import lib
    print (os.path.dirname (lib.__file__))"
  OUTPUT_VARIABLE PYSCFLIB
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Build the QC-DMET shared library
set (BUILD_SHARED_LIBS ON)
set (QCDMET_SOURCE_FILES "rhf_response.cc")
add_library (clib_mrh_qcdmet SHARED ${QCDMET_SOURCE_FILES})
target_link_libraries (clib_mrh_qcdmet ${LAPACK_LIBRARIES})
set_target_properties (clib_mrh_qcdmet PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "qcdmet")

# Build the CSFstring shared library
set (CSF_SOURCE_FILES "csfstring.c")
add_library (clib_mrh_csf SHARED ${CSF_SOURCE_FILES})
target_link_libraries (clib_mrh_csf ${LAPACK_LIBRARIES})
set_target_properties (clib_mrh_csf PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "csf")

# Build the sparse density matrix stuff
set (SINT_SOURCE_FILES "sparsedf.c")
add_library (clib_mrh_sint SHARED ${SINT_SOURCE_FILES})
target_link_libraries (clib_mrh_sint ${LAPACK_LIBRARIES})
set_target_properties (clib_mrh_sint PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "sint")


