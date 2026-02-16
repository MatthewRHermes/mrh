# Build the FCI RDM library for PBC CI
find_library(FCI_LIBRARY fci HINTS ${PYSCFLIB})
if(NOT FCI_LIBRARY)
  message(FATAL_ERROR "PySCF fci library not found in ${PYSCFLIB}")
endif()

include_directories(${PYSCFLIB}/mcscf)
set (PBC_FCI_RDMs_SOURCE_FILES "pbc_fci_rdms.c")
add_library (clib_mrh_pbc_fci_rdms SHARED ${PBC_FCI_RDMs_SOURCE_FILES})
target_link_libraries(clib_mrh_pbc_fci_rdms ${FCI_LIBRARY} ${LAPACK_LIBRARIES})
set_target_properties (clib_mrh_pbc_fci_rdms PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "pbc_fci_rdms")
