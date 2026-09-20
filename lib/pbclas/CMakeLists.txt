# Build complex one- and two-particle RDM kernels for periodic FCI.
find_library(FCI_LIBRARY fci HINTS ${PYSCFLIB})
if(NOT FCI_LIBRARY)
  message(FATAL_ERROR "PySCF fci library not found in ${PYSCFLIB}")
endif()

find_library(NP_HELPER_LIBRARY np_helper HINTS ${PYSCFLIB})
if(NOT NP_HELPER_LIBRARY)
  message(FATAL_ERROR "PySCF np_helper library not found in ${PYSCFLIB}")
endif()

include_directories(${PYSCFLIB}/mcscf)
set (PBC_FCI_RDMs_SOURCE_FILES "pbc_fci_rdms.c")
add_library (clib_mrh_pbc_fci_rdms SHARED ${PBC_FCI_RDMs_SOURCE_FILES})
#target_link_libraries(clib_mrh_pbc_fci_rdms ${FCI_LIBRARY} ${LAPACK_LIBRARIES})
target_link_libraries(
    clib_mrh_pbc_fci_rdms
    ${FCI_LIBRARY}
    ${NP_HELPER_LIBRARY}
    ${LAPACK_LIBRARIES}
)

set_target_properties (clib_mrh_pbc_fci_rdms PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "pbc_fci_rdms")


# Build optimized complex two-electron contraction kernels for periodic FCI.
include_directories(${PYSCFLIB}/mcscf)
set (PBC_FCI_CONTRACT_OPT_SOURCE_FILES "pbc_fci_contract_opt.c")
add_library (clib_mrh_pbc_fci_contract_opt SHARED ${PBC_FCI_CONTRACT_OPT_SOURCE_FILES})
#target_link_libraries(clib_mrh_pbc_fci_contract_opt ${FCI_LIBRARY} ${LAPACK_LIBRARIES})
target_link_libraries(
    clib_mrh_pbc_fci_contract_opt
    ${FCI_LIBRARY}
    ${NP_HELPER_LIBRARY}
    ${LAPACK_LIBRARIES}
)

set_target_properties (clib_mrh_pbc_fci_contract_opt PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "pbc_fci_contract_opt")


# Build the complex spin-adapted CSF one-electron contraction kernel.
set (CPLX_CSF_SOURCE_FILES "cplx_csf.c")
add_library (clib_mrh_cplx_csf SHARED ${CPLX_CSF_SOURCE_FILES})
target_link_libraries(clib_mrh_cplx_csf ${FCI_LIBRARY} ${LAPACK_LIBRARIES})

set_target_properties (clib_mrh_cplx_csf PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "cplx_csf")

# Build k-aware CI string, link-index, and contraction-structure helpers.
set (PBC_FCI_KCISTRING_SOURCE_FILES "pbc_kcistring.c")
add_library (clib_mrh_pbc_kcistring SHARED ${PBC_FCI_KCISTRING_SOURCE_FILES})
target_link_libraries(clib_mrh_pbc_kcistring ${FCI_LIBRARY} ${LAPACK_LIBRARIES})
set_target_properties (clib_mrh_pbc_kcistring PROPERTIES
      LINKER_LANGUAGE C
      CLEAN_DIRECT_OUTPUT 1
      LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_NAME "pbc_kcistring")

# Define utilities shared by the momentum-sector kFCI libraries.
set (PBC_KFCI_COMMON_SOURCE_FILES "pbc_kfci_common.c")

# Build spin-resolved RDM kernels for momentum-sector kFCI.
set (PBC_KFCI_RDM_SOURCE_FILES
     "pbc_kfci_rdm.c"
     ${PBC_KFCI_COMMON_SOURCE_FILES})
add_library (clib_mrh_pbc_kfci_rdm SHARED ${PBC_KFCI_RDM_SOURCE_FILES})
target_link_libraries(clib_mrh_pbc_kfci_rdm ${FCI_LIBRARY} ${LAPACK_LIBRARIES})
set_target_properties (clib_mrh_pbc_kfci_rdm PROPERTIES
      LINKER_LANGUAGE C
      CLEAN_DIRECT_OUTPUT 1
      LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_NAME "pbc_kfci_rdm")

# Build 1e, 2e, and specificspin contractions for kFCI sectors.
set (PBC_FCI_CONTRACT_K_SOURCE_FILES
     "pbc_fci_contract_k.c"
     ${PBC_KFCI_COMMON_SOURCE_FILES})
add_library (clib_mrh_pbc_fci_contract_k SHARED ${PBC_FCI_CONTRACT_K_SOURCE_FILES})
target_link_libraries(clib_mrh_pbc_fci_contract_k ${FCI_LIBRARY} ${LAPACK_LIBRARIES})
set_target_properties (clib_mrh_pbc_fci_contract_k PROPERTIES
      LINKER_LANGUAGE C
      CLEAN_DIRECT_OUTPUT 1
      LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_NAME "pbc_fci_contract_k")

# HDIAG kernels for kFCI.
set (PBC_KFCI_HDIAG_SOURCE_FILES
     "pbc_kfci_hdiag.c"
     ${PBC_KFCI_COMMON_SOURCE_FILES})
add_library (clib_mrh_pbc_kfci_hdiag SHARED ${PBC_KFCI_HDIAG_SOURCE_FILES})
target_link_libraries(clib_mrh_pbc_kfci_hdiag ${FCI_LIBRARY} ${LAPACK_LIBRARIES})
set_target_properties (clib_mrh_pbc_kfci_hdiag PROPERTIES
      LINKER_LANGUAGE C
      CLEAN_DIRECT_OUTPUT 1
      LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_NAME "pbc_kfci_hdiag")
