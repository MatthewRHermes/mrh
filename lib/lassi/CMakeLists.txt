# Build the LASSI library
add_library (clib_mrh_lassi SHARED
  rdm.c slatercondon.c hsi.c
)
target_link_libraries (clib_mrh_lassi ${LAPACK_LIBRARIES})
set_target_properties (clib_mrh_lassi PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "lassi")

