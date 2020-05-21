find_path(ATLAS_INCLUDE_DIR cblas.h
/usr/include/atlas/
/usr/include/
/usr/local/include/atlas/
/usr/local/include/
)

set(ATLAS_NAMES)
set(ATLAS_NAMES ${ATLAS_NAMES} tatlas)
set(ATLAS_NAMES ${ATLAS_NAMES} satlas)
set(ATLAS_NAMES ${ATLAS_NAMES} atlas )

set(ATLAS_TMP_LIBRARY)
set(ATLAS_TMP_LIBRARIES)


foreach (ATLAS_NAME ${ATLAS_NAMES})
  find_library(${ATLAS_NAME}_LIBRARY
    NAMES ${ATLAS_NAME}
    PATHS ${CMAKE_SYSTEM_LIBRARY_PATH} /usr/lib64/atlas /usr/lib64/ /usr/local/lib64/atlas /usr/local/lib64 /usr/lib/atlas /usr/lib /usr/local/lib/atlas /usr/local/lib
    )
  
  set(ATLAS_TMP_LIBRARY ${${ATLAS_NAME}_LIBRARY})
  
  if(ATLAS_TMP_LIBRARY)
    set(ATLAS_TMP_LIBRARIES ${ATLAS_TMP_LIBRARIES} ${ATLAS_TMP_LIBRARY})
  endif()
endforeach()


# use only one library

if(ATLAS_TMP_LIBRARIES)
  list(GET ATLAS_TMP_LIBRARIES 0 ATLAS_LIBRARY)
endif()


if(ATLAS_LIBRARY AND ATLAS_INCLUDE_DIR)
  set(ATLAS_LIBRARIES ${ATLAS_LIBRARY})
  set(ATLAS_FOUND "TRUE")
else()
  set(ATLAS_FOUND "FALSE")
endif()

set(ATLAS_FIND_QUIETLY "TRUE")
if(ATLAS_FOUND)
  if(NOT ATLAS_FIND_QUIETLY)
    message(STATUS "Found ATLAS: ${ATLAS_LIBRARIES}")
  endif()
else()
  if(ATLAS_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find ATLAS")
  endif()
endif()


# mark_as_advanced(ATLAS_LIBRARY ATLAS_INCLUDE_DIR)
