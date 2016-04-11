cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
message(STATUS "Configuring PGURE-SVT...")

project (PGURE-SVT CXX)

set(OS unix)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_aux/Modules/")

option(BUILD_STATIC_LIBS "Build the static library" OFF)
option(BUILD_SHARED_LIBS "Build the shared library" ON)

include(CheckIncludeFileCXX)
include(CheckLibraryExists)
enable_language(C)

message(STATUS "CMAKE_SYSTEM_NAME          = ${CMAKE_SYSTEM_NAME}"         )
message(STATUS "CMAKE_CXX_COMPILER_ID      = ${CMAKE_CXX_COMPILER_ID}"     )
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_COMPILER_IS_GNUCXX   = ${CMAKE_COMPILER_IS_GNUCXX}"  )

find_package(OpenMP)
include(FindArmadillo)
include(FindTiff)
include(FindOpenBLAS)
include(FindATLAS)
include(FindBLAS)
include(FindLAPACK)
include(FindNLOPT)

if(NLOPT_FOUND)
	set(SVT_LIBS ${SVT_LIBS} ${NLOPT_LIBRARIES})
	set(SVT_INCL ${SVT_INCL} ${NLOPT_INCLUDE_DIR})
endif()
if(TIFF_FOUND)
	set(SVT_LIBS ${SVT_LIBS} ${TIFF_LIBRARIES})
	set(SVT_INCL ${SVT_INCL} ${TIFF_INCLUDE_DIR})
else()
	message(SEND_ERROR "*** ERROR: libtiff not found; Standalone program will not be compiled")
endif()
if(OpenBLAS_FOUND AND ATLAS_FOUND)	
	message(STATUS "*** WARNING: found both OpenBLAS and ATLAS; ATLAS will not be used")
endif()    
if(OpenBLAS_FOUND AND BLAS_FOUND)
	message(STATUS "*** WARNING: found both OpenBLAS and BLAS; BLAS will not be used")
endif()
if(OpenBLAS_FOUND)
	set(SVT_LIBS ${SVT_LIBS} ${OpenBLAS_LIBRARIES})      
else()      
    if(ATLAS_FOUND)
    	set(SVT_LIBS ${SVT_LIBS} ${ATLAS_LIBRARIES})        
    endif()      
    if(BLAS_FOUND)
    	set(SVT_LIBS ${SVT_LIBS} ${BLAS_LIBRARIES})
    endif()          
endif()
if(LAPACK_FOUND)
	set(SVT_LIBS ${SVT_LIBS} ${LAPACK_LIBRARIES})
endif()
if(ARMADILLO_FOUND)
	set(SVT_LIBS ${SVT_LIBS} ${ARMADILLO_LIBRARIES})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wall -march=native -std=c++11")
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
  
include_directories(${CMAKETEST_BINARY_DIR}/src)
link_directories(${CMAKETEST_BINARY_DIR}/src)

add_executable(PGURE-SVT src/medfilter.c src/PGURE-SVT.cpp)
set_target_properties(PGURE-SVT
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
target_link_libraries(PGURE-SVT ${SVT_LIBS})




