cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
message(STATUS "Configuring PGURE-SVT...")

project (PGURE-SVT CXX)

set(OS unix)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_aux/Modules/")

option(BUILD_LIBRARY "Build the shared library" ON)
option(BUILD_EXECUTABLE "Build a standalone executable" OFF)

include(CheckIncludeFileCXX)
include(CheckLibraryExists)
enable_language(C)

message(STATUS "CMAKE_SYSTEM_NAME          = ${CMAKE_SYSTEM_NAME}"         )
message(STATUS "CMAKE_CXX_COMPILER_ID      = ${CMAKE_CXX_COMPILER_ID}"     )
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_COMPILER_IS_GNUCXX   = ${CMAKE_COMPILER_IS_GNUCXX}"  )

########################################

# Look for packages
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

if(!TIFF_FOUND)
    if(BUILD_EXECUTABLE)
        message(SEND_ERROR "*** ERROR: libtiff not found; Standalone program will not be compiled")
    endif()
else()
	set(SVT_LIBS ${SVT_LIBS} ${TIFF_LIBRARIES})
	set(SVT_INCL ${SVT_INCL} ${TIFF_INCLUDE_DIR})
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

########################################

# Enable OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wall -march=native -std=c++11")
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

########################################

# Build libraries and executable
if(BUILD_LIBRARY)
    add_library(pguresvt SHARED src/medfilter.c src/lib-PGURE-SVT.cpp)
    target_link_libraries(pguresvt ${SVT_LIBS})
endif()

if(BUILD_EXECUTABLE AND TIFF_FOUND)
    add_executable(PGURE-SVT src/medfilter.c src/PGURE-SVT.cpp)
    target_link_libraries(PGURE-SVT ${SVT_LIBS})
endif()




