# Author: Tom Furnival
# License: GPLv3

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(OS unix)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake/")

include(CheckIncludeFileCXX)
include(CheckLibraryExists)
enable_language(C)

project (PGURE-SVT)

option(USE_OPENBLAS "Use OpenBLAS instead of BLAS if available" ON)
option(USE_OPENMP "Use OpenMP if available" ON)
option(BUILD_EXECUTABLE "Build a standalone executable" ON)

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
        message(SEND_ERROR "ERROR: libtiff not found; Standalone program cannot be compiled")
    endif()
else()
	set(SVT_LIBS ${SVT_LIBS} ${TIFF_LIBRARIES})
	set(SVT_INCL ${SVT_INCL} ${TIFF_INCLUDE_DIR})
endif()

if(OpenBLAS_FOUND AND USE_OPENBLAS)
	if(OpenBLAS_FOUND AND BLAS_FOUND)
		message(STATUS "WARNING: found both OpenBLAS and BLAS; BLAS will not be used")
	endif()
	if(OpenBLAS_FOUND AND ATLAS_FOUND)
		message(STATUS "WARNING: found both OpenBLAS and ATLAS; ATLAS will not be used")
	endif()
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
    message(STATUS "Armadillo config: ${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp")
endif()

# Enable OpenMP & std::thread
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DARMA_DONT_USE_WRAPPER")

if (OPENMP_FOUND AND USE_OPENMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(BUILD_EXECUTABLE AND TIFF_FOUND)
    set(BINARY_NAME "PGURE-SVT")
    add_executable(${BINARY_NAME} src/${BINARY_NAME}.cpp)
    target_link_libraries(${BINARY_NAME} ${SVT_LIBS})
    install(TARGETS ${BINARY_NAME} DESTINATION bin)
endif()

