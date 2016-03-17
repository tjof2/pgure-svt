cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

###########################

message(STATUS "Configuring PGURE-SVT ${ARMADILLO_VERSION}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_aux/Modules/")

include(CheckIncludeFileCXX)
include(CheckLibraryExists)
project (PGURE-SVT CXX)
enable_language(C)

message(STATUS "CMAKE_SYSTEM_NAME          = ${CMAKE_SYSTEM_NAME}"         )
message(STATUS "CMAKE_CXX_COMPILER_ID      = ${CMAKE_CXX_COMPILER_ID}"     )
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_COMPILER_IS_GNUCXX   = ${CMAKE_COMPILER_IS_GNUCXX}"  )

###########################

set(OS unix)

find_package(OpenMP)
include(FindArmadillo)
include(FindTiff)
include(FindOpenBLAS)
include(FindATLAS)
include(FindBLAS)
include(FindLAPACK)
include(FindNLOPT)

message(STATUS "")
message(STATUS "   OPENMP_FOUND = ${OPENMP_FOUND}   ")
message(STATUS "    NLopt_FOUND = ${NLOPT_FOUND}    ")
message(STATUS "  LibTIFF_FOUND = ${TIFF_FOUND}     ")
message(STATUS " OpenBLAS_FOUND = ${OpenBLAS_FOUND} ")
message(STATUS "    ATLAS_FOUND = ${ATLAS_FOUND}    ")
message(STATUS "     BLAS_FOUND = ${BLAS_FOUND}     ")
message(STATUS "   LAPACK_FOUND = ${LAPACK_FOUND}   ")
message(STATUS "Armadillo_FOUND = ${ARMADILLO_FOUND}")
message(STATUS "")

if(NLOPT_FOUND)
	set(SVT_LIBS ${SVT_LIBS} ${NLOPT_LIBRARIES})
	set(SVT_INCL ${SVT_INCL} ${NLOPT_INCLUDE_DIR})
endif()

if(TIFF_FOUND)
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

###########################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -march=native -std=c++11")
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

message(STATUS "")
message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "")
message(STATUS "LIBS = ${SVT_LIBS}")
message(STATUS "")

###########################
  
add_subdirectory(src)




