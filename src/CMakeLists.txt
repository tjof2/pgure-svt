cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project (PGURE-SVT)

include_directories(${CMAKETEST_SOURCE_DIR}/src)
link_directories(${CMAKETEST_BINARY_DIR}/src)

add_executable(PGURE-SVT medfilter.c arps.cpp noise.cpp params.cpp pgure.cpp svt.cpp PGURE-SVT.cpp)

set_target_properties( PGURE-SVT
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
target_link_libraries( PGURE-SVT ${SVT_LIBS} )
