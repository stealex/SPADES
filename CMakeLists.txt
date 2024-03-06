cmake_minimum_required(VERSION 3.1)

project(RADIAL VERSION 0.1
        DESCRIPTION "Radial project"
        LANGUAGES Fortran)
enable_language(Fortran)

if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -Wall -shared -fpic -finteger-4-integer-8")
    set(CMAKE_Fortran_FLAGS_DEBUG   "-g -traceback -shared -fpic -finteger-4-integer-8")
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -ip -xHOST -shared -fpic -finteger-4-integer-8")
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
    set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -Wall -shared -fpic -finteger-4-integer-8")
    set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g3 -shared -fpic -finteger-4-integer-8")
    set(CMAKE_Fortran_FLAGS_RELEASE "-Ofast -march=native -shared -fpic -finteger-4-integer-8")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_subdirectory(src)

