cmake_minimum_required(VERSION 3.18.1)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project (pimdb)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/")

# Search the directory tree for files that match the pattern
file(GLOB_RECURSE PIMDB_SRC src/*.cpp)
file(GLOB_RECURSE PIMDB_HEADERS src/*.h)
file(GLOB_RECURSE PIMDB_LIBS libs/*.cpp libs/*.h libs/*.c)

# Include directories
include_directories(include)
include_directories(libs)

# Create the executable
add_executable(pimdb ${PIMDB_SRC} ${PIMDB_HEADERS} ${PIMDB_LIBS})

# MPI
find_package(MPI REQUIRED)
#include_directories(${FFTW3_INCLUDE_DIR})

include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_FLAGS "${CMAKE_FLAGS} ${MPI_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_EXE_LINKER_FLAGS}")

target_link_libraries(pimdb ${MPI_C_LIBRARIES})
#target_link_libraries(pimdb ${MPI_C_LIBRARIES} ${FFTW3_LIBRARIES})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET pimdb PROPERTY CXX_STANDARD 20)
endif()

# Determine the dimension of space (default = 1)
if (NOT NDIM)
    set(NDIM 3)
endif()
add_definitions(-D NDIM=${NDIM})

# Determine if we want to compile with the naive (factorially-scaling) bosonic algorithm
set(FACTORIAL_BOSONIC_ALGORITHM false CACHE BOOL "Set to true to compile with the naive bosonic algorithm")

if (FACTORIAL_BOSONIC_ALGORITHM)
    add_compile_definitions(FACTORIAL_BOSONIC_ALGORITHM)
endif()

# Unit tests of the ladder (linear-time) bosonic exchange: the recursion against brute force, and
# the forces of the full exchange classes against finite differences (MPI driver).
set(BUILD_LADDER_TESTS false CACHE BOOL "Build the ladder exchange unit tests")
if (BUILD_LADDER_TESTS)
    add_executable(ladder_recursion_test tests/ladder_recursion_test.cpp)
    set_property(TARGET ladder_recursion_test PROPERTY CXX_STANDARD 20)
    set(PIMDB_LIB_SRC ${PIMDB_SRC})
    list(FILTER PIMDB_LIB_SRC EXCLUDE REGEX "src/pimdb.cpp$")
    add_executable(ladder_force_test tests/ladder_force_test.cpp ${PIMDB_LIB_SRC} ${PIMDB_LIBS})
    target_link_libraries(ladder_force_test ${MPI_C_LIBRARIES})
    set_property(TARGET ladder_force_test PROPERTY CXX_STANDARD 20)
endif()

# Register deterministic checks; all MPI fixtures run in separate build directories.
if (BUILD_LADDER_TESTS)
    enable_testing()
    add_test(NAME ladder_recursion COMMAND ladder_recursion_test)
    foreach(wind false true)
        set(case_dir "${CMAKE_CURRENT_BINARY_DIR}/test-force-${wind}")
        file(MAKE_DIRECTORY "${case_dir}")
        add_test(NAME ladder_force_${wind}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ladder_force_test>
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/cases/ladder_force/ladder_force_wind_${wind}.ini"
                ${MPIEXEC_POSTFLAGS})
        set_tests_properties(ladder_force_${wind} PROPERTIES
            WORKING_DIRECTORY "${case_dir}" PROCESSORS 4 TIMEOUT 300)
    endforeach()
endif()

# Optional compute-node microbenchmark; not a scientific acceptance test.
if (BUILD_LADDER_TESTS)
    add_executable(exchange_benchmark tests/exchange_benchmark.cpp ${PIMDB_LIB_SRC} ${PIMDB_LIBS})
    target_link_libraries(exchange_benchmark ${MPI_C_LIBRARIES})
    set_property(TARGET exchange_benchmark PROPERTY CXX_STANDARD 20)
endif()
