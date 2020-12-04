cmake_minimum_required (VERSION 2.8)

# ivf-hnsw project
project(ivf-hnsw C CXX)

include_directories("${PROJECT_BINARY_DIR}")

# add faiss header and lib
include_directories("${PROJECT_SOURCE_DIR}/faisslib/")
link_directories("${PROJECT_SOURCE_DIR}/faisslib/lib")

# add ivf-hnsw header
include_directories("${PROJECT_SOURCE_DIR}/../")

add_subdirectory(hnswlib)

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# specify header and cpp files
file(GLOB ivf-hnsw_cpu_headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB ivf-hnsw_cpu_cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(ivf-hnsw STATIC ${ivf-hnsw_cpu_headers} ${ivf-hnsw_cpu_cpp})

# SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
SET( CMAKE_CXX_FLAGS  "-O2 -g -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
target_link_libraries(ivf-hnsw faiss hnswlib pq)

# build tests
# add_subdirectory(tests)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
INSTALL(DIRECTORY lib DESTINATION IVF_Release)
INSTALL(FILES faisslib/lib/libfaiss.a DESTINATION IVF_Release/lib)
INSTALL(DIRECTORY faisslib/faiss_release_o2/header/ DESTINATION IVF_Release/faiss)
INSTALL(DIRECTORY hnswlib DESTINATION IVF_Release FILES_MATCHING PATTERN "*.h" PATTERN "CMakeFiles" EXCLUDE)
INSTALL(FILES ${ivf-hnsw_cpu_headers} DESTINATION IVF_Release/ivf-hnsw)

# add_custom_target (create_tar ALL COMMAND ${CMAKE_COMMAND} -E tar "cvf" "ivf.zip" --format=zip IVF_Release)
