cmake_minimum_required (VERSION 2.8)

project(test C CXX)

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
include_directories(../../)	# ivf-hnsw root directory
LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/)

SET( CMAKE_CXX_FLAGS  "-O2 -g -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../bin/")

foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)

    # target
    add_executable(${name} ${source})
    #target_link_libraries(${name} libfaiss.a libivf-hnsw.a libhnswlib.a openblas pq)
    target_link_libraries(${name}  libivf-hnsw.a libfaiss.a libhnswlib.a openblas pq)

    # Install
    install(TARGETS ${name} DESTINATION test)
endforeach(source)
