CC = g++
CFLAGS = -ggdb3 -Wall -std=c++11
SET_OBJ_DIR = -o
LIBS = ../../lib/libivf-hnsw.a ../../lib/libfaiss.a ../../lib/libhnswlib.a -fopenmp -lopenblas -lpq -lopenblas

all : build_precomputed_index

.PHONY : all

build_precomputed_index : build_precomputed_index.cpp
	$(CC) build_precomputed_index.cpp $(CFLAGS) $(LIBS) $(SET_OBJ_DIR) build_precomputed_index

.PHONY:clean
clean:
	rm -rf *.o  build_precomputed_index core*
	@echo "Clean done!"

