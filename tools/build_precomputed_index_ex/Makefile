CC = g++
CFLAGS = -ggdb3 -Wall -std=c++11
SET_OBJ_DIR = -o
LIBS = -livf-hnsw -lhnswlib -lfaiss -fopenmp -lopenblas -lpq

all : build_precomputed_index_ex

.PHONY : all

build_precomputed_index_ex : build_precomputed_index_ex.cpp
	$(CC) build_precomputed_index_ex.cpp $(CFLAGS) $(LIBS) $(SET_OBJ_DIR) build_precomputed_index_ex

.PHONY:clean
clean:
	rm -rf *.o  build_precomputed_index_ex core*
	@echo "Clean done!"

