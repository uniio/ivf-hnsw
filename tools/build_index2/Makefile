CC = g++
CFLAGS = -ggdb3 -Wall -std=c++11
SET_OBJ_DIR = -o
LIBS = ../../lib/libivf-hnsw.a ../../lib/libhnswlib.a -lfaiss -fopenmp -lopenblas -lpq -lopenblas

all : build_index

.PHONY : all

build_index : build_index.cpp
	$(CC) build_index.cpp $(CFLAGS) $(LIBS) $(SET_OBJ_DIR) build_index

.PHONY:clean
clean:
	rm -rf *.o  build_index core*
	@echo "Clean done!"

