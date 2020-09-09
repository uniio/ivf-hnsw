CC = g++
CFLAGS = -ggdb3 -Wall -std=c++11
SET_OBJ_DIR = -o
LIBS = ../../lib/libivf-hnsw.a ../../lib/libhnswlib.a -lfaiss -fopenmp -lopenblas -lpq -lopenblas

all : build_pq

.PHONY : all

build_pq : build_pq.cpp
	$(CC) build_pq.cpp $(CFLAGS) $(LIBS) $(SET_OBJ_DIR) build_pq

.PHONY:clean
clean:
	rm -rf *.o  build_pq core*
	@echo "Clean done!"

