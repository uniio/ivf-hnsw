CC = g++
CFLAGS = -ggdb3 -Wall -std=c++11
SET_OBJ_DIR = -o
LIBS = ../../lib/libivf-hnsw.a ../../lib/libhnswlib.a -lfaiss -fopenmp -lopenblas -lpq

all : load_index

.PHONY : all

load_index : load_index.cpp
	$(CC) load_index.cpp $(CFLAGS) $(LIBS) $(SET_OBJ_DIR) load_index

.PHONY:clean
clean:
	rm -rf *.o  load_index core*
	@echo "Clean done!"

