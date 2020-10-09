CC = g++
CFLAGS = -ggdb3 -Wall -std=c++11
SET_OBJ_DIR = -o
LIBS = -livf-hnsw -lhnswlib -lfaiss -fopenmp -lopenblas -lpq

all : search_index

.PHONY : all

search_index : search_index.cpp
	$(CC) search_index.cpp $(CFLAGS) $(LIBS) $(SET_OBJ_DIR) search_index

.PHONY:clean
clean:
	rm -rf *.o  search_index core*
	@echo "Clean done!"

