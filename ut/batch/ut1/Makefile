CC = g++
CFLAGS = -ggdb3 -Wall -std=c++11
SET_OBJ_DIR = -o
LIBS = ../../../lib/libivf-hnsw.a ../../../lib/libfaiss.a ../../../lib/libhnswlib.a -fopenmp -lopenblas -lpq -lopenblas

all : ut_main

.PHONY : all

ut_main : ut_main.cpp
	$(CC) ut_main.cpp $(CFLAGS) $(LIBS) $(SET_OBJ_DIR) ut_main

.PHONY:clean
clean:
	rm -rf *.o  ut_main core*
	@echo "Clean done!"

