CC = g++
CFLAGS = -ggdb3 -Wall -std=c++11
SET_OBJ_DIR = -o

all : vector_split

.PHONY : all

vector_split : vector_split.cpp
	$(CC) vector_split.cpp $(CFLAGS) $(SET_OBJ_DIR) vector_split

.PHONY:clean
clean:
	rm -rf *.o  vector_split core*
	@echo "Clean done!"

