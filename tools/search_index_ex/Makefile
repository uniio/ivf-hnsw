TARGET = search_index
OBJS = $(TARGET).o
CC = g++
CFLAGS = -ggdb3 -Wall -std=c++11
LFLAGS = -livf-hnsw -lhnswlib -lfaiss -fopenmp -lopenblas -lpq
INCS = -I../../../ -I../../ -I../../faisslib/
LIBS = -L../../lib -L../../faisslib/lib

$(TARGET) : $(OBJS)
	$(CC) -o $@ $^ $(INCS) $(LIBS) $(CFLAGS) $(LFLAGS)
$(OBJS) : %.o : %.cpp
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@

%.d:%.cpp
	@set -e; rm -f $@; $(CC) -MM $< $(INCS) > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

-include $(OBJS:.o=.d)

.PHONY:clean
clean:
	rm -rf *.o $(TARGET) $(TARGET).d* core*
	@echo "Clean done!"

