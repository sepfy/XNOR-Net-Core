INCLUDE = -I ./include/
SRC = $(wildcard src/*.cc)
CXXFLAGS = -O2 -std=c++11 -Wno-unused-result 
LIBS = -lm -fopenmp

all:
	$(CXX) $(CXXFLAGS) $(INCLUDE) mnist.cc $(SRC) $(LIBS) -o mnist

objs: $(SRC) 
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(SRC) -c

clean:
	rm -r mnist
