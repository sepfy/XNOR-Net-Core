
INC = -I ./include/
LAYERS = $(wildcard layers/*.cc)

CXXFLAGS = -O2 -std=c++11 -Wno-unused-result 
all:
	#g++ nn.cc read.cc -lm -o nn
	$(CXX) $(CXXFLAGS) $(INC) mnist.cc blas.cc gemm.cc loss.cc utils.cc $(LAYERS) -lm -fopenmp -o mnist

clean:
	rm -r mnist
