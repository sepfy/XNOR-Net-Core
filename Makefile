
INC = -I ./include/
LAYERS = $(wildcard layers/*.cc)

all:
	#g++ nn.cc read.cc -lm -o nn
	g++ $(INC) mnist.cc blas.cc gemm.cc loss.cc utils.cc $(LAYERS) -lm -fopenmp -o mnist

clean:
	rm -r mnist
