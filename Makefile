
all:
	#g++ nn.cc read.cc -lm -o nn
	g++ unit_test.cc gemm.cc blas.cc utils.cc -lm -o test
	g++ mnist.cc blas.cc gemm.cc loss.cc utils.cc -lm -o mnist
