

CFLAGS = -std=c++11 -O2
INC = -I ../include/

LIB = -fopenmp
SRC = ../gemm.cc ../blas.cc ../utils.cc  ../layers/pooling.cc


all:
	g++ $(CFLAGS) $(INC) nn_gtest.cc $(SRC) -lgmock -lgtest -lpthread -o gtest 
	g++ $(CFLAGS) $(INC) pooling_gtest.cc $(SRC) -lgmock -lgtest -lpthread -o pooling_gtest 

conv:
	g++ $(CFLAGS) $(INC) convolution_gtest.cc ../layers/convolution.cc $(SRC) -lgmock -lgtest -lpthread -o conv_gtest

gemm:
	$(CXX) $(CFLAGS) $(INC) $(LIB) gemm_gtest.cc ../gemm.cc ../utils.cc -lgmock -lgtest -lpthread -o gemm_test
