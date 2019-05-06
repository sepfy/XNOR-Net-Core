

CFLAG = -std=c++11
INC = -I ../include/

SRC = ../gemm.cc ../blas.cc ../utils.cc  ../layers/pooling.cc


all:
	g++ $(CFLAG) $(INC) nn_gtest.cc $(SRC) -lgmock -lgtest -lpthread -o gtest 
	g++ $(CFLAG) $(INC) pooling_gtest.cc $(SRC) -lgmock -lgtest -lpthread -o pooling_gtest 

conv:
	g++ $(CFLAG) $(INC) convolution_gtest.cc ../layers/convolution.cc $(SRC) -lgmock -lgtest -lpthread -o conv_gtest

