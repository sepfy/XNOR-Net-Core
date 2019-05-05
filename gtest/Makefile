

CFLAG = -std=c++11
INC = -I ../include/

SRC = ../gemm.cc ../blas.cc ../utils.cc  ../layers/pooling.cc


all:
	g++ $(CFLAG) $(INC) nn_gtest.cc $(SRC) -lgmock -lgtest -lpthread -o gtest 
	g++ $(CFLAG) $(INC) layers_gtest.cc $(SRC) -lgmock -lgtest -lpthread -o layers_gtest 

