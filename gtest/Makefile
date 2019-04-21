

CFLAG = -std=c++11
INC = -I ../include/

SRC = ../gemm.cc ../blas.cc ../utils.cc


all:
	g++ $(CFLAG) $(INC) nn_gtest.cc $(SRC) -lgmock -lgtest -lpthread -o gtest 

