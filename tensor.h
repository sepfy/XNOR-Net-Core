#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <string.h>
#include <iostream>
#include <time.h>
using namespace std;

class Tensor {
  public:
    string name;
    int shape[2];
    float **value; 

    Tensor operator*(const Tensor& t) {
      Tensor tensor(this->shape[0], t.shape[1]);    // 10x1
      
      for(int i = 0; i < tensor.shape[0]; i++){
        for(int j = 0; j < tensor.shape[1]; j++) {
          tensor.value[i][j] = 0.0;
          for(int k = 0; k < this->shape[1]; k++) {
            tensor.value[i][j] += this->value[i][k]*t.value[k][j];
          }
        }
      }
      return tensor;
    }

    Tensor operator+(const Tensor& t) {
      Tensor tensor(this->shape[0], this->shape[1]);    // 10x1
      for(int i = 0; i < tensor.shape[0]; i++)
        for(int j = 0; j < tensor.shape[1]; j++) 
            tensor.value[i][j] = this->value[i][j] + t.value[i%t.shape[0]][j];
      return tensor;
    }

    Tensor operator-(const Tensor& t) {
      Tensor tensor(this->shape[0], this->shape[1]);    // 10x1
      for(int i = 0; i < tensor.shape[0]; i++)
        for(int j = 0; j < tensor.shape[1]; j++) 
            tensor.value[i][j] = this->value[i][j] - t.value[i%t.shape[0]][j];
      return tensor;
    }


    Tensor(int rows, int cols) {
        //this->name = name;
        this->shape[0] = rows;
        this->shape[1] = cols;
        init();
    }

    void init() {
      this->value = (float**)malloc(this->shape[0]*sizeof(float*));
      srand(time(NULL));
      for(int i = 0; i < this->shape[0]; i++) {
        this->value[i] = (float*)malloc(this->shape[1]*sizeof(float));
        for(int j = 0; j < this->shape[1]; j++) {
          this->value[i][j] = 0.1*((float) rand()/(RAND_MAX + 1.0) -0.5);
          //cout << this->value[i][j] << ", ";
          //this->value[i][j] = 1.0;
        }
      }
    }

};
