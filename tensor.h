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

    Tensor operator*(const float x) {

      Tensor tensor(this->shape[0], this->shape[1]);    
      for(int i = 0; i < tensor.shape[0]; i++)
        for(int j = 0; j < tensor.shape[1]; j++) 
          tensor.value[i][j] = this->value[i][j]*x;
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

    Tensor() {}

    // (batch, width, height, channel)
    Tensor(int arr[]) {
    }

    Tensor(int rows, int cols, float val) {
        init(rows, cols, val);
    }

    Tensor(int rows, int cols) {
        init(rows, cols);
    }

    void init(int rows, int cols) {
      this->shape[0] = rows;
      this->shape[1] = cols;
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

    void init(int rows, int cols, float val) {
      this->shape[0] = rows;
      this->shape[1] = cols;
      this->value = (float**)malloc(this->shape[0]*sizeof(float*));
      for(int i = 0; i < this->shape[0]; i++) {
        this->value[i] = (float*)malloc(this->shape[1]*sizeof(float));
        for(int j = 0; j < this->shape[1]; j++) {
          this->value[i][j] = val;
        }
      }
    }

    Tensor T() {
    
      Tensor _t(this->shape[1], this->shape[0]);
      for(int i = 0; i < _t.shape[0]; i++)
        for(int j = 0; j < _t.shape[1]; j++)
          _t.value[i][j] = this->value[j][i];
      return _t;
    }

};



