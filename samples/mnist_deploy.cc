#include "mnist.h"

int main(void) {

  float *X, *Y;  
  Network network;
  network.load(100);

  X = read_validate_data();
  Y = read_validate_label();

  int batch = 100;
  float total_err = 0.0;
  int batch_num = 100/batch;

  ms_t start = getms();
  for(int iter = 0; iter < batch_num; iter++) {

    int step = (iter*batch);
    float *batch_xs = X + step*784;
    float *batch_ys = Y + step*10;
    float *output = network.inference(batch_xs);
    total_err += accuracy(batch, 10, output, batch_ys);
  }
  cout << "Validate set error = " << (1.0 - total_err/batch_num)*100 
       << ", time = " << getms() -start  << endl;

  return 0;
}


