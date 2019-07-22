#include "mnist.h"

int main(void) {
  float *X, *Y;
  X = read_train_data();
  Y = read_train_label();

  int max_iter = 200;
  float total_err = 0;
  int batch = 100;

  Convolution conv1(28, 28, 1, 5, 5, 32, 1, true);
  Relu relu1(28*28*32);
  Pooling pool1(28, 28, 32, 2, 2, 32, 2, false); 
  Convolution conv2(14, 14, 32, 5, 5, 64, 1, true);
  Relu relu2(14*14*64);
  Pooling pool2(14, 14, 64, 2, 2, 64, 2, false); 
  Convolution conv3(7, 7, 64, 7, 7, 1024, 1, false);
  Relu relu3(1024);
  Connected conn1(1024, 10);
  SoftmaxWithCrossEntropy softmax(10);

  Network network;
  network.add(&conv1);
  network.add(&relu1);
  network.add(&pool1);
  network.add(&conv2);
  network.add(&relu2);
  network.add(&pool2);
  network.add(&conv3);
  network.add(&relu3);
  network.add(&conn1);
  network.add(&softmax);
  network.initial(batch, 1.0e-4);
 
  for(int iter = 0; iter < max_iter; iter++) {

    ms_t start = getms();
    int step = (iter*batch)%60000;
    float *batch_xs = X + step*784;
    float *batch_ys = Y + step*10;

    float *output = network.inference(batch_xs);
    network.train(batch_ys);// + step);
 
    total_err = accuracy(batch, 10, output, batch_ys);// + step);

    if(iter%1 == 0) {
      cout << "iter = " << iter << ", time = " << (getms() - start) << "ms, error = "
       << total_err << endl;
    }
  }
  

  X = read_validate_data();
  Y = read_validate_label();

  total_err = 0.0;
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

  network.save();

  return 0;
}


