#include "mnist.h"
#include "params.h"
int main(void) {

  float *X, *Y;
  X = read_train_data();
  Y = read_train_label();
  int batch = 100;


  Convolution conv1(28, 28, 1, 5, 5, 30, 1, false);
  Relu relu1(24*24*30);
  Pooling pool1(24, 24, 30, 2, 2, 30, 2, false);
  Connected conn1(12*12*30, 100);
  Relu relu2(100);
  Connected conn2(100, 10);
  SoftmaxWithCrossEntropy softmax(10);

  Network network;
  network.add(&conv1);
  network.add(&relu1);
  network.add(&pool1);
  network.add(&conn1);
  network.add(&relu2);
  network.add(&conn2);
  network.add(&softmax);
  network.initial(batch, 0.001);

  read_param("data/W1.bin", 750, conv1.weight);
  read_param("data/b1.bin", 30, conv1.bias);
  read_param("data/W2.bin", 432000, conn1.weight);
  read_param("data/b2.bin", 100, conn1.bias);
  read_param("data/W3.bin", 1000, conn2.weight);
  read_param("data/b3.bin", 10, conn2.bias);

  float *output = network.inference(X);
  int max_iter = 20000;
  /*
  float loss = cross_entropy(100, 10, output, Y);
  float loss1;
  read_param("data/loss.bin", 1, &loss1);
  cout << "loss = " << loss << endl;
  cout << "correct loss = " << loss1 << endl;
  */

  for(int iter = 0; iter < max_iter; iter++) {

    ms_t start = getms();
    int step = (iter*batch)%60000;
    float *batch_xs = X + step*784;
    float *batch_ys = Y + step*10;

    float *output = network.inference(batch_xs);
    network.train(batch_ys);

    //total_err = accuracy(batch, 10, output, batch_ys);
    float loss = cross_entropy(batch, 10, output, batch_ys);

    if(iter%1 == 0) {
      cout << "iter = " << iter << ", time = " << (getms() - start) << "ms, error = "
       << loss << endl;
    }
  }


  X = read_validate_data();
  Y = read_validate_label();

  float total_err = 0.0;
  int batch_num = 10000/batch;

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

/*
*/
  return 0;
}


