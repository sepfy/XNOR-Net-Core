#include "mnist.h"
#include "params.h"
int main(void) {

  float *X, *Y;
  X = read_train_data();
  Y = read_train_label();
  int batch = 100;


  Convolution conv1(28, 28, 1, 5, 5, 16, 1, false);
  conv1.xnor = false;
  Relu relu1(24*24*16);
  Dropout conv1_drop(24*24*16, 0.5);
  Pooling pool1(24, 24, 16, 2, 2, 16, 2, false);
  Batchnorm bn1(12*12*16);

  Convolution conv2(12, 12, 16, 5, 5, 32, 1, false);
  Relu relu2(8*8*32);
  Dropout conv2_drop(8*8*32, 0.5);
  Pooling pool2(8, 8, 32, 2, 2, 32, 2, false);
  Batchnorm bn2(4*4*32);

  Convolution conv3(4, 4, 32, 4, 4, 512, 1, false);
  Relu relu3(512);
  Dropout conv3_drop(256, 0.5);

  Connected conn1(512, 10);
  SoftmaxWithCrossEntropy softmax(10);

  Network network;
  network.add(&conv1);
  network.add(&relu1);
  network.add(&conv1_drop);
  network.add(&pool1);

  network.add(&bn1);
  network.add(&conv2);
  network.add(&relu2);
  //network.add(&conv2_drop);
  network.add(&pool2);

  network.add(&bn2);
  network.add(&conv3);
  network.add(&relu3);
  //network.add(&conv3_drop);

  network.add(&conn1);
  network.add(&softmax);
  network.initial(batch, 0.001);
/*
  read_param("data/W1.bin", 750, conv1.weight);
  read_param("data/b1.bin", 30, conv1.bias);
  read_param("data/W2.bin", 432000, conn1.weight);
  read_param("data/b2.bin", 100, conn1.bias);
  read_param("data/W3.bin", 1000, conn2.weight);
  read_param("data/b3.bin", 10, conn2.bias);
*/
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
  network.deploy();
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


