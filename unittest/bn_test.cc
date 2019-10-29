#include "mnist.h"
#include "params.h"
int main(void) {

  float *X, *Y;
  X = read_train_data();
  Y = read_train_label();

  Connected conn1(784, 100);
  Batchnorm bn1(100);
  Relu relu1(100);

  Connected conn2(100, 100);
  Batchnorm bn2(100);
  Relu relu2(100);

  Connected conn3(100, 10);
  SoftmaxWithCrossEntropy softmax(10);

  Network network;
  network.add(&conn1);
  network.add(&bn1);
  network.add(&relu1);

  network.add(&conn2);
  network.add(&bn2);
  network.add(&relu2);

  network.add(&conn3);
  network.add(&softmax);

  int batch = 1000;      
  network.initial(batch, 0.001);
  int max_iter = 10000;

  for(int iter = 0; iter < max_iter; iter++) {

    ms_t start = getms();
    int step = 0;//(iter*batch)%60000;
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
  read_param("bn_data/W1.bin", 78400, conn1.weight);
  read_param("bn_data/b1.bin", 100, conn1.bias);
  read_param("bn_data/W2.bin", 10000, conn2.weight);
  read_param("bn_data/b2.bin", 100, conn2.bias);
  read_param("bn_data/W3.bin", 1000, conn3.weight);
  read_param("bn_data/b3.bin", 10, conn3.bias);

  float loss1;
  read_param("bn_data/loss.bin", 1, &loss1);


  float *output = network.inference(X);

  network.train(Y);

  float sum;

  sum = 0.0;
  for(int i = 0; i < 78400; i++)
    sum += conn1.grad_weight[i];
  cout << sum << endl;

  sum = 0.0;
  for(int i = 0; i < 10000; i++)
    sum += conn2.grad_weight[i];
  cout << sum << endl;


  sum = 0.0;
  for(int i = 0; i < 1000; i++)
    sum += conn3.grad_weight[i];
  cout << sum << endl;

  sum = 0.0;
  for(int i = 0; i < 100; i++)
    sum += bn1.dgamma[i];
  cout << sum << endl;

  sum = 0.0;
  for(int i = 0; i < 100; i++)
    sum += bn2.dgamma[i];
  cout << sum << endl;

  sum = 0.0;
  for(int i = 0; i < 100; i++)
    sum += bn1.dbeta[i];
  cout << sum << endl;

  sum = 0.0;
  for(int i = 0; i < 100; i++)
    sum += bn2.dbeta[i];
  cout << sum << endl;

  float gw1sum, gw2sum, gw3sum;
  read_param("bn_data/gw1sum.bin", 1, &gw1sum);
  read_param("bn_data/gw2sum.bin", 1, &gw2sum);
  read_param("bn_data/gw3sum.bin", 1, &gw3sum);
  cout << "gw1sum: " << gw1sum << endl;
  cout << "gw2sum: " << gw2sum << endl;
  cout << "gw3sum: " << gw3sum << endl;

  float gg1sum, gg2sum, gb1sum, gb2sum;
  read_param("bn_data/gg1sum.bin", 1, &gg1sum);
  read_param("bn_data/gg2sum.bin", 1, &gg2sum);
  read_param("bn_data/gb1sum.bin", 1, &gb1sum);
  read_param("bn_data/gb2sum.bin", 1, &gb2sum);

  cout << "gg1sum: " << gg1sum << endl;
  cout << "gg2sum: " << gg2sum << endl;
  cout << "gb1sum: " << gb1sum << endl;
  cout << "gb2sum: " << gb2sum << endl;

  float loss = cross_entropy(5, 10, output, Y);
  cout << "loss = " << loss << endl;
  cout << "correct loss = " << loss1 << endl;
  */
  return 0;
}


