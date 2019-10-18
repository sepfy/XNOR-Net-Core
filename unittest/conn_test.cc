#include "mnist.h"
#include "params.h"
int main(void) {

  float *X, *Y;
  X = read_train_data();
  Y = read_train_label();

  Connected conn1(784, 100);
  Relu relu1(100);
  Connected conn2(100, 10);
  SoftmaxWithCrossEntropy softmax(10);

  Network network;
  network.add(&conn1);
  network.add(&relu1);
  network.add(&conn2);
  network.add(&softmax);
  network.initial(5, 1.0e-4);

  read_param("data/W1.bin", 78400, conn1.weight);
  read_param("data/b1.bin", 100, conn1.bias);
  read_param("data/W2.bin", 1000, conn2.weight);
  read_param("data/b2.bin", 10, conn2.bias);

  float loss1;
  read_param("data/loss.bin", 1, &loss1);


  float *output = network.inference(X);
  float loss = cross_entropy(5, 10, output, Y);
  cout << "loss = " << loss << endl;
  cout << "correct loss = " << loss1 << endl;

  return 0;
}


