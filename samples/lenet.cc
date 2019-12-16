#include <iostream>
#include <dirent.h> 
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "lenet.h"

using namespace std;


#define LEARNING_RATE 1.0e-3
#define BATCH 100
#define MAX_ITER 1000

void LeNet(Network *network) {

  Convolution *conv1 = new Convolution(32, 32, 3, 5, 5, 20, 1, false);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(28*28*20);
  Relu *relu1 = new Relu(28*28*20);
  Pooling *pool1 = new Pooling(28, 28, 20, 2, 2, 20, 2, false);

  Convolution *conv2 = new Convolution(14, 14, 20, 5, 5, 50, 1, false);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(10*10*50);
  Relu *relu2 = new Relu(10*10*50);
  Pooling *pool2 = new Pooling(10, 10, 50, 2, 2, 50, 2, false);

  Convolution *conv3 = new Convolution(5, 5, 50, 5, 5, 500, 1, false);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(500);
  Relu *relu3 = new Relu(500);

  Connected *conn4 = new Connected(500, 2);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(2);

  
  network->add(conv1);
  network->add(bn1);
  network->add(relu1);
  network->add(pool1);

  network->add(conv2);
  network->add(bn2);
  network->add(relu2);
  network->add(pool2);

  network->add(conv3);
  network->add(bn3);
  network->add(relu3);

  network->add(conn4);
  network->add(softmax);

}

void help() {
  cout << "Usage: ./lenet <train/deploy> <model name> <dataset>" << endl; 
  exit(1);
}

int main( int argc, char** argv ) {

 float *train_data, *train_label;

  if(argc < 4) {
    help();
  }

  Network network;

  if(strcmp(argv[1], "train") == 0) {

    LeNet(&network);
    network.initial(BATCH, LEARNING_RATE);

    float *batch_xs, *batch_ys;
    batch_xs = new float[BATCH*IM_SIZE];
    batch_ys = new float[BATCH*NUM_OF_CLASS];

    float *train_data, *train_label;
    int num_of_samples = read_data(argv[3], train_data, train_label);

    for(int iter = 0; iter < MAX_ITER; iter++) {

      get_mini_batch(num_of_samples, BATCH, train_data, train_label, batch_xs, batch_ys);

      ms_t start = getms();
      float *output = network.inference(batch_xs);
      network.train(batch_ys);

      float loss = cross_entropy(BATCH, NUM_OF_CLASS, output, batch_ys);
      float acc = accuracy(BATCH, NUM_OF_CLASS, output, batch_ys); 

      if(iter%1 == 0) {
        cout << "iter = " << iter << ", time = " << (getms() - start) << "ms, loss = "
         << loss << " (accuracy = " << acc << ")" << endl;
      }
    }

    network.save(argv[2]);
    delete []train_data;
    delete []train_label;
  }
  else if(strcmp(argv[1], "deploy") == 0) {
    network.load(argv[2], BATCH);
  } 
  else {
    help();
  }

 

/*
  float *test_data, *test_label;

  test_data = new float[10000*IM_SIZE];
  test_label = new float[10000*NUM_OF_CLASS];
  read_test_data(argv[3], test_data, test_label);

  float total = 0.0;
  ms_t start = getms();

  for(int iter = 0; iter < 100; iter++) {
    int step = (iter*BATCH)%10000;
    float *batch_xs = test_data + step*IM_SIZE;
    float *batch_ys = test_label + step*NUM_OF_CLASS;
    float *output = network.inference(batch_xs);

    total += accuracy(BATCH, NUM_OF_CLASS, output, batch_ys); 
  }

  cout << "Validation accuracy = " << (total/100.0) 
       << ", time = " << (getms() - start) << endl;


  delete []test_data;
  delete []test_label;
*/

  return 0;

}
