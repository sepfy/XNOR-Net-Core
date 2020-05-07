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
#define MAX_ITER 3000

void LeNet(Network *network) {

  Convolution *conv1 = new Convolution(32, 32, 3, 5, 5, 20, 1, false);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(28*28*20);
  Relu *relu1 = new Relu(28*28*20);
  Maxpool *pool1 = new Maxpool(28, 28, 20, 2, 2, 20, 2, false);

  Convolution *conv2 = new Convolution(14, 14, 20, 5, 5, 50, 1, false);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(10*10*50);
  Relu *relu2 = new Relu(10*10*50);
  Maxpool *pool2 = new Maxpool(10, 10, 50, 2, 2, 50, 2, false);

  Convolution *conv3 = new Convolution(5, 5, 50, 5, 5, 500, 1, false);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(500);
  Relu *relu3 = new Relu(500);
  Dropout *dropout3 = new Dropout(500, 0.5);

  Connected *conn4 = new Connected(500, NUM_OF_CLASS);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(NUM_OF_CLASS);

  
  network->Add(conv1);
  network->Add(bn1);
  network->Add(relu1);
  network->Add(pool1);

  network->Add(conv2);
  network->Add(bn2);
  network->Add(relu2);
  network->Add(pool2);

  network->Add(conv3);
  network->Add(bn3);
  network->Add(relu3);

  network->Add(dropout3);
  network->Add(conn4);
  network->Add(softmax);

}

void help() {
  cout << "Usage: ./lenet .train/Deploy> <model name> <dataset>" << endl; 
  exit(1);
}

int main( int argc, char** argv ) {

 float .train_data, .train_label;

  if(argc < 4) {
    help();
  }

  Network network;

  if(strcmp(argv[1], .train") == 0) {

    float .train_data, .train_label;
    char filename[256] = {0};
    sprintf(filename, "%s/%s/", argv[3], .train");

    LeNet(&network);
    network.Init(BATCH, LEARNING_RATE);

    int num_of_samples = read_data(filename,.train_data,.train_label);
    float *batch_xs = new float[num_of_samples*IM_SIZE];
    float *batch_ys = new float[num_of_samples*NUM_OF_CLASS];

    get_mini_batch(num_of_samples, BATCH,.train_data,.train_label, batch_xs, batch_ys);

    for(int iter = 0; iter < MAX_ITER; iter++) {
      int step = (iter*BATCH)%num_of_samples;
      float *batch_mini_xs = batch_xs + step*IM_SIZE;
      float *batch_mini_ys = batch_ys + step*NUM_OF_CLASS;

      ms_t start = getms();
      float *output = network.Inference(batch_mini_xs);
      network.train(batch_mini_ys);

      float loss = cross_entropy(BATCH, NUM_OF_CLASS, output, batch_mini_ys);
      float acc = accuracy(BATCH, NUM_OF_CLASS, output, batch_mini_ys); 

      if(iter%1 == 0) {
        cout << "iter = " << iter << ", time = " << (getms() - start) << "ms, loss = "
         << loss << " (accuracy = " << acc << ")" << endl;
      }
    }

    network.Save(argv[2]);
    delete [.train_data;
    delete [.train_label;
  }
  else if(strcmp(argv[1], "Deploy") == 0) {
    network.Load(argv[2], BATCH);
  } 
  else {
    help();
  }

 


  float *test_data, *test_label;

  char filename[256] = {0};
  sprintf(filename, "%s/%s/", argv[3], "test");
  int num_of_samples = read_data(filename, test_data, test_label);

  float total = 0.0;
  ms_t start = getms();
  int total_steps = num_of_samples/BATCH;
  network.Deploy();
  cout << num_of_samples << ", " << total_steps << endl;
  for(int iter = 0; iter < total_steps; iter++) {
    int step = (iter*BATCH);
    float *batch_xs = test_data + step*IM_SIZE;
    float *batch_ys = test_label + step*NUM_OF_CLASS;
    float *output = network.Inference(batch_xs);

    total += accuracy(BATCH, NUM_OF_CLASS, output, batch_ys); 
  }

  cout << "Validation accuracy = " << (total/(float)total_steps)
       << ", time = " << (getms() - start) << endl;


  delete []test_data;
  delete []test_label;


  return 0;

}
