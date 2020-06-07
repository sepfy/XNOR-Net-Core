#include <iostream>
#include <dirent.h> 
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "cifar.h"
#include "network.h"

using namespace std;

#define LEARNING_RATE 1.0e-3
#define BATCH 100
#define MAX_ITER 5000

void CifarXnorNet(Network *network) {


  Convolution *conv1 = new Convolution(32, 32, 3, 3, 3, 128, 1, 1);
  Batchnorm *bn1 = new Batchnorm(32*32, 128);
  Activation *actv1 = new Activation(32*32*128, LEAKY);

  Batchnorm *bn2 = new Batchnorm(32*32, 128);
  BinaryConv *bin_conv2 = new BinaryConv(32, 32, 128, 3, 3, 128, 1, 1);
  Activation *actv2 = new Activation(32*32*128, LEAKY);

  Batchnorm *bn3 = new Batchnorm(32*32, 128);
  BinaryConv *bin_conv3 = new BinaryConv(32, 32, 128, 3, 3, 128, 1, 1);
  Activation *actv3 = new Activation(32*32*128, LEAKY);

  Maxpool *pool1 = new Maxpool(32, 32, 128, 2, 2, 128, 2, false);
  Dropout *dropout1 = new Dropout(16*16*128, 0.5);

  Batchnorm *bn4 = new Batchnorm(16*16, 128);
  BinaryConv *bin_conv4 = new BinaryConv(16, 16, 128, 3, 3, 256, 1, 1);
  Activation *actv4 = new Activation(16*16*256, LEAKY);


  Batchnorm *bn5 = new Batchnorm(16*16, 256);
  BinaryConv *bin_conv5 = new BinaryConv(16, 16, 256, 3, 3, 256, 1, 1);
  Activation *actv5 = new Activation(16*16*256, LEAKY);


  Batchnorm *bn6 = new Batchnorm(16*16, 256);
  BinaryConv *bin_conv6 = new BinaryConv(16, 16, 256, 3, 3, 256, 1, 1);
  Activation *actv6 = new Activation(16*16*256, LEAKY);

  Maxpool *pool2 = new Maxpool(16, 16, 256, 2, 2, 256, 2, false);

  Dropout *dropout2 = new Dropout(8*8*256, 0.5);


  Batchnorm *bn7 = new Batchnorm(8*8, 256);
  BinaryConv *bin_conv7 = new BinaryConv(8, 8, 256, 3, 3, 512, 1, 1);
  Activation *actv7 = new Activation(8*8*512, LEAKY);

  Batchnorm *bn8 = new Batchnorm(8*8, 512);
  BinaryConv *bin_conv8 = new BinaryConv(8, 8, 512, 3, 3, 512, 1, 1);
  Activation *actv8 = new Activation(8*8*512, LEAKY);

  Batchnorm *bn9 = new Batchnorm(8*8, 512);
  BinaryConv *bin_conv9 = new BinaryConv(8, 8, 512, 3, 3, 512, 1, 1);
  Activation *actv9 = new Activation(8*8*512, LEAKY);

  Dropout *dropout3 = new Dropout(8*8*512, 0.5);


  Convolution *conv10 = new Convolution(8, 8, 512, 3, 3, 10, 1, 1);
  Batchnorm *bn10 = new Batchnorm(8*8, 10);
  Activation *actv10 = new Activation(8*8*10, LEAKY);

  Avgpool *avgpool = new Avgpool(8, 8, 10, 8, 8, 10, 1, false);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);

  network->Add(conv1);
  network->Add(bn1);
  network->Add(actv1);

  network->Add(bn2);
  network->Add(bin_conv2);
  network->Add(actv2);

  network->Add(bn3);
  network->Add(bin_conv3);
  network->Add(actv3);

  network->Add(pool1);

  network->Add(bn4);
  network->Add(bin_conv4);
  network->Add(actv4);

  network->Add(bn5);
  network->Add(bin_conv5);
  network->Add(actv5);

  network->Add(bn6);
  network->Add(bin_conv6);
  network->Add(actv6);

  network->Add(pool2);

  network->Add(bn7);
  network->Add(bin_conv7);
  network->Add(actv7);

  network->Add(bn8);
  network->Add(bin_conv8);
  network->Add(actv8);

  network->Add(bn9);
  network->Add(bin_conv9);
  network->Add(actv9);

  network->Add(conv10);
  network->Add(bn10);
  network->Add(actv10);
  
  network->Add(avgpool);
  network->Add(softmax);

}


void CifarNet(Network *network) {

  Convolution *conv1 = new Convolution(32, 32, 3, 3, 3, 128, 1, 1);
  Batchnorm *bn1 = new Batchnorm(32*32, 128);
  Activation *actv1 = new Activation(32*32*128, LEAKY);

  Convolution *conv2 = new Convolution(32, 32, 128, 3, 3, 128, 1, 1);
  Batchnorm *bn2 = new Batchnorm(32*32, 128);
  Activation *actv2 = new Activation(32*32*128, LEAKY);

  Convolution *conv3 = new Convolution(32, 32, 128, 3, 3, 128, 1, 1);
  Batchnorm *bn3 = new Batchnorm(32*32, 128);
  Activation *actv3 = new Activation(32*32*128, LEAKY);

  Maxpool *pool1 = new Maxpool(32, 32, 128, 2, 2, 128, 2, false);
  Dropout *dropout1 = new Dropout(16*16*128, 0.5);

  Convolution *conv4 = new Convolution(16, 16, 128, 3, 3, 256, 1, 1);
  Batchnorm *bn4 = new Batchnorm(16*16, 256);
  Activation *actv4 = new Activation(16*16*256, LEAKY);

  Convolution *conv5 = new Convolution(16, 16, 256, 3, 3, 256, 1, 1);
  Batchnorm *bn5 = new Batchnorm(16*16, 256);
  Activation *actv5 = new Activation(16*16*256, LEAKY);

  Convolution *conv6 = new Convolution(16, 16, 256, 3, 3, 256, 1, 1);
  Batchnorm *bn6 = new Batchnorm(16*16, 256);
  Activation *actv6 = new Activation(16*16*256, LEAKY);

  Maxpool *pool2 = new Maxpool(16, 16, 256, 2, 2, 256, 2, false);

  Dropout *dropout2 = new Dropout(8*8*256, 0.5);

  Convolution *conv7 = new Convolution(8, 8, 256, 3, 3, 512, 1, 1);
  Batchnorm *bn7 = new Batchnorm(8*8, 512);
  Activation *actv7 = new Activation(8*8*512, LEAKY);

  Convolution *conv8 = new Convolution(8, 8, 512, 3, 3, 512, 1, 1);
  Batchnorm *bn8 = new Batchnorm(8*8, 512);
  Activation *actv8 = new Activation(8*8*512, LEAKY);

  Convolution *conv9 = new Convolution(8, 8, 512, 3, 3, 512, 1, 1);
  Batchnorm *bn9 = new Batchnorm(8*8, 512);
  Activation *actv9 = new Activation(8*8*512, LEAKY);

  Dropout *dropout3 = new Dropout(8*8*512, 0.5);

  Convolution *conv10 = new Convolution(8, 8, 512, 3, 3, 10, 1, 1);
  Batchnorm *bn10 = new Batchnorm(8*8, 10);
  Activation *actv10 = new Activation(8*8*10, LEAKY);

  Avgpool *avgpool = new Avgpool(8, 8, 10, 8, 8, 10, 1, false);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);

  network->Add(conv1);
  network->Add(bn1);
  network->Add(actv1);

  network->Add(conv2);
  network->Add(bn2);
  network->Add(actv2);

  network->Add(conv3);
  network->Add(bn3);
  network->Add(actv3);

  network->Add(pool1);
  network->Add(dropout1);

  network->Add(conv4);
  network->Add(bn4);
  network->Add(actv4);

  network->Add(conv5);
  network->Add(bn5);
  network->Add(actv5);

  network->Add(conv6);
  network->Add(bn6);
  network->Add(actv6);

  network->Add(pool2);
  network->Add(dropout2);

  network->Add(conv7);
  network->Add(bn7);
  network->Add(actv7);

  network->Add(conv8);
  network->Add(bn8);
  network->Add(actv8);

  network->Add(conv9);
  network->Add(bn9);
  network->Add(actv9);

  network->Add(dropout3);

  network->Add(conv10);
  network->Add(bn10);
  network->Add(actv10);
  
  network->Add(avgpool);
  network->Add(softmax);
}



void help() {
  cout << "Usage: ./cifar <train/deploy> <model name> <cifar dataset>" << endl; 
  exit(1);
}

int main( int argc, char** argv ) {


  if(argc < 4) {
    help();
  }

  Network network;

  if(strcmp(argv[1], "train") == 0) {

    CifarXnorNet(&network);
    //CifarNet(&network);

    network.Init(BATCH, LEARNING_RATE, true);
    float *train_data, *train_label;

    train_data = new float[50000*IM_SIZE];
    train_label = new float[50000*NUM_OF_CLASS];

    memset(train_data, 0, 50000*IM_SIZE*sizeof(float));
    memset(train_label, 0, 50000*NUM_OF_CLASS*sizeof(float));

    read_train_data(argv[3], train_data, train_label);

#ifdef GPU    
    float *batch_xs = malloc_gpu(BATCH*IM_SIZE);
    float *batch_ys = malloc_gpu(BATCH*NUM_OF_CLASS);
#else
    float *batch_xs;
    float *batch_ys;
#endif

    for(int iter = 0; iter < MAX_ITER; iter++) {

      if(iter > 2000) network.update_args_.lr = 1.0e-4;
      if(iter > 3500) network.update_args_.lr = 1.0e-5;

      ms_t start = getms();
      int step = (iter*BATCH)%50000;
#ifdef GPU
      gpu_push_array(batch_xs, train_data + step*IM_SIZE, BATCH*IM_SIZE);
      gpu_push_array(batch_ys, train_label + step*NUM_OF_CLASS, BATCH*NUM_OF_CLASS);
#else
      batch_xs = train_data + step*IM_SIZE;
      batch_ys = train_label + step*NUM_OF_CLASS;
#endif

      network.Inference(batch_xs);
      float *output = network.output(); 
      network.Train(batch_ys);


      float acc = accuracy(BATCH, NUM_OF_CLASS, output, batch_ys); 

      if(iter%1 == 0) {
        float loss = cross_entropy(BATCH, NUM_OF_CLASS, output, batch_ys);
        cout << "iter = " << iter << ", time = " << (getms() - start) << "ms, loss = "
         << loss << " (acc = " << acc << ")" << endl;
      }

    }
    network.Save(argv[2]);
#ifdef GPU
    cudaFree(batch_xs);
    cudaFree(batch_ys);
#endif
    delete []train_data;
    delete []train_label;
  }
  else if(strcmp(argv[1], "deploy") == 0) {
    network.Load(argv[2], BATCH);
  } 
  else {
    help();
  }

  network.Deploy();


  float *test_data, *test_label;

  test_data = new float[10000*IM_SIZE];
  memset(test_data, 0, 10000*IM_SIZE*sizeof(float));
  test_label = new float[10000*NUM_OF_CLASS];
  memset(test_label, 0, 10000*NUM_OF_CLASS*sizeof(float));
  read_test_data(argv[3], test_data, test_label);

  int batch_num = 10000/BATCH;

#ifdef GPU
  float *batch_xs = malloc_gpu(BATCH*IM_SIZE);
  float *batch_ys = malloc_gpu(BATCH*NUM_OF_CLASS);
#else
  float *batch_xs;
  float *batch_ys;
#endif

  float total = 0.0;
  ms_t start = getms();

  for(int iter = 0; iter < batch_num; iter++) {
    int step = (iter*BATCH)%10000;
    
#ifdef GPU
      gpu_push_array(batch_xs, test_data + step*IM_SIZE, BATCH*IM_SIZE);
      gpu_push_array(batch_ys, test_label + step*NUM_OF_CLASS, BATCH*NUM_OF_CLASS);
#else
      batch_xs = test_data + step*IM_SIZE;
      batch_ys = test_label + step*NUM_OF_CLASS;
#endif

    network.Inference(batch_xs);
    float *output = network.output();
    total += accuracy(BATCH, NUM_OF_CLASS, output, batch_ys); 
  }

  cout << "Validation accuracy = " << (total/(float)batch_num)
       << ", time = " << (getms() - start) << endl;

#ifdef GPU
    cudaFree(batch_xs);
    cudaFree(batch_ys);
#endif
    delete []test_data;
    delete []test_label;

  return 0;


}
