#include <iostream>
#include <dirent.h> 
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "resnet.h"

using namespace std;


#define LEARNING_RATE 1.0e-3
#define BATCH 15
#define MAX_ITER 1000

void Resnet18(Network *network) {

  Convolution *conv1 = new Convolution(225, 225, 3, 7, 7, 64, 2, 2);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(112*112*64);
  Activation *actv1 = new Activation(112*112*64, LEAKY);

  Pooling *pool1 = new Pooling(112, 112, 64, 2, 2, 64, 2, false);

  Convolution *conv2 = new Convolution(56, 56, 64, 3, 3, 64, 1, 1);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(56*56*64);
  Activation *actv2 = new Activation(56*56*64, LEAKY);

  Convolution *conv3 = new Convolution(56, 56, 64, 3, 3, 64, 1, 1);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(56*56*64);
  Activation *actv3 = new Activation(56*56*64, LEAKY);

  Convolution *conv4 = new Convolution(56, 56, 64, 3, 3, 64, 1, 1);
  conv4->xnor = false;
  Batchnorm *bn4 = new Batchnorm(56*56*64);
  Activation *actv4 = new Activation(56*56*64, LEAKY);

  Convolution *conv5 = new Convolution(56, 56, 64, 3, 3, 64, 1, 1);
  conv5->xnor = false;
  Batchnorm *bn5 = new Batchnorm(56*56*64);
  Shortcut *shortcut5 = new Shortcut(56, 56, 64, 56, 56, 64, -6, actv3);
  Activation *actv5 = new Activation(56*56*64, LEAKY);


  Convolution *conv6 = new Convolution(56, 56, 64, 3, 3, 128, 2, 1);
  conv6->xnor = false;
  Batchnorm *bn6 = new Batchnorm(28*28*128);
  Activation *actv6 = new Activation(28*28*128, LEAKY);

  Convolution *conv7 = new Convolution(28, 28, 128, 3, 3, 128, 1, 1);
  conv7->xnor = false;
  Batchnorm *bn7 = new Batchnorm(28*28*128);
  Shortcut *shortcut7 = new Shortcut(56, 56, 64, 28, 28, 128, -6, actv5);
  Activation *actv7 = new Activation(28*28*128, LEAKY);

  Convolution *conv8 = new Convolution(28, 28, 128, 3, 3, 128, 1, 1);
  conv8->xnor = false;
  Batchnorm *bn8 = new Batchnorm(28*28*128);
  Activation *actv8 = new Activation(28*28*128, LEAKY);

  Convolution *conv9 = new Convolution(28, 28, 128, 3, 3, 128, 1, 1);
  conv9->xnor = false;
  Batchnorm *bn9 = new Batchnorm(28*28*128);
  Shortcut *shortcut9 = new Shortcut(28, 28, 128, 28, 28, 128, -6, actv7);
  Activation *actv9 = new Activation(28*28*128, LEAKY);


  Convolution *conv10 = new Convolution(28, 28, 128, 3, 3, 256, 2, 1);
  conv10->xnor = false;
  Batchnorm *bn10 = new Batchnorm(14*14*256);
  Activation *actv10 = new Activation(14*14*256, LEAKY);

  Convolution *conv11 = new Convolution(14, 14, 256, 3, 3, 256, 1, 1);
  conv11->xnor = false;
  Batchnorm *bn11 = new Batchnorm(14*14*256);
  Shortcut *shortcut11 = new Shortcut(28, 28, 128, 14, 14, 256, -6, actv9);
  Activation *actv11 = new Activation(14*14*256, LEAKY);

  Convolution *conv12 = new Convolution(14, 14, 256, 3, 3, 256, 1, 1);
  conv12->xnor = false;
  Batchnorm *bn12 = new Batchnorm(14*14*256);
  Activation *actv12 = new Activation(14*14*256, LEAKY);

  Convolution *conv13 = new Convolution(14, 14, 256, 3, 3, 256, 1, 1);
  conv13->xnor = false;
  Batchnorm *bn13 = new Batchnorm(14*14*256);
  Shortcut *shortcut13 = new Shortcut(14, 14, 256, 14, 14, 256, -6, actv11);
  Activation *actv13 = new Activation(14*14*256, LEAKY);


  Convolution *conv14 = new Convolution(14, 14, 256, 3, 3, 512, 2, 1);
  conv14->xnor = false;
  Batchnorm *bn14 = new Batchnorm(7*7*512);
  Activation *actv14 = new Activation(7*7*512, LEAKY);

  Convolution *conv15 = new Convolution(7, 7, 512, 3, 3, 512, 1, 1);
  conv15->xnor = false;
  Batchnorm *bn15 = new Batchnorm(7*7*512);
  Shortcut *shortcut15 = new Shortcut(14, 14, 256, 7, 7,512, -6, actv13);
  Activation *actv15 = new Activation(7*7*512, LEAKY);

  Convolution *conv16 = new Convolution(7, 7, 512, 3, 3, 512, 1, 1);
  conv16->xnor = false;
  Batchnorm *bn16 = new Batchnorm(7*7*512);
  Activation *actv16 = new Activation(7*7*512, LEAKY);

  Convolution *conv17 = new Convolution(7, 7, 512, 3, 3, 512, 1, 1);
  conv17->xnor = false;
  Batchnorm *bn17 = new Batchnorm(7*7*512);
  Shortcut *shortcut17 = new Shortcut(7, 7, 512, 7, 7,512, -6, actv15);
  Activation *actv17 = new Activation(7*7*512, LEAKY);

  AvgPool *avgpool1 = new AvgPool(7, 7, 512, 7, 7, 512, 1, false);
  Connected *conn1 = new Connected(512, 3);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(3);

  
  network->add(conv1);
  network->add(bn1);
  network->add(actv1);

  network->add(pool1);

  network->add(conv2);
  network->add(bn2);
  network->add(actv2);

  network->add(conv3);
  network->add(bn3);
  network->add(actv3);

  network->add(conv4);
  network->add(bn4);
  network->add(actv4);

  network->add(conv5);
  network->add(bn5);
  network->add(shortcut5);
  network->add(actv5);

  network->add(conv6);
  network->add(bn6);
  network->add(actv6);

  network->add(conv7);
  network->add(bn7);
  //network->add(shortcut7);
  network->add(actv7);

  network->add(conv8);
  network->add(bn8);
  network->add(actv8);

  network->add(conv9);
  network->add(bn9);
  network->add(shortcut9);
  network->add(actv9);


  network->add(conv10);
  network->add(bn10);
  network->add(actv10);

  network->add(conv11);
  network->add(bn11);
  //network->add(shortcut11);
  network->add(actv11);

  network->add(conv12);
  network->add(bn12);
  network->add(actv12);

  network->add(conv13);
  network->add(bn13);
  network->add(shortcut13);
  network->add(actv13);

  network->add(conv14);
  network->add(bn14);
  network->add(actv14);

  network->add(conv15);
  network->add(bn15);
  network->add(shortcut15);
  network->add(actv15);

  network->add(conv16);
  network->add(bn16);
  network->add(actv16);

  network->add(conv17);
  network->add(bn17);
  //network->add(shortcut17);
  network->add(actv17);

  network->add(avgpool1);
  network->add(conn1);
  network->add(softmax);

}



void help() {
  cout << "Usage: ./cifar <train/deploy> <model name> <cifar dataset>" << endl; 
  exit(1);
}

int main( int argc, char** argv ) {

  Network network;

  if(strcmp(argv[1], "train") == 0) {

    Resnet18(&network);
    network.initial(BATCH, LEARNING_RATE, true);

    float *inputs, *outputs;
    char train_folder[64] = {0};
    sprintf(train_folder, "%s/Train", argv[3]);
    int n = read_data(train_folder, inputs, outputs);
    cout << "Train set = " << n << endl;
    //show_image(35, inputs, outputs);
    float *batch_xs;
    float *batch_ys;

#ifdef GPU    
    batch_xs = malloc_gpu(BATCH*IM_SIZE);
    batch_ys = malloc_gpu(BATCH*NUM_OF_CLASS);
#endif


    for(int iter = 0; iter < MAX_ITER; iter++) {

      ms_t start = getms();
      int step = (iter*BATCH)%n;
#ifdef GPU
      gpu_push_array(batch_xs, inputs + step*IM_SIZE, BATCH*IM_SIZE);
      gpu_push_array(batch_ys, outputs + step*NUM_OF_CLASS, BATCH*NUM_OF_CLASS);
#else
      batch_xs = inputs + step*IM_SIZE;
      batch_ys = outputs + step*NUM_OF_CLASS;
#endif

      float *output = network.inference(batch_xs);
      network.train(batch_ys);


      float acc = accuracy(BATCH, NUM_OF_CLASS, output, batch_ys); 

      if(iter%1 == 0) {
        float loss = cross_entropy(BATCH, NUM_OF_CLASS, output, batch_ys);
        cout << "iter = " << iter << ", time = " << (getms() - start) << "ms, loss = "
         << loss << " (acc = " << acc << ")" << endl;
      }

    }

    network.save(argv[2]);
#ifdef GPU
    cudaFree(batch_xs);
    cudaFree(batch_ys);
#endif
    delete []inputs;
    delete []outputs;
  }
  else if(strcmp(argv[1], "deploy") == 0) {
    network.load(argv[2], BATCH);
  } 
  else {
    help();
  }

 


  float *test_data, *test_label;

  char test_folder[64] = {0};
  sprintf(test_folder, "%s/Test", argv[3]);
  int n = read_data(test_folder, test_data, test_label);
  cout << "Test set = " << n << endl;

  int batch_num = n/BATCH;
  float *batch_xs;
  float *batch_ys;

#ifdef GPU
  batch_xs = malloc_gpu(BATCH*IM_SIZE);
  batch_ys = malloc_gpu(BATCH*NUM_OF_CLASS);
#endif

  float total = 0.0;
  ms_t start = getms();
  network.deploy();

  for(int iter = 0; iter < batch_num; iter++) {
    int step = (iter*BATCH)%n;

#ifdef GPU
      gpu_push_array(batch_xs, test_data + step*IM_SIZE, BATCH*IM_SIZE);
      gpu_push_array(batch_ys, test_label + step*NUM_OF_CLASS, BATCH*NUM_OF_CLASS);
#else
      batch_xs = test_data + step*IM_SIZE;
      batch_ys = test_label + step*NUM_OF_CLASS;
#endif

    float *output = network.inference(batch_xs);
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
