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
#define BATCH 30
#define MAX_ITER 10000

void Resnet18(Network *network) {

  Convolution *conv1 = new Convolution(225, 225, 3, 7, 7, 64, 2, true);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(112*112*64);
  Activation *actv1 = new Activation(112*112*64, LEAKY);
/*
  Convolution *conv1 = new Convolution(112, 112, 3, 3, 3, 64, 1, true);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(112*112*64);
  Activation *actv1 = new Activation(112*112*64, LEAKY);
*/

  Pooling *pool1 = new Pooling(112, 112, 64, 2, 2, 64, 2, false);

  // Residual Block 1
  Convolution *conv2 = new Convolution(56, 56, 64, 3, 3, 64, 1, true);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(56*56*64);
  Activation *actv2 = new Activation(56*56*64, LEAKY);

  Convolution *conv3 = new Convolution(56, 56, 64, 3, 3, 64, 1, true);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(56*56*64);
  Shortcut *shortcut1 = new Shortcut(56, 56, 64, conv1, actv1);
  Activation *actv3 = new Activation(56*56*64, LEAKY);


  // Residual Block 2
  Convolution *conv4 = new Convolution(56, 56, 64, 3, 3, 64, 1, true);
  conv4->xnor = false;
  Batchnorm *bn4 = new Batchnorm(56*56*64);
  Activation *actv4 = new Activation(56*56*64, LEAKY);

  Convolution *conv5 = new Convolution(56, 56, 64, 3, 3, 64, 1, true);
  conv5->xnor = false;
  Batchnorm *bn5 = new Batchnorm(56*56*64);
  Shortcut *shortcut2 = new Shortcut(56, 56, 64, conv3, actv3);
  Activation *actv5 = new Activation(56*56*64, LEAKY);

  Pooling *pool2 = new Pooling(56, 56, 64, 2, 2, 64, 2, false);

  // Residual Block 3
  Convolution *conv6 = new Convolution(28, 28, 64, 3, 3, 128, 1, true);
  conv6->xnor = false;
  Batchnorm *bn6 = new Batchnorm(28*28*128);
  Activation *actv6 = new Activation(28*28*128, LEAKY);

  Convolution *conv7 = new Convolution(28, 28, 128, 3, 3, 128, 1, true);
  conv7->xnor = false;
  Batchnorm *bn7 = new Batchnorm(28*28*128);
  Activation *actv7 = new Activation(28*28*128, LEAKY);

  // Residual Block 4
  Convolution *conv8 = new Convolution(28, 28, 128, 3, 3, 128, 1, true);
  conv8->xnor = false;
  Batchnorm *bn8 = new Batchnorm(28*28*128);
  Activation *actv8 = new Activation(28*28*128, LEAKY);

  Convolution *conv9 = new Convolution(28, 28, 128, 3, 3, 128, 1, true);
  conv9->xnor = false;
  Batchnorm *bn9 = new Batchnorm(28*28*128);
  Shortcut *shortcut4 = new Shortcut(28, 28, 128, conv7, actv7);
  Activation *actv9 = new Activation(28*28*128, LEAKY);

  Pooling *pool3 = new Pooling(28, 28, 128, 2, 2, 128, 2, false);

  // Residual Block 5
  Convolution *conv10 = new Convolution(14, 14, 128, 3, 3, 256, 1, true);
  conv10->xnor = false;
  Batchnorm *bn10 = new Batchnorm(14*14*256);
  Activation *actv10 = new Activation(14*14*256, LEAKY);

  Convolution *conv11 = new Convolution(14, 14, 256, 3, 3, 256, 1, true);
  conv11->xnor = false;
  Batchnorm *bn11 = new Batchnorm(14*14*256);
  Activation *actv11 = new Activation(14*14*256, LEAKY);

  // Residual Block 6 
  Convolution *conv12 = new Convolution(14, 14, 256, 3, 3, 256, 1, true);
  conv12->xnor = false;
  Batchnorm *bn12 = new Batchnorm(14*14*256);
  Activation *actv12 = new Activation(14*14*256, LEAKY);

  Convolution *conv13 = new Convolution(14, 14, 256, 3, 3, 256, 1, true);
  conv13->xnor = false;
  Batchnorm *bn13 = new Batchnorm(14*14*256);
  Shortcut *shortcut6 = new Shortcut(14, 14, 256, conv11, actv11);
  Activation *actv13 = new Activation(14*14*256, LEAKY);

  Pooling *pool4 = new Pooling(14, 14, 256, 2, 2, 256, 2, false);


  // Residual Block 7
  Convolution *conv14 = new Convolution(7, 7, 256, 3, 3, 512, 1, true);
  conv14->xnor = false;
  Batchnorm *bn14 = new Batchnorm(7*7*512);
  Activation *actv14 = new Activation(7*7*512, LEAKY);

  Convolution *conv15 = new Convolution(7, 7, 512, 3, 3, 512, 1, true);
  conv15->xnor = false;
  Batchnorm *bn15 = new Batchnorm(7*7*512);
  Activation *actv15 = new Activation(7*7*512, LEAKY);

  // Residual Block 8
  Convolution *conv16 = new Convolution(7, 7, 512, 3, 3, 512, 1, true);
  conv16->xnor = false;
  Batchnorm *bn16 = new Batchnorm(7*7*512);
  Activation *actv16 = new Activation(7*7*512, LEAKY);

  Convolution *conv17 = new Convolution(7, 7, 512, 3, 3, 512, 1, true);
  conv17->xnor = false;
  Batchnorm *bn17 = new Batchnorm(7*7*512);
  Shortcut *shortcut8 = new Shortcut(7, 7, 512, conv15, actv15);
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
  network->add(shortcut1);
  network->add(actv3);

  network->add(conv4);
  network->add(bn4);
  network->add(actv4);

  network->add(conv5);
  network->add(bn5);
  network->add(shortcut2);
  network->add(actv5);

  network->add(pool2);

  network->add(conv6);
  network->add(bn6);
  network->add(actv6);

  network->add(conv7);
  network->add(bn7);
  network->add(actv7);

  network->add(conv8);
  network->add(bn8);
  network->add(actv8);

  network->add(conv9);
  network->add(bn9);
  network->add(shortcut4);
  network->add(actv9);

  network->add(pool3);

  network->add(conv10);
  network->add(bn10);
  network->add(actv10);

  network->add(conv11);
  network->add(bn11);
  network->add(actv11);

  network->add(conv12);
  network->add(bn12);
  network->add(actv12);

  network->add(conv13);
  network->add(bn13);
  network->add(shortcut6);
  network->add(actv13);

  network->add(pool4);

  network->add(conv14);
  network->add(bn14);
  network->add(actv14);

  network->add(conv15);
  network->add(bn15);
  network->add(actv15);

  network->add(conv16);
  network->add(bn16);
  network->add(actv16);

  network->add(conv17);
  network->add(bn17);
  network->add(shortcut8);
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
  Resnet18(&network);
  network.initial(BATCH, LEARNING_RATE);

  float *inputs, *outputs;
  int n = read_data("FIRE-SMOKE-DATASET/Train/", inputs, outputs);
  cout << "Train set = " << n << endl;
  //show_image(35, inputs, outputs);
#ifdef GPU    
    float *batch_xs = malloc_gpu(BATCH*IM_SIZE);
    float *batch_ys = malloc_gpu(BATCH*NUM_OF_CLASS);
#else
    float *batch_xs;
    float *batch_ys;
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
/*
    //network.save(argv[2]);
#ifdef GPU
    cudaFree(batch_xs);
    cudaFree(batch_ys);
#else
    delete []batch_xs;
    delete []batch_ys;
#endif
    delete []train_data;
    delete []train_label;
  }
  else if(strcmp(argv[1], "deploy") == 0) {
    network.load(argv[2], BATCH);
  } 
  else {
    help();
  }

 


  float *test_data, *test_label;

  test_data = new float[10000*IM_SIZE];
  test_label = new float[10000*NUM_OF_CLASS];
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
  //network.deploy();

  for(int iter = 0; iter < batch_num; iter++) {
    int step = (iter*BATCH)%10000;

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
#else
    delete []batch_xs;
    delete []batch_ys;
#endif
    delete []test_data;
    delete []test_label;

  return 0;
*/

}
