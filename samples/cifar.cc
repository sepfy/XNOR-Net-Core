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


#define LEARNING_RATE 1.0e-1
#define BATCH 32
#define MAX_ITER 80000

void CifarXnorNet(Network *network) {

  Convolution *conv1 = new Convolution(32, 32, 3, 5, 5, 20, 1, false);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(28*28*20);
  Relu *relu1 = new Relu(28*28*20, LEAKY);
  Pooling *pool1 = new Pooling(28, 28, 20, 2, 2, 20, 2, false);

  Convolution *conv2 = new Convolution(14, 14, 20, 5, 5, 50, 1, false);
  Batchnorm *bn2 = new Batchnorm(14*14*20);
  Relu *relu2 = new Relu(10*10*50, LEAKY);
  Pooling *pool2 = new Pooling(10, 10, 50, 2, 2, 50, 2, false);

  Convolution *conv3 = new Convolution(5, 5, 50, 5, 5, 500, 1, false);
  Batchnorm *bn3 = new Batchnorm(5*5*50);
  Relu *relu3 = new Relu(500, LEAKY);
  Dropout *dropout3 = new Dropout(500, 0.5);

  Connected *conn4 = new Connected(500, 10);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);

  
  network->add(conv1);
  network->add(bn1);
  network->add(relu1);
  network->add(pool1);

  network->add(bn2);
  network->add(conv2);
  network->add(relu2);
  network->add(pool2);

  network->add(bn3);
  network->add(conv3);
  network->add(relu3);

  network->add(dropout3);
  network->add(conn4);
  network->add(softmax);

}

void CifarDarkNet(Network *network) {

  Convolution *conv1 = new Convolution(32, 32, 3, 3, 3, 128, 1, true);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(32*32*128);
  Relu *relu1 = new Relu(32*32*128, LEAKY);

  Convolution *conv2 = new Convolution(32, 32, 128, 3, 3, 128, 1, true);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(32*32*128);
  Relu *relu2 = new Relu(32*32*128, LEAKY);

  Convolution *conv3 = new Convolution(32, 32, 128, 3, 3, 128, 1, true);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(32*32*128);
  Shortcut *shortcut1 = new Shortcut(32, 32, 128, conv1, relu1);
  Relu *relu3 = new Relu(32*32*128, LEAKY);

  Pooling *pool1 = new Pooling(32, 32, 128, 2, 2, 128, 2, false);

  Convolution *conv4 = new Convolution(16, 16, 128, 3, 3, 256, 1, true);
  conv4->xnor = false;
  Batchnorm *bn4 = new Batchnorm(16*16*256);
  Relu *relu4 = new Relu(16*16*256, LEAKY);

  Convolution *conv5 = new Convolution(16, 16, 256, 3, 3, 256, 1, true);
  conv5->xnor = false;
  Batchnorm *bn5 = new Batchnorm(16*16*256);
  Relu *relu5 = new Relu(16*16*256, LEAKY);

  Convolution *conv6 = new Convolution(16, 16, 256, 3, 3, 256, 1, true);
  conv6->xnor = false;
  Batchnorm *bn6 = new Batchnorm(16*16*256);
  Relu *relu6 = new Relu(16*16*256, LEAKY);

  Pooling *pool2 = new Pooling(16, 16, 256, 2, 2, 256, 2, false);

  Convolution *conv7 = new Convolution(8, 8, 256, 3, 3, 512, 1, true);
  conv7->xnor = false;
  Batchnorm *bn7 = new Batchnorm(8*8*512);
  Relu *relu7 = new Relu(8*8*512, LEAKY);

  Convolution *conv8 = new Convolution(8, 8, 512, 3, 3, 512, 1, true);
  conv8->xnor = false;
  Batchnorm *bn8 = new Batchnorm(8*8*512);
  Relu *relu8 = new Relu(8*8*512, LEAKY);

  Convolution *conv9 = new Convolution(8, 8, 512, 3, 3, 512, 1, true);
  conv9->xnor = false;
  Batchnorm *bn9 = new Batchnorm(8*8*512);
  Relu *relu9 = new Relu(8*8*512, LEAKY);

  Convolution *conv10 = new Convolution(8, 8, 512, 3, 3, 10, 1, true);
  conv10->xnor = false;
  Batchnorm *bn10 = new Batchnorm(8*8*10);
  Relu *relu10 = new Relu(8*8*10, LEAKY);

  AvgPool *avgpool1 = new AvgPool(8, 8, 10, 8, 8, 10, 1, false);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);

  network->add(conv1);
  network->add(relu1);
  network->add(conv2);
  network->add(relu2);
  network->add(conv3);
  network->add(relu3);
  network->add(pool1);
  network->add(conv4);
  network->add(relu4);
  network->add(conv5);
  network->add(relu5);
  network->add(conv6);
  network->add(relu6);
  network->add(pool2);

  network->add(conv7);
  //network->add(bn7);
  network->add(relu7);

  network->add(conv8);
  //network->add(bn8);
  network->add(relu8);

  network->add(conv9);
  //network->add(bn9);
  network->add(relu9);

  network->add(conv10);
  //network->add(bn10);
  network->add(relu10);

  network->add(avgpool1);
  network->add(softmax);



}

void ResNet(Network *network) {

  Convolution *conv1 = new Convolution(32, 32, 3, 3, 3, 64, 1, true);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(32*32*64);
  Relu *relu1 = new Relu(32*32*64, LEAKY);

  // Residual Block 1
  Convolution *conv2 = new Convolution(32, 32, 64, 3, 3, 64, 1, true);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(32*32*64);
  Relu *relu2 = new Relu(32*32*64, LEAKY);

  Convolution *conv3 = new Convolution(32, 32, 64, 3, 3, 64, 1, true);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(32*32*64);
  Shortcut *shortcut1 = new Shortcut(32, 32, 64, conv1, relu1);
  Relu *relu3 = new Relu(32*32*64, LEAKY);


  // Residual Block 2
  Convolution *conv4 = new Convolution(32, 32, 64, 3, 3, 64, 1, true);
  conv4->xnor = false;
  Batchnorm *bn4 = new Batchnorm(32*32*64);
  Relu *relu4 = new Relu(32*32*64, LEAKY);

  Convolution *conv5 = new Convolution(32, 32, 64, 3, 3, 64, 1, true);
  conv5->xnor = false;
  Batchnorm *bn5 = new Batchnorm(32*32*64);
  Shortcut *shortcut2 = new Shortcut(32, 32, 64, conv3, relu3);
  Relu *relu5 = new Relu(32*32*64, LEAKY);

  Pooling *pool1 = new Pooling(32, 32, 64, 2, 2, 64, 2, false);

  // Residual Block 3
  Convolution *conv6 = new Convolution(16, 16, 64, 3, 3, 128, 1, true);
  conv6->xnor = false;
  Batchnorm *bn6 = new Batchnorm(16*16*128);
  Relu *relu6 = new Relu(16*16*128, LEAKY);

  Convolution *conv7 = new Convolution(16, 16, 128, 3, 3, 128, 1, true);
  conv7->xnor = false;
  Batchnorm *bn7 = new Batchnorm(16*16*128);
  Relu *relu7 = new Relu(16*16*128, LEAKY);

  // Residual Block 4
  Convolution *conv8 = new Convolution(16, 16, 128, 3, 3, 128, 1, true);
  conv8->xnor = false;
  Batchnorm *bn8 = new Batchnorm(16*16*128);
  Relu *relu8 = new Relu(16*16*128, LEAKY);

  Convolution *conv9 = new Convolution(16, 16, 128, 3, 3, 128, 1, true);
  conv9->xnor = false;
  Batchnorm *bn9 = new Batchnorm(16*16*128);
  Shortcut *shortcut4 = new Shortcut(16, 16, 128, conv7, relu7);
  Relu *relu9 = new Relu(16*16*128, LEAKY);

  Pooling *pool2 = new Pooling(16, 16, 128, 2, 2, 128, 2, false);

  // Residual Block 5
  Convolution *conv10 = new Convolution(8, 8, 128, 3, 3, 256, 1, true);
  conv10->xnor = false;
  Batchnorm *bn10 = new Batchnorm(8*8*256);
  Relu *relu10 = new Relu(8*8*256, LEAKY);

  Convolution *conv11 = new Convolution(8, 8, 256, 3, 3, 256, 1, true);
  conv11->xnor = false;
  Batchnorm *bn11 = new Batchnorm(8*8*256);
  Relu *relu11 = new Relu(8*8*256, LEAKY);

  // Residual Block 6 
  Convolution *conv12 = new Convolution(8, 8, 256, 3, 3, 256, 1, true);
  conv12->xnor = false;
  Batchnorm *bn12 = new Batchnorm(8*8*256);
  Relu *relu12 = new Relu(8*8*256, LEAKY);

  Convolution *conv13 = new Convolution(8, 8, 256, 3, 3, 256, 1, true);
  conv13->xnor = false;
  Batchnorm *bn13 = new Batchnorm(8*8*256);
  Shortcut *shortcut6 = new Shortcut(8, 8, 256, conv11, relu11);
  Relu *relu13 = new Relu(8*8*256, LEAKY);

  Pooling *pool3 = new Pooling(8, 8, 256, 2, 2, 256, 2, false);


  // Residual Block 7
  Convolution *conv14 = new Convolution(4, 4, 256, 3, 3, 512, 1, true);
  conv14->xnor = false;
  Batchnorm *bn14 = new Batchnorm(4*4*512);
  Relu *relu14 = new Relu(4*4*512, LEAKY);

  Convolution *conv15 = new Convolution(4, 4, 512, 3, 3, 512, 1, true);
  conv15->xnor = false;
  Batchnorm *bn15 = new Batchnorm(4*4*512);
  Relu *relu15 = new Relu(4*4*512, LEAKY);

  // Residual Block 8
  Convolution *conv16 = new Convolution(4, 4, 512, 3, 3, 512, 1, true);
  conv16->xnor = false;
  Batchnorm *bn16 = new Batchnorm(4*4*512);
  Relu *relu16 = new Relu(4*4*512, LEAKY);

  Convolution *conv17 = new Convolution(4, 4, 512, 3, 3, 512, 1, true);
  conv17->xnor = false;
  Batchnorm *bn17 = new Batchnorm(4*4*512);
  Shortcut *shortcut8 = new Shortcut(4, 4, 512, conv15, relu15);
  Relu *relu17 = new Relu(4*4*512, LEAKY);

  AvgPool *avgpool1 = new AvgPool(4, 4, 512, 4, 4, 512, 1, false);
  //Pooling *avgpool1 = new Pooling(4, 4, 512, 2, 2, 512, 2, false);
  Connected *conn = new Connected(512, 10);
  //Connected *conn = new Connected(512, 10);

  //Pooling *pool4 = new Pooling(4, 4, 512, 2, 2, 512, 2, false);

  //Connected *conn5 = new Connected(2*2*512, 500);


  //Convolution *conv44 = new Convolution(4, 4, 256, 4, 4, 500, 1, false);
  //conv44->xnor = false;
  //Relu *relu44 = new Relu(500, LEAKY);

  //Relu *relu18 = new Relu(500, RELU);
  //Connected *conn6 = new Connected(500, 10);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);

  
  network->add(conv1);
  network->add(bn1);
  network->add(relu1);

  network->add(conv2);
  network->add(bn2);
  network->add(relu2);

  network->add(conv3);
  network->add(bn3);
  network->add(shortcut1);
  network->add(relu3);

  network->add(conv4);
  network->add(bn4);
  network->add(relu4);

  network->add(conv5);
  network->add(bn5);
  network->add(shortcut2);
  network->add(relu5);

  network->add(pool1);


  network->add(conv6);
  network->add(bn6);
  network->add(relu6);

  network->add(conv7);
  network->add(bn7);
  network->add(relu7);

  network->add(conv8);
  network->add(bn8);
  network->add(relu8);

  network->add(conv9);
  network->add(bn9);
  network->add(shortcut4);
  network->add(relu9);

  network->add(pool2);

  network->add(conv10);
  network->add(bn10);
  network->add(relu10);

  network->add(conv11);
  network->add(bn11);
  network->add(relu11);

  network->add(conv12);
  network->add(bn12);
  network->add(relu12);

  network->add(conv13);
  network->add(bn13);
  network->add(shortcut6);
  network->add(relu13);

  network->add(pool3);

  network->add(conv14);
  network->add(bn14);
  network->add(relu14);

  network->add(conv15);
  network->add(bn15);
  network->add(relu15);

  network->add(conv16);
  network->add(bn16);
  network->add(relu16);

  network->add(conv17);
  network->add(bn17);
  network->add(shortcut8);
  network->add(relu17);

  //network->add(pool4);
  network->add(avgpool1);
  network->add(conn);

  //network->add(conn5);
  //network->add(relu18);
  //network->add(conn6);


  //network->add(conv44);
  //network->add(bn44);
  //network->add(relu44);
  //network->add(conn4);
  network->add(softmax);

}



void CifarNet(Network *network) {

  Convolution *conv1 = new Convolution(32, 32, 3, 5, 5, 20, 1, false);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(28*28*20);
  Relu *relu1 = new Relu(28*28*20, LEAKY);
  Pooling *pool1 = new Pooling(28, 28, 20, 2, 2, 20, 2, false);

  Convolution *conv2 = new Convolution(14, 14, 20, 5, 5, 50, 1, false);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(10*10*50);
  Relu *relu2 = new Relu(10*10*50, LEAKY);
  Pooling *pool2 = new Pooling(10, 10, 50, 2, 2, 50, 2, false);

  Convolution *conv3 = new Convolution(5, 5, 50, 5, 5, 500, 1, false);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(500);
  Relu *relu3 = new Relu(500, LEAKY);
  Dropout *dropout3 = new Dropout(500, 0.5);

  Connected *conn4 = new Connected(500, 10);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);

  
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

  //network->add(dropout3);
  network->add(conn4);
  network->add(softmax);

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

    //CifarXnorNet(&network);
    //CifarNet(&network);
    CifarDarkNet(&network);
    //ResNet(&network);
    network.initial(BATCH, LEARNING_RATE);
    float *train_data, *train_label;

    /*
#ifdef GPU
    float *train_data_tmp = new float[50000*IM_SIZE];
    float *train_label_tmp = new float[50000*NUM_OF_CLASS];
    read_train_data(argv[3], train_data_tmp, train_label_tmp);

    train_data = malloc_gpu(50000*IM_SIZE);
    train_label = malloc_gpu(50000*NUM_OF_CLASS);

    delete []train_data_tmp;
    delete []train_label_tmp;

#else
    */
    train_data = new float[50000*IM_SIZE];
    train_label = new float[50000*NUM_OF_CLASS];
    read_train_data(argv[3], train_data, train_label);
//#endif
/*
float *A;
float *B;
float *C;
A= malloc_gpu(2048);
B= malloc_gpu(512);
C= malloc_gpu(2304);

gemm_gpu(TRS_N, TRS_N, 2408, 512, 2304, 1, A, B, C);
*/


    float *batch_xs = malloc_gpu(BATCH*IM_SIZE);
    float *batch_ys = malloc_gpu(BATCH*NUM_OF_CLASS);
    for(int iter = 0; iter < MAX_ITER; iter++) {

      ms_t start = getms();
      int step = (iter*BATCH)%50000;
      //float *batch_xs = train_data + step*IM_SIZE;
      //float *batch_ys = train_label + step*NUM_OF_CLASS;

      gpu_push_array(batch_xs, train_data + step*IM_SIZE, BATCH*IM_SIZE);
      gpu_push_array(batch_ys, train_label + step*NUM_OF_CLASS, BATCH*NUM_OF_CLASS);
      float *output = network.inference(batch_xs);
      network.train(batch_ys);
//      float acc = accuracy(BATCH, NUM_OF_CLASS, output, batch_ys); 

      if(iter%1 == 0) {
        float loss = cross_entropy(BATCH, NUM_OF_CLASS, output, batch_ys);
        cout << "iter = " << iter << ", time = " << (getms() - start) << "ms, loss = "
         << loss << endl;
      }
    }

    network.save(argv[2]);
#ifdef GPU
    cudaFree(train_data);
    cudaFree(train_label);
#else
    delete []train_data;
    delete []train_label;
#endif
  }
  else if(strcmp(argv[1], "deploy") == 0) {
    network.load(argv[2], BATCH);
  } 
  else {
    help();
  }

 


  float *test_data, *test_label;

#ifdef GPU
  test_data = malloc_gpu(10000*IM_SIZE);
  test_label = malloc_gpu(10000*NUM_OF_CLASS);
#else
  test_data = new float[10000*IM_SIZE];
  test_label = new float[10000*NUM_OF_CLASS];
#endif
  read_test_data(argv[3], test_data, test_label);

  float total = 0.0;
  ms_t start = getms();
  network.deploy();
  for(int iter = 0; iter < 200; iter++) {
    int step = (iter*BATCH)%10000;
    float *batch_xs = test_data + step*IM_SIZE;
    float *batch_ys = test_label + step*NUM_OF_CLASS;
    float *output = network.inference(batch_xs);

    total += accuracy(BATCH, NUM_OF_CLASS, output, batch_ys); 
  }

  cout << "Validation accuracy = " << (total/200.0) 
       << ", time = " << (getms() - start) << endl;

#ifdef GPU
  cudaFree(test_data);
  cudaFree(test_label);
#else
  delete []test_data;
  delete []test_label;
#endif

  return 0;


}
