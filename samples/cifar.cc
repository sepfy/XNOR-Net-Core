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
#define BATCH 50
#define MAX_ITER 10000

void CifarXnorNet(Network *network) {

  Convolution *conv1 = new Convolution(32, 32, 3, 5, 5, 20, 1, false);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(28*28*20);
  Activation *actv1 = new Activation(28*28*20, LEAKY);
  Maxpool *pool1 = new Maxpool(28, 28, 20, 2, 2, 20, 2, false);

  Convolution *conv2 = new Convolution(14, 14, 20, 5, 5, 50, 1, false);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(10*10*50);
  Activation *actv2 = new Activation(10*10*50, LEAKY);
  Maxpool *pool2 = new Maxpool(10, 10, 50, 2, 2, 50, 2, false);

  Convolution *conv3 = new Convolution(5, 5, 50, 5, 5, 500, 1, false);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(500);
  Activation *actv3 = new Activation(500, LEAKY);
  //Dropout *dropout3 = new Dropout(500, 0.5);

  Connected *conn4 = new Connected(32*32*3, 10);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);

/*
  network->Add(conv1);
  network->Add(bn1);
  network->Add(actv1);
  network->Add(pool1);

  network->Add(conv2);
  network->Add(bn2);
  network->Add(actv2);
  network->Add(pool2);

  network->Add(conv3);
  network->Add(bn3);
  network->Add(actv3);
*/
  //network->Add(dropout3);
  network->Add(conn4);
  network->Add(softmax);

}


void CifarNet(Network *network) {

  Convolution *conv1 = new Convolution(32, 32, 3, 3, 3, 128, 1, 1);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(32*32*128);
  Activation *actv1 = new Activation(32*32*128, LEAKY);

  Convolution *conv2 = new Convolution(32, 32, 128, 3, 3, 128, 1, 1);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(32*32*128);
  Activation *actv2 = new Activation(32*32*128, LEAKY);

  Convolution *conv3 = new Convolution(32, 32, 128, 3, 3, 128, 1, 1);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(32*32*128);
  Activation *actv3 = new Activation(32*32*128, LEAKY);

  Maxpool *pool1 = new Maxpool(32, 32, 128, 2, 2, 128, 2, false);
  Dropout *dropout1 = new Dropout(16*16*128, 0.5);

  Convolution *conv4 = new Convolution(16, 16, 128, 3, 3, 256, 1, 1);
  conv4->xnor = false;
  Batchnorm *bn4 = new Batchnorm(16*16*256);
  Activation *actv4 = new Activation(16*16*256, LEAKY);

  Convolution *conv5 = new Convolution(16, 16, 256, 3, 3, 256, 1, 1);
  conv5->xnor = false;
  Batchnorm *bn5 = new Batchnorm(16*16*256);
  Activation *actv5 = new Activation(16*16*256, LEAKY);

  Convolution *conv6 = new Convolution(16, 16, 256, 3, 3, 256, 1, 1);
  conv6->xnor = false;
  Batchnorm *bn6 = new Batchnorm(16*16*256);
  Activation *actv6 = new Activation(16*16*256, LEAKY);

  Maxpool *pool2 = new Maxpool(16, 16, 256, 2, 2, 256, 2, false);

  Dropout *dropout2 = new Dropout(8*8*256, 0.5);

  Convolution *conv7 = new Convolution(8, 8, 256, 3, 3, 512, 1, 1);
  conv7->xnor = false;
  Batchnorm *bn7 = new Batchnorm(8*8*512);
  Activation *actv7 = new Activation(8*8*512, LEAKY);

  Convolution *conv8 = new Convolution(8, 8, 512, 3, 3, 512, 1, 1);
  conv8->xnor = false;
  Batchnorm *bn8 = new Batchnorm(8*8*512);
  Activation *actv8 = new Activation(8*8*512, LEAKY);

  Convolution *conv9 = new Convolution(8, 8, 512, 3, 3, 512, 1, 1);
  conv9->xnor = false;
  Batchnorm *bn9 = new Batchnorm(8*8*512);
  Activation *actv9 = new Activation(8*8*512, LEAKY);

  Dropout *dropout3 = new Dropout(8*8*512, 0.5);

  Convolution *conv10 = new Convolution(8, 8, 512, 3, 3, 10, 1, 1);
  conv10->xnor = false;
  Batchnorm *bn10 = new Batchnorm(8*8*10);
  Activation *actv10 = new Activation(8*8*10, LEAKY);
  Connected *conn = new Connected(8*8*10, 10);
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
  
  //network->Add(avgpool);
  network->Add(conn);
  network->Add(softmax);
}

void ResNet(Network *network) {

  Convolution *conv1 = new Convolution(32, 32, 3, 3, 3, 64, 1, 1);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(32*32*64);
  Activation *actv1 = new Activation(32*32*64, LEAKY);

  // Residual Block 1
  Convolution *conv2 = new Convolution(32, 32, 64, 3, 3, 64, 1, 1);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(32*32*64);
  Activation *actv2 = new Activation(32*32*64, LEAKY);

  Convolution *conv3 = new Convolution(32, 32, 64, 3, 3, 64, 1, 1);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(32*32*64);
  Shortcut *shortcut1 = new Shortcut(32, 32, 64, 32, 32, 64, -6, actv1);
  Activation *actv3 = new Activation(32*32*64, LEAKY);


  // Residual Block 2
  Convolution *conv4 = new Convolution(32, 32, 64, 3, 3, 64, 1, 1);
  conv4->xnor = false;
  Batchnorm *bn4 = new Batchnorm(32*32*64);
  Activation *actv4 = new Activation(32*32*64, LEAKY);

  Convolution *conv5 = new Convolution(32, 32, 64, 3, 3, 64, 1, 1);
  conv5->xnor = false;
  Batchnorm *bn5 = new Batchnorm(32*32*64);
  Shortcut *shortcut2 = new Shortcut(32, 32, 64, 32, 32, 64, -6, actv3);
  Activation *actv5 = new Activation(32*32*64, LEAKY);

  //Maxpool *pool1 = new Maxpool(32, 32, 64, 2, 2, 64, 2, false);

  // Residual Block 3
  Convolution *conv6 = new Convolution(32, 32, 64, 3, 3, 128, 2, 1);
  conv6->xnor = false;
  Batchnorm *bn6 = new Batchnorm(16*16*128);
  Activation *actv6 = new Activation(16*16*128, LEAKY);

  Convolution *conv7 = new Convolution(16, 16, 128, 3, 3, 128, 1, 1);
  conv7->xnor = false;
  Batchnorm *bn7 = new Batchnorm(16*16*128);
  Shortcut *shortcut3 = new Shortcut(32, 32, 64, 16, 16, 128, -6, actv5);
  Activation *actv7 = new Activation(16*16*128, LEAKY);

  // Residual Block 4
  Convolution *conv8 = new Convolution(16, 16, 128, 3, 3, 128, 1, 1);
  conv8->xnor = false;
  Batchnorm *bn8 = new Batchnorm(16*16*128);
  Activation *actv8 = new Activation(16*16*128, LEAKY);

  Convolution *conv9 = new Convolution(16, 16, 128, 3, 3, 128, 1, 1);
  conv9->xnor = false;
  Batchnorm *bn9 = new Batchnorm(16*16*128);
  Shortcut *shortcut4 = new Shortcut(16, 16, 128, 16, 16, 128, -6, actv7);
  Activation *actv9 = new Activation(16*16*128, LEAKY);

  //Maxpool *pool2 = new Maxpool(16, 16, 128, 2, 2, 128, 2, false);

  // Residual Block 5
  Convolution *conv10 = new Convolution(16, 16, 128, 3, 3, 256, 2, 1);
  conv10->xnor = false;
  Batchnorm *bn10 = new Batchnorm(8*8*256);
  Activation *actv10 = new Activation(8*8*256, LEAKY);

  Convolution *conv11 = new Convolution(8, 8, 256, 3, 3, 256, 1, 1);
  conv11->xnor = false;
  Batchnorm *bn11 = new Batchnorm(8*8*256);
  Shortcut *shortcut5 = new Shortcut(16, 16, 128, 8, 8, 256, -7, actv9);
  Activation *actv11 = new Activation(8*8*256, LEAKY);

  // Residual Block 6 
  Convolution *conv12 = new Convolution(8, 8, 256, 3, 3, 256, 1, 1);
  conv12->xnor = false;
  Batchnorm *bn12 = new Batchnorm(8*8*256);
  Activation *actv12 = new Activation(8*8*256, LEAKY);

  Convolution *conv13 = new Convolution(8, 8, 256, 3, 3, 256, 1, 1);
  conv13->xnor = false;
  Batchnorm *bn13 = new Batchnorm(8*8*256);
  Shortcut *shortcut6 = new Shortcut(8, 8, 256, 8, 8, 256, -6, actv11);
  Activation *actv13 = new Activation(8*8*256, LEAKY);

  //Maxpool *pool3 = new Maxpool(8, 8, 256, 2, 2, 256, 2, false);


  // Residual Block 7
  Convolution *conv14 = new Convolution(8, 8, 256, 3, 3, 512, 2, 1);
  conv14->xnor = false;
  Batchnorm *bn14 = new Batchnorm(4*4*512);
  Activation *actv14 = new Activation(4*4*512, LEAKY);

  Convolution *conv15 = new Convolution(4, 4, 512, 3, 3, 512, 1, 1);
  conv15->xnor = false;
  Batchnorm *bn15 = new Batchnorm(4*4*512);
  Shortcut *shortcut7 = new Shortcut(8, 8, 256, 4, 4, 512, -7, actv13);
  Activation *actv15 = new Activation(4*4*512, LEAKY);

  // Residual Block 8
  Convolution *conv16 = new Convolution(4, 4, 512, 3, 3, 512, 1, 1);
  conv16->xnor = false;
  Batchnorm *bn16 = new Batchnorm(4*4*512);
  Activation *actv16 = new Activation(4*4*512, LEAKY);

  Convolution *conv17 = new Convolution(4, 4, 512, 3, 3, 512, 1, 1);
  conv17->xnor = false;
  Batchnorm *bn17 = new Batchnorm(4*4*512);
  Shortcut *shortcut8 = new Shortcut(4, 4, 512, 4, 4, 512, -6, actv15);
  Activation *actv17 = new Activation(4*4*512, LEAKY);


  Avgpool *avgpool = new Avgpool(4, 4, 512, 4, 4, 512, 1, false);
  Connected *conn = new Connected(512, 10);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);

  
  network->Add(conv1);
  network->Add(bn1);
  network->Add(actv1);

  network->Add(conv2);
  network->Add(bn2);
  network->Add(actv2);

  network->Add(conv3);
  network->Add(bn3);
  network->Add(shortcut1);
  network->Add(actv3);

  network->Add(conv4);
  network->Add(bn4);
  network->Add(actv4);

  network->Add(conv5);
  network->Add(bn5);
  network->Add(shortcut2);
  network->Add(actv5);

//  network->Add(pool1);


  network->Add(conv6);
  network->Add(bn6);
  network->Add(actv6);

  network->Add(conv7);
  network->Add(bn7);
  network->Add(shortcut3);
  network->Add(actv7);

  network->Add(conv8);
  network->Add(bn8);
  network->Add(actv8);

  network->Add(conv9);
  network->Add(bn9);
  network->Add(shortcut4);
  network->Add(actv9);

 // network->Add(pool2);

  network->Add(conv10);
  network->Add(bn10);
  network->Add(actv10);

  network->Add(conv11);
  network->Add(bn11);
  network->Add(shortcut5);
  network->Add(actv11);

  network->Add(conv12);
  network->Add(bn12);
  network->Add(actv12);

  network->Add(conv13);
  network->Add(bn13);
  network->Add(shortcut6);
  network->Add(actv13);

//  network->Add(pool3);

  network->Add(conv14);
  network->Add(bn14);
  network->Add(actv14);

  network->Add(conv15);
  network->Add(bn15);
  network->Add(shortcut7);
  network->Add(actv15);

  network->Add(conv16);
  network->Add(bn16);
  network->Add(actv16);

  network->Add(conv17);
  network->Add(bn17);
  network->Add(shortcut8);
  network->Add(actv17);

  network->Add(avgpool);
  network->Add(conn);
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

    //CifarXnorNet(&network);
    CifarNet(&network);
    //ResNet(&network);
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
