#include <iostream>
#include <dirent.h> 
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "resnet.h"

using namespace std;


#define LEARNING_RATE 1.0e-4
#define BATCH 60
#define MAX_ITER 5000


void TinyResnetXnor(Network *network) {

  Convolution *conv1 = new Convolution(224, 224, 3, 2, 2, 32, 2, 0);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(112*112*32);
  Activation *actv1 = new Activation(112*112*32, LEAKY);

  Maxpool *pool1 = new Maxpool(112, 112, 32, 2, 2, 32, 2, false);

  Batchnorm *bn2 = new Batchnorm(56*56*32);
  Convolution *conv2 = new Convolution(56, 56, 32, 3, 3, 64, 1, 1);

  Batchnorm *bn3 = new Batchnorm(56*56*64);
  Convolution *conv3 = new Convolution(56, 56, 64, 3, 3, 64, 1, 1);

  Batchnorm *bn4 = new Batchnorm(56*56*64);
  Convolution *conv4 = new Convolution(56, 56, 64, 3, 3, 128, 2, 1);

  Batchnorm *bn5 = new Batchnorm(28*28*128);
  Convolution *conv5 = new Convolution(28, 28, 128, 3, 3, 128, 1, 1);

  Batchnorm *bn6 = new Batchnorm(28*28*128);
  Convolution *conv6 = new Convolution(28, 28, 128, 3, 3, 256, 2, 1);

  Batchnorm *bn7 = new Batchnorm(14*14*256);
  Convolution *conv7 = new Convolution(14, 14, 256, 3, 3, 256, 1, 1);

  Batchnorm *bn8 = new Batchnorm(14*14*256);
  Convolution *conv8 = new Convolution(14, 14, 256, 3, 3, 512, 2, 1);

  Batchnorm *bn9 = new Batchnorm(7*7*512);
  Convolution *conv9 = new Convolution(7, 7, 512, 3, 3, 512, 1, 1);

  Avgpool *avgpool1 = new Avgpool(7, 7, 512, 7, 7, 512, 1, false);
  Connected *conn1 = new Connected(512, 3);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(3);


  network->Add(conv1);
  network->Add(bn1);
  network->Add(actv1);

  network->Add(pool1);

  network->Add(bn2);
  network->Add(conv2);

  //network->Add(bn3);
  //network->Add(conv3);

  network->Add(bn4);
  network->Add(conv4);

  //network->Add(bn5);
  //network->Add(conv5);

  network->Add(bn6);
  network->Add(conv6);

  //network->Add(bn7);
  //network->Add(conv7);

  network->Add(bn8);
  network->Add(conv8);

  //network->Add(bn9);
  //network->Add(conv9);

  network->Add(avgpool1);
  network->Add(conn1);
  network->Add(softmax);

}


void ResnetXnor18(Network *network) {


  Convolution *conv1 = new Convolution(224, 224, 3, 2, 2, 64, 2, 0);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(112*112*64);
  Activation *actv1 = new Activation(112*112*64, LEAKY);

  Maxpool *pool1 = new Maxpool(112, 112, 64, 2, 2, 64, 2, false);

  Batchnorm *bn2 = new Batchnorm(56*56*64);
  Convolution *conv2 = new Convolution(56, 56, 64, 3, 3, 64, 1, 1);

  Batchnorm *bn3 = new Batchnorm(56*56*64);
  Convolution *conv3 = new Convolution(56, 56, 64, 3, 3, 64, 1, 1);

  Batchnorm *bn4 = new Batchnorm(56*56*64);
  Convolution *conv4 = new Convolution(56, 56, 64, 3, 3, 64, 1, 1);

  Batchnorm *bn5 = new Batchnorm(56*56*64);
  Convolution *conv5 = new Convolution(56, 56, 64, 3, 3, 64, 1, 1);
  //Shortcut *shortcut5 = new Shortcut(56, 56, 64, 56, 56, 64, -6, actv3);


  Batchnorm *bn6 = new Batchnorm(56*56*64);
  Convolution *conv6 = new Convolution(56, 56, 64, 3, 3, 128, 2, 1);


  Batchnorm *bn7 = new Batchnorm(28*28*128);
  Convolution *conv7 = new Convolution(28, 28, 128, 3, 3, 128, 1, 1);
  //Shortcut *shortcut7 = new Shortcut(56, 56, 64, 28, 28, 128, -6, actv5);

  Batchnorm *bn8 = new Batchnorm(28*28*128);
  Convolution *conv8 = new Convolution(28, 28, 128, 3, 3, 128, 1, 1);

  Batchnorm *bn9 = new Batchnorm(28*28*128);
  Convolution *conv9 = new Convolution(28, 28, 128, 3, 3, 128, 1, 1);
  //Shortcut *shortcut9 = new Shortcut(28, 28, 128, 28, 28, 128, -6, actv7);

  Batchnorm *bn10 = new Batchnorm(28*28*128);
  Convolution *conv10 = new Convolution(28, 28, 128, 3, 3, 256, 2, 1);

  Batchnorm *bn11 = new Batchnorm(14*14*256);
  Convolution *conv11 = new Convolution(14, 14, 256, 3, 3, 256, 1, 1);
  //Shortcut *shortcut11 = new Shortcut(28, 28, 128, 14, 14, 256, -6, actv9);

  Batchnorm *bn12 = new Batchnorm(14*14*256);
  Convolution *conv12 = new Convolution(14, 14, 256, 3, 3, 256, 1, 1);

  Batchnorm *bn13 = new Batchnorm(14*14*256);
  Convolution *conv13 = new Convolution(14, 14, 256, 3, 3, 256, 1, 1);
  //Shortcut *shortcut13 = new Shortcut(14, 14, 256, 14, 14, 256, -6, actv11);

  Batchnorm *bn14 = new Batchnorm(14*14*256);
  Convolution *conv14 = new Convolution(14, 14, 256, 3, 3, 512, 2, 1);

  Batchnorm *bn15 = new Batchnorm(7*7*512);
  Convolution *conv15 = new Convolution(7, 7, 512, 3, 3, 512, 1, 1);
  //Shortcut *shortcut15 = new Shortcut(14, 14, 256, 7, 7,512, -6, actv13);

  Batchnorm *bn16 = new Batchnorm(7*7*512);
  Convolution *conv16 = new Convolution(7, 7, 512, 3, 3, 512, 1, 1);

  Batchnorm *bn17 = new Batchnorm(7*7*512);
  Convolution *conv17 = new Convolution(7, 7, 512, 3, 3, 512, 1, 1);
  //Shortcut *shortcut17 = new Shortcut(7, 7, 512, 7, 7,512, -6, actv15);

  Avgpool *avgpool1 = new Avgpool(7, 7, 512, 7, 7, 512, 1, false);
  Connected *conn1 = new Connected(512, 3);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(3);


  network->Add(conv1);
  network->Add(bn1);
  network->Add(actv1);

  network->Add(pool1);

  network->Add(bn2);
  network->Add(conv2);

  network->Add(bn3);
  network->Add(conv3);

  network->Add(bn4);
  network->Add(conv4);

  network->Add(bn5);
  network->Add(conv5);

  network->Add(bn6);
  network->Add(conv6);

  network->Add(bn7);
  network->Add(conv7);

  network->Add(bn8);
  network->Add(conv8);

  network->Add(bn9);
  network->Add(conv9);

  network->Add(bn10);
  network->Add(conv10);

  network->Add(bn11);
  network->Add(conv11);

  network->Add(bn12);
  network->Add(conv12);

  network->Add(bn13);
  network->Add(conv13);

  network->Add(bn14);
  network->Add(conv14);

  network->Add(bn15);
  network->Add(conv15);

  network->Add(bn16);
  network->Add(conv16);

  network->Add(bn17);
  network->Add(conv17);

  network->Add(avgpool1);
  network->Add(conn1);
  network->Add(softmax);

}



void Resnet18(Network *network) {

  Convolution *conv1 = new Convolution(224, 224, 3, 7, 7, 64, 2, 3);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(112*112*64);
  Activation *actv1 = new Activation(112*112*64, LEAKY);

  Maxpool *pool1 = new Maxpool(112, 112, 64, 2, 2, 64, 2, false);

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

  Avgpool *avgpool1 = new Avgpool(7, 7, 512, 7, 7, 512, 1, false);
  Connected *conn1 = new Connected(512, 3);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(3);

  
  network->Add(conv1);
  network->Add(bn1);
  network->Add(actv1);

  network->Add(pool1);

  network->Add(conv2);
  network->Add(bn2);
  network->Add(actv2);

  network->Add(conv3);
  network->Add(bn3);
  network->Add(actv3);

  network->Add(conv4);
  network->Add(bn4);
  network->Add(actv4);

  network->Add(conv5);
  network->Add(bn5);
  network->Add(shortcut5);
  network->Add(actv5);

  network->Add(conv6);
  network->Add(bn6);
  network->Add(actv6);

  network->Add(conv7);
  network->Add(bn7);
  //network->Add(shortcut7);
  network->Add(actv7);

  network->Add(conv8);
  network->Add(bn8);
  network->Add(actv8);

  network->Add(conv9);
  network->Add(bn9);
  network->Add(shortcut9);
  network->Add(actv9);


  network->Add(conv10);
  network->Add(bn10);
  network->Add(actv10);

  network->Add(conv11);
  network->Add(bn11);
  //network->Add(shortcut11);
  network->Add(actv11);

  network->Add(conv12);
  network->Add(bn12);
  network->Add(actv12);

  network->Add(conv13);
  network->Add(bn13);
  network->Add(shortcut13);
  network->Add(actv13);

  network->Add(conv14);
  network->Add(bn14);
  network->Add(actv14);

  network->Add(conv15);
  network->Add(bn15);
  network->Add(shortcut15);
  network->Add(actv15);

  network->Add(conv16);
  network->Add(bn16);
  network->Add(actv16);

  network->Add(conv17);
  network->Add(bn17);
  //network->Add(shortcut17);
  network->Add(actv17);

  network->Add(avgpool1);
  network->Add(conn1);
  network->Add(softmax);

}



void help() {
  cout << "Usage: ./cifar <train/deploy> <model name> <cifar dataset>" << endl; 
  exit(1);
}

int main( int argc, char** argv ) {

  Network network;

  if(strcmp(argv[1], "train") == 0) {

    TinyResnetXnor(&network);
    network.Init(BATCH, LEARNING_RATE, true);

    float *inputs, *outputs;
    char train_folder[64] = {0};
    sprintf(train_folder, "%s/train", argv[3]);
    int n = read_data(train_folder, inputs, outputs);
    cout << "train set = " << n << endl;
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
    delete []inputs;
    delete []outputs;
  }
  else if(strcmp(argv[1], "deploy") == 0) {
    network.Load(argv[2], BATCH);
  }
  else if(strcmp(argv[1], "eval") == 0) {

    network.Load(argv[2], 1);
    float *input = new float[IM_SIZE];
    eval_model(argv[3], input);
    ms_t start = getms();
    network.Inference(input);
    float *output = network.output();
    for(int i = 0; i < NUM_OF_CLASS; i++)
      cout << "Class " << i << ": " << output[i] << endl;
    cout << "time = " << (getms() - start) << endl;
    return 0;
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
  network.Deploy();

  for(int iter = 0; iter < batch_num; iter++) {
    int step = (iter*BATCH)%n;

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
