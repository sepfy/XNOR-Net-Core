#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h> 
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "cifar.h"
#include "network.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{

  float *train_data, *train_label;

  train_data = new float[50000*IM_SIZE];
  train_label = new float[50000*NUM_OF_CLASS];
   
  read_train_data(train_data, train_label);

  
  //Mat im(32, 32, CV_8UC3);
  Mat im(32, 32, CV_32FC3);
  memcpy(im.data, train_data+100*IM_SIZE, IM_SIZE*sizeof(float)); 
  im = (im*127.5)+127.5;
  im.convertTo(im, CV_8UC3);
  cvtColor(im, im, CV_RGB2BGR);
  for(int i = 0; i < 10; i++)
    cout << train_label[i+100*NUM_OF_CLASS];
  cout << endl;
  //imshow("im", im);
  //waitKey(0); 
  
  int max_iter = 10000;
  float total_err = 0;
  int batch = 100;

  //lenet
  Convolution conv1(32, 32, 3, 3, 3, 64, 1, true);
  conv1.xnor = false;
  Relu relu1(32*32*64);

  Convolution conv11(32, 32, 64, 3, 3, 64, 1, true);
  conv11.xnor = false;
  Batchnorm bn11(32*32*64);
  Relu relu11(32*32*64);

  Convolution conv12(32, 32, 64, 3, 3, 64, 1, true);
  conv12.xnor = false;
  Batchnorm bn12(32*32*64);
  Shortcut shortcut1(32, 32, 64, &conv11, &relu11);
  Relu relu12(32*32*64);



  Convolution conv21(32, 32, 64, 3, 3, 64, 1, true);
  conv21.xnor = false;
  Batchnorm bn21(32*32*64);
  Relu relu21(32*32*64);

  Convolution conv22(32, 32, 64, 3, 3, 64, 1, true);
  conv22.xnor = false;
  Batchnorm bn22(32*32*64);
  Relu relu22(32*32*64);


  Pooling pool1(32, 32, 64, 2, 2, 64, 2, false);


  // residual block 1
  Convolution conv2(16, 16, 64, 3, 3, 128, 1, true);
  conv2.xnor = false;
  Relu relu2(16*16*128);





  Pooling pool2(16, 16, 128, 2, 2, 128, 2, false);

  Connected conn3(8*8*128, 128);
  Relu relu3(128);

  Connected conn4(128, 10);
  SoftmaxWithCrossEntropy softmax(10);

  Network network;


  network.add(&conv1);
  network.add(&relu1);


  network.add(&conv11);
  network.add(&bn11);
  network.add(&relu11);

  

  network.add(&conv12);
  network.add(&bn12);
  network.add(&shortcut1);
  network.add(&relu12);

/*
  network.add(&conv21);
  network.add(&bn21);
  network.add(&relu21);

  network.add(&conv22);
  network.add(&bn22);
  network.add(&relu12);
*/



  network.add(&pool1);



  network.add(&conv2);
  network.add(&relu2);
  network.add(&pool2);

  network.add(&conn3);
  network.add(&relu3);
  network.add(&conn4);
  network.add(&softmax);


/*

  Convolution conv(32, 32, 3, 3, 3, 64, 1, true);
  conv.xnor = false;
  Relu relu(32*32*64);

  // Residual Block 1
  Convolution conv11(32, 32, 64, 3, 3, 64, 1, true);
  conv11.xnor = false;
  Relu relu11(32*32*64);
  Convolution conv12(32, 32, 64, 3, 3, 64, 1, true);
  conv12.xnor = false;
  Relu relu12(32*32*64);

  // Residual Block 2
  Convolution conv21(32, 32, 64, 3, 3, 64, 1, true);
  conv21.xnor = false;
  Relu relu21(32*32*64);
  Convolution conv22(32, 32, 64, 3, 3, 64, 1, true);
  conv22.xnor = false;
  Relu relu22(32*32*64);

  // Residual Block 3
  Pooling pool3(32, 32, 64, 2, 2, 64, 2, false);
  Convolution conv31(16, 16, 64, 3, 3, 128, 1, true);
  conv31.xnor = false;
  Relu relu31(16*16*128);
  Convolution conv32(16, 16, 128, 3, 3, 128, 1, true);
  conv32.xnor = false;
  Batchnorm bn32(16*16*128);
  Relu relu32(16*16*128);

  // Residual Block 4
  Convolution conv41(16, 16, 128, 3, 3, 128, 1, true);
  conv41.xnor = false;
  Relu relu41(16*16*128);
  Convolution conv42(16, 16, 128, 3, 3, 128, 1, true);
  conv42.xnor = false;
  Relu relu42(16*16*128);

  // Residual Block 5
  Pooling pool5(16, 16, 128, 2, 2, 128, 2, false);
  Convolution conv51(8, 8, 128, 3, 3, 256, 1, true);
  conv51.xnor = false;
  Relu relu51(8*8*256);
  Convolution conv52(8, 8, 256, 3, 3, 256, 1, true);
  conv52.xnor = false;
  Relu relu52(8*8*256);

  // Residual Block 6
  Convolution conv61(8, 8, 256, 3, 3, 256, 1, true);
  conv61.xnor = false;
  Relu relu61(8*8*256);
  Convolution conv62(8, 8, 256, 3, 3, 256, 1, true);
  conv62.xnor = false;
  Batchnorm bn62(8*8*256);
  Relu relu62(8*8*256);

  // Residual Block 7
  Pooling pool7(8, 8, 256, 2, 2, 256, 2, false);
  Convolution conv71(4, 4, 256, 3, 3, 512, 1, true);
  conv71.xnor = false;
  Relu relu71(4*4*512);
  Convolution conv72(4, 4, 512, 3, 3, 512, 1, true);
  conv72.xnor = false;
  Relu relu72(4*4*512);

  // Residual Block 8
  Convolution conv81(4, 4, 512, 3, 3, 512, 1, true);
  conv81.xnor = false;
  Relu relu81(4*4*512);
  Convolution conv82(4, 4, 512, 3, 3, 512, 1, true);
  conv82.xnor = false;
  Batchnorm bn82(4*4*512);
  Relu relu82(4*4*512);

  Pooling pool(4, 4, 512, 2, 2, 512, 2, false);
  Connected conn(2*2*512, 128);
  Relu relu_conn(128);


  Connected conn1(128, 10);
  SoftmaxWithCrossEntropy softmax(10);


  Network network;
  network.add(&conv);
  network.add(&relu);

  network.add(&conv11);
  network.add(&relu11);
  network.add(&conv12);
  network.add(&relu12);

  network.add(&conv21);
  network.add(&relu21);
  network.add(&conv22);
  network.add(&relu22);

  network.add(&pool3);
  network.add(&conv31);
  network.add(&relu31);
  network.add(&conv32);
  network.add(&bn32);
  network.add(&relu32);

  network.add(&conv41);
  network.add(&relu41);
  network.add(&conv42);
  network.add(&relu42);

  network.add(&pool5);
  network.add(&conv51);
  network.add(&relu51);
  network.add(&conv52);
  network.add(&relu52);

  network.add(&conv61);
  network.add(&relu61);
  network.add(&conv62);
  network.add(&bn62);
  network.add(&relu62);

  network.add(&pool7);
  network.add(&conv71);
  network.add(&relu71);
  network.add(&conv72);
  network.add(&relu72);

  network.add(&conv81);
  network.add(&relu81);
  network.add(&conv82);
  network.add(&bn82);
  network.add(&relu82);

  network.add(&pool);
  network.add(&conn);
  network.add(&relu_conn);
  network.add(&conn1);
  network.add(&softmax);
*/

  network.initial(batch, 0.001);
  for(int iter = 0; iter < max_iter; iter++) {
    ms_t start = getms();
    int step = (iter*batch)%50000;
    float *batch_xs = train_data + step*IM_SIZE;
    float *batch_ys = train_label + step*NUM_OF_CLASS;
    float *output = network.inference(batch_xs);
    network.train(batch_ys);

    float loss = cross_entropy(batch, 10, output, batch_ys);
    float acc = accuracy(batch, 10, output, batch_ys); 
    if(iter%1 == 0) {
      cout << "iter = " << iter << ", time = " << (getms() - start) << "ms, loss = "
       << loss << " (accuracy = " << acc << ")" << endl;
    }
  }

  network.save();

  return 0;
}
