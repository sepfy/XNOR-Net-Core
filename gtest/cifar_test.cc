#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "network.h"
#include "cifar.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  
  int num = 100; 

  if(argc < 2) {
    cout << "Usage: ./cifar_test <cifar dataset>" << endl;
    exit(-1);
  }

  float *train_data, *train_label;
  train_data = new float[50000*IM_SIZE];
  train_label = new float[50000*NUM_OF_CLASS];
  read_train_data(argv[1], train_data, train_label);

 
  Mat im(32, 32, CV_32FC3);
  memcpy(im.data, train_data+num*IM_SIZE, IM_SIZE*sizeof(float)); 
  im = (im*127.5)+127.5;
  im.convertTo(im, CV_8UC3);
  cvtColor(im, im, CV_RGB2BGR);
  for(int i = 0; i < 10; i++)
    cout << train_label[i+num*NUM_OF_CLASS];
  cout << endl;
  imshow("im", im);
  waitKey(0); 

  return 0; 
}

