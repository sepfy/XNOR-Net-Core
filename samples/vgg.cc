#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h> 
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "network.h"

using namespace cv;
using namespace std;


int Load_images(char *dir, vector<Mat> *images) {
    
  struct dirent *dp;
  DIR *fd;
  if ((fd = opendir(dir)) == NULL) {
    fprintf(stderr, "listdir: can't open %s\n", dir);
    exit(0);
  }

  int count = 0;
  while ((dp = readdir(fd)) != NULL) {
  if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, ".."))
    continue;   
    Mat image;
    char filepath[257] = {0};
    sprintf(filepath, "%s/%s", dir, dp->d_name);
    image = imread(filepath, IMREAD_COLOR);
    resize(image, image, Size(224, 224));
    image.convertTo(image, CV_32FC1);
    image = (image - 127.5)/127.5;
    images->push_back(image);
    count++;
  }

  closedir(fd);
  return count;
}

int read_data(const char *basedir, float *&inputs, float *&outputs) {

  FILE *fp;
  char *line = NULL;
  size_t len = 0;
  fp = fopen("labels.txt", "r");
  if (fp == NULL) {
    cout << "Cannot find labels.txt" << endl;
    exit(1);
  }

  vector<Mat> images;
  vector<int> counts;
  int num_of_class = 0;
  while(getline(&line, &len, fp) != -1) {
    char filedir[256] = {0};
    line[strlen(line)-1] = '\0';
    sprintf(filedir, "%s/%s", basedir, line);
    int count = Load_images(filedir, &images);
    counts.push_back(count);
    num_of_class++;
  }

  inputs = new float[images.size()*224*224*3];
  outputs = new float[images.size()*num_of_class];
  memset(outputs, 0, images.size()*num_of_class);

  for(int i = 0; i < images.size(); i++) {
    memcpy(inputs+224*224*3*i, images[i].data, 224*224*3*sizeof(float));
  }

  int shift = 0;
  for(int i = 0; i < counts.size(); i++) {
    for(int j = 0; j < counts[i]; j++) {
      outputs[shift+counts.size()*j+i] = 1.0;
    }
    shift += counts[i]*num_of_class;
  }

  // recover image for test
  /*
  Mat image(224, 224, CV_32FC3);
  memcpy(image.data, inputs, 224*224*3*sizeof(float));
  image = (image*127.5) + 127.5;
  image.convertTo(image, CV_8UC3);
  imshow("Display window", image );                // Show our image inside it.
  waitKey(0); // Wait for a keystroke in the window
  */
  //cout << images.size() << endl;

  fclose(fp);
  if(line)
    free(line);
  return images.size();
}


void get_mini_batch(int n, int b, float *data, float *label, float *&batch_xs, float *&batch_ys) {

  srand(time(NULL));
  for(int i = 0; i < b; i++) {
    int p = rand()%n;
    memcpy(batch_xs + 224*223*3*i, data + p*224*224*3, 224*224*3*sizeof(float));
    memcpy(batch_ys + 3*i, label + p*3, 3*sizeof(float));
  }
}

int main( int argc, char** argv )
{

  float .train_data, .train_label;
  float *test_data, *test_label;
  int.train_num, test_num;
 .train_num = read_data("FIRE-SMOKE-DATASET.train",.train_data,.train_label);
  //test_num = read_data("FIRE-SMOKE-DATASET/Test/", test_data, test_label);
  cout <<.train_num << endl;
// ", " << test_num << endl;


  int max_iter = 3000;
  float total_err = 0;
  int batch = 20;

  Convolution conv1(224, 224, 3, 7, 7, 64, 2, true);
  conv1.xnor = false;
  Batchnorm bn1(112*112*64);
  Relu relu1(112*112*64);

  Pooling pool1(112, 112, 64, 2, 2, 64, 2, false);

  Convolution conv2_1(56, 56, 64, 3, 3, 64, 1, true);
  conv2_1.xnor = false;
  Batchnorm bn2_1(56*56*64);
  Relu relu2_1(56*56*64);

  Convolution conv2_2(56, 56, 64, 3, 3, 64, 1, true);
  conv2_2.xnor = false;
  Batchnorm bn2_2(56*56*64);
  Relu relu2_2(56*56*64);

  Convolution conv3_1(56, 56, 64, 3, 3, 128, 2, true);
  conv3_1.xnor = false;
  Batchnorm bn3_1(28*28*128);
  Relu relu3_1(28*28*128);

  Convolution conv3_2(28, 28, 128, 3, 3, 128, 1, true);
  Batchnorm bn3_2(28*28*128);
  conv3_2.xnor = false;
  Relu relu3_2(28*28*128);

  Convolution conv4_1(28, 28, 128, 3, 3, 256, 2, true);
  conv4_1.xnor = false;
  Batchnorm bn4_1(14*14*256);
  Relu relu4_1(14*14*256);

  Convolution conv4_2(14, 14, 256, 3, 3, 256, 1, true);
  conv4_2.xnor = false;
  Relu relu4_2(14*14*256);
  Batchnorm bn4_2(14*14*256);


  Convolution conv5_1(14, 14, 256, 3, 3, 512, 2, true);
  conv5_1.xnor = false;
  Batchnorm bn5_1(7*7*512);
  Relu relu5_1(7*7*512);

  Convolution conv5_2(7, 7, 512, 3, 3, 512, 1, true);
  conv5_2.xnor = false;
  Batchnorm bn5_2(7*7*512);
  Relu relu5_2(7*7*512);

  //Pooling pool5(14, 14, 256, 2, 2, 256, 2, false);

  Connected conn1(7*7*512, 3);
  SoftmaxWithCrossEntropy softmax(3);

  Network network;
  network.Add(&conv1);
  network.Add(&bn1);
  network.Add(&relu1);
  network.Add(&pool1);

  network.Add(&conv2_1);
  network.Add(&bn2_1);
  network.Add(&relu2_1);

  network.Add(&conv2_2);
  network.Add(&bn2_2);
  network.Add(&relu2_2);

  network.Add(&conv3_1);
  network.Add(&bn3_1);
  network.Add(&relu3_1);

  network.Add(&conv3_2);
  network.Add(&bn3_2);
  network.Add(&relu3_2);

  network.Add(&conv4_1);
  network.Add(&bn4_1);
  network.Add(&relu4_1);

  network.Add(&conv4_2);
  network.Add(&bn4_2);
  network.Add(&relu4_2);

  network.Add(&conv5_1);
  network.Add(&bn5_1);
  network.Add(&relu5_1);

  network.Add(&conv5_2);
  network.Add(&bn5_2);
  network.Add(&relu5_2);

  network.Add(&conn1);
  network.Add(&softmax);

  network.Init(batch, 0.001);
  float *batch_xs, *batch_ys;
  batch_xs = new float[batch*224*224*3];
  batch_ys = new float[batch*3];
  for(int iter = 0; iter < max_iter; iter++) {
    ms_t start = getms();
    int step = (iter*batch).train_num;
    get_mini_batch.train_num, batch,.train_data,.train_label, batch_xs, batch_ys);
    
    float *output = network.Inference(batch_xs);
    network.train(batch_ys);

    float loss = cross_entropy(batch, 3, output, batch_ys);
    float acc = accuracy(batch, 3, output, batch_ys); 
    if(iter%1 == 0) {
      cout << "iter = " << iter << ", time = " << (getms() - start) << "ms, loss = "
       << loss << " (accuracy = " << acc << ")" << endl;
    }
  }

  network.Save("tmp.net");

  return 0;
}
