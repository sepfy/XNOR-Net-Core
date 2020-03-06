#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h> 
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include "network.h"

using namespace cv;
using namespace std;

#define WIDTH 112
#define HEIGHT 112
#define IM_SIZE WIDTH*HEIGHT*3
#define NUM_OF_CLASS 3

int load_images(char *dir, vector<Mat> *images) {
    
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
    resize(image, image, Size(HEIGHT, WIDTH), 0, 0, INTER_LINEAR);
    cvtColor(image, image, CV_BGR2RGB);
    image.convertTo(image, CV_32FC3);
    image = (image - 127.5)/127.5;
    images->push_back(image);
    count++;
  }

  closedir(fd);
  return count;
}


void get_random_images(int n, float *inputs, float *outputs) {

  float *in_tmp = new float[n*IM_SIZE];
  float *out_tmp = new float[n*NUM_OF_CLASS];

  memcpy(in_tmp, inputs, n*IM_SIZE*sizeof(float));
  memcpy(out_tmp, outputs, n*NUM_OF_CLASS*sizeof(float));

  vector<int> range;
  for(int i = 0; i < n; i++) {
    range.push_back(i);
  }

  shuffle(range.begin(), range.end(), default_random_engine());

  for(int i = 0; i < n; i++) {
    int p = range[i];
    memcpy(inputs + IM_SIZE*i, in_tmp + IM_SIZE*p, IM_SIZE*sizeof(float));
    memcpy(outputs + NUM_OF_CLASS*i, out_tmp + NUM_OF_CLASS*p, NUM_OF_CLASS*sizeof(float));
  }
  delete []in_tmp;
  delete []out_tmp;
}




void load_classes(const char *dir, vector<string> *classes) {

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
    classes->push_back(dp->d_name);
    count++;
  }

  closedir(fd);
}


int read_data(const char *basedir, float *&inputs, float *&outputs) {

  FILE *fp;
  char *line = NULL;
  size_t len = 0;

  vector<string> classes;
  load_classes(basedir, &classes);
  sort(classes.begin(), classes.end());
  vector<Mat> images;
  vector<int> counts;

  for(size_t i = 0; i < classes.size(); i++) {
    cout << "Class " << i << ": " << classes[i] << endl;
    char filedir[256] = {0};
    sprintf(filedir, "%s/%s", basedir, classes[i].c_str());
    int count = load_images(filedir, &images);
    counts.push_back(count);
  }
  
  inputs = new float[images.size()*IM_SIZE];
  outputs = new float[images.size()*NUM_OF_CLASS];
  memset(outputs, 0, images.size()*NUM_OF_CLASS*sizeof(float));

  for(int i = 0; i < images.size(); i++) {
    memcpy(inputs+IM_SIZE*i, images[i].data, IM_SIZE*sizeof(float));
  }

  int shift = 0;
  for(int i = 0; i < counts.size(); i++) {
    for(int j = 0; j < counts[i]; j++) {
      outputs[shift+NUM_OF_CLASS*j+i] = 1.0;
    }
    shift += counts[i]*NUM_OF_CLASS;
  }



  get_random_images(images.size(), inputs, outputs);
  return images.size();
}

int get_label(int n, float *x) {

  for(int i = 0; i < n; i++)
    if(x[i] == 1.0)
      return i;
}

void show_image(int num, float *inputs, float *outputs) {

  Mat image(HEIGHT, WIDTH, CV_32FC3);
  memcpy(image.data, inputs + IM_SIZE*num, IM_SIZE*sizeof(float));
  image = (image*127.5) + 127.5;
  image.convertTo(image, CV_8UC3);
  cvtColor(image, image, CV_RGB2BGR);

  char text[32] = {0};
  sprintf(text, "Label = %d", get_label(NUM_OF_CLASS, outputs + NUM_OF_CLASS*num)); 

  cv::putText(image, text,
            cv::Point(10, 30), 
            cv::FONT_HERSHEY_DUPLEX,
            1.0, CV_RGB(118, 185, 0), 2);
	    
  imshow("Display window", image );                
  waitKey(0); // Wait for a keystroke in the window

}



void get_mini_batch(int n, int b, float *data, float *label, float *batch_xs, float *batch_ys) {

  vector<int> range;
  for(int i = 0; i < n; i++) {
    range.push_back(i);
  }
 
  shuffle(range.begin(), range.end(), default_random_engine(0));
  for(int i = 0; i < n; i++) {
    int p = range[i];
    memcpy(batch_xs + IM_SIZE*i, data + IM_SIZE*p, IM_SIZE*sizeof(float));
    memcpy(batch_ys + NUM_OF_CLASS*i, label + NUM_OF_CLASS*p, NUM_OF_CLASS*sizeof(float));
  }

}

