#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h> 
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include "network.h"

using namespace cv;
using namespace std;

#define IM_SIZE 32*32*3
int NUM_OF_CLASS;

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
    image.convertTo(image, CV_32FC1);
    image = (image - 127.5)/127.5;
    images->push_back(image);
    count++;
  }

  closedir(fd);
  return count;
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
  NUM_OF_CLASS = classes.size();
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
      outputs[shift+counts.size()*j+i] = 1.0;
    }
    shift = counts[i]*NUM_OF_CLASS;
  }

  // recover image for test
  /*
  Mat image(32, 32, CV_32FC3);
  memcpy(image.data, inputs, IM_SIZE*sizeof(float));
  image = (image*127.5) + 127.5;
  image.convertTo(image, CV_8UC3);
  imshow("Display window", image );                
  // Show our image inside it.
  waitKey(0); // Wait for a keystroke in the window
  */ 
  //cout << images.size() << endl;

  return images.size();
}


void get_mini_batch(int n, int b, float *data, float *label, float *&batch_xs, float *&batch_ys) {

  srand(time(NULL));
  for(int i = 0; i < b; i++) {
    int p = rand()%n;
    memcpy(batch_xs + IM_SIZE*i, data + IM_SIZE*p, IM_SIZE*sizeof(float));
    memcpy(batch_ys + NUM_OF_CLASS*i, label + NUM_OF_CLASS*p, NUM_OF_CLASS*sizeof(float));
  }
}

