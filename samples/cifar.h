#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <byteswap.h>
#include <iostream>
using namespace std;

#define IM_SIZE 32*32*3
#define NUM_OF_CLASS 10


void read_images(char *filename, float *inputs, float*outputs) {

  FILE *fp;

  fp = fopen(filename, "rb");
  if(!fp) {
    cout << "Cannot open file " << filename << endl;
    exit(1);
  }

  uint8_t label;
  uint8_t buf;
  float pixel;
  for(int b = 0; b < 10000; b++) {

    fread(&label, sizeof(uint8_t), 1, fp);
    outputs[b*NUM_OF_CLASS+label] = 1.0;
    for(int i = 0; i < 1024; i++) {
      fread(&buf, sizeof(uint8_t), 1, fp);
      pixel = ((float)buf - 127.5)/127.5;
      inputs[b*IM_SIZE+3*i] = pixel; 
    }

    for(int i = 0; i < 1024; i++) {
      fread(&buf, sizeof(uint8_t), 1, fp);
      pixel = ((float)buf - 127.5)/127.5;
      inputs[b*IM_SIZE+3*i+1] = pixel;
    }

    for(int i = 0; i < 1024; i++) {
      fread(&buf, sizeof(uint8_t), 1, fp);
      pixel = ((float)buf - 127.5)/127.5;
      inputs[b*IM_SIZE+3*i+2] = pixel;
    }
  }

  fclose(fp);
}


void read_train_data(char *path, float *inputs, float *outputs) {
  memset(inputs, 0, 50000*IM_SIZE);
  memset(outputs, 0, 50000*NUM_OF_CLASS);
  for(int i = 0; i < 5; i++) {
    char filename[64] = {0};
    sprintf(filename, "%s/data_batch_%d.bin", path, (i+1));
    read_images(filename, inputs+10000*IM_SIZE*i, outputs+10000*NUM_OF_CLASS*i);
  }

}

void read_test_data(char *path, float *inputs, float *outputs) {
  char filename[64] = {0};
  sprintf(filename, "%s/test_batch.bin", path);
  read_images(filename, inputs, outputs);
}

