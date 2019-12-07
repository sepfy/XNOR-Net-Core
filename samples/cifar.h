#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <byteswap.h>
#include <iostream>
using namespace std;

#define IM_SIZE 32*32*3
#define NUM_OF_CLASS 10



void read_images(const char *filename, float *inputs, float*outputs) {

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


void read_train_data(float *inputs, float *outputs) {
  memset(inputs, 0, 50000*IM_SIZE);
  memset(outputs, 0, 50000*NUM_OF_CLASS);
  for(int i = 0; i < 5; i++) {
    char filename[32] = {0};
    sprintf(filename, "data_batch_%d.bin", (i+1));
    read_images(filename, inputs+10000*IM_SIZE*i, outputs+10000*NUM_OF_CLASS*i);
  }

}

void read_test_data(float *inputs, float *outputs) {
  read_images("test_batch.bin", inputs, outputs);
}

/*
int main(void) {


  float *inputs, *outputs;
  inputs = new float[50000*IM_SIZE];
  outputs = new float[50000*NUM_OF_CLASS];

  read_train_data(inputs, outputs);
  read_test_data(inputs, outputs);



  return 0;
}
 
*/
