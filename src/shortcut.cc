#include "layers.h"

Shortcut::Shortcut(int _w, int _h, int _c, 
	int conv_idx, Convolution *_conv, int actv_idx, Activation *_activation) {
  w = _w;
  h = _h;
  c = _c;
  conv = _conv;
  activation = _activation;

  this->conv_idx = conv_idx;
  this->actv_idx = actv_idx;
}

Shortcut::~Shortcut() {

}

void Shortcut::print() { printf("Shortcut\n"); }

void Shortcut::init() {

#ifdef GPU
  output = malloc_gpu(batch*h*w*c);
  m_delta = malloc_gpu(batch*h*w*c);
  identity = activation->output;

#else
  output = new float[batch*h*w*c];
  m_delta = new float[batch*h*w*c];
  identity = activation->output;
#endif
}



void Shortcut::forward() {

  for(int b = 0; b < batch; b++) {
    for(int i = 0; i < h; i++) {
      for(int j = 0; j < w; j++) {
        for(int k = 0; k < c; k++) {
          int out_idx = b*h*w*c + i*w*c + j*c + k;
          output[out_idx] = input[out_idx] + identity[out_idx];
        }
      }
    }
  }

}

void Shortcut::backward(float *delta) {

  for(int b = 0; b < batch; b++) {
    for(int i = 0; i < h; i++) {
      for(int j = 0; j < w; j++) {
        for(int k = 0; k < c; k++) {
          int out_idx = b*h*w*c + i*w*c + j*c + k;
          m_delta[out_idx] = delta[out_idx];
          activation->cut[out_idx] = delta[out_idx];
        }
      }
    }
  }

}

void Shortcut::update(update_args a) {
}

void Shortcut::save(fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Shortcut,%d,%d,%d,%d,%d", w, h, c, conv_idx, actv_idx);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
}


Shortcut* Shortcut::load(char* buf) {

  int para[5] = {0};
  char *token;
  int idx = 0;
  while (buf) {
    token = strtok(NULL, ",");
    para[idx] = atoi(token);
    idx++;
    if(idx > 4)
      break;
  }

  Shortcut *shortcut = new Shortcut(para[0], para[1], para[2], para[3], NULL, para[4], NULL);
  return shortcut;


}


