#include "layers.h"

Shortcut::Shortcut(int _w, int _h, int _c, Convolution *_conv, Relu *_activation) {
  w = _w;
  h = _h;
  c = _c;
  conv = _conv;
  activation = _activation;
}

Shortcut::~Shortcut() {

}

void Shortcut::init() {
  output = new float[batch*h*w*c];
  m_delta = new float[batch*h*w*c];
  identity = activation->output;
  
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
}


Shortcut* Shortcut::load(char* buf) {

}


