#include "layers.h"

Shortcut::Shortcut(int iw, int ih, int ic, int ow, int oh, int oc,
	int actv_idx, Activation *_activation) {

  this->iw = iw;
  this->ih = ih;
  this->ic = ic;
  this->ow = ow;
  this->oh = oh;
  this->oc = oc;

  activation = _activation;
  this->actv_idx = actv_idx;


}

Shortcut::~Shortcut() {

}

void Shortcut::print() { 

  printf("Shortcut\t %d x %d x %d \t %d x %d x %d\n", iw, ih, ic, ow, oh, oc); 
}

void Shortcut::init() {

#ifdef GPU
  output = malloc_gpu(batch*oh*ow*oc);
  m_delta = malloc_gpu(batch*oh*ow*oc);
  identity = activation->output;
 
#else
  output = new float[batch*oh*ow*oc];
  m_delta = new float[batch*oh*ow*oc];
  identity = activation->output;
#endif
}


void Shortcut::forward() {

  int sample = iw/ow;
  int stride = ic/oc;

  int minw = (ow < iw) ? ow : iw;
  int minh = (oh < ih) ? oh : ih;
  int maxc = (oc < ic) ? oc : ic;

  for(int b = 0; b < batch; b++) {
    for(int i = 0; i < minh; i++) {
      for(int j = 0; j < minw; j++) {
        for(int k = 0; k < maxc; k++) {
          int out_idx = b*oh*ow*oc + i*ow*oc + j*oc + k;
          int idt_idx = b*ih*iw*ic + sample*i*iw*ic + sample*j*ic + k*stride;
          output[out_idx] = input[out_idx] + identity[idt_idx];
        }
      }
    }
  }

#if 0
  for(int b = 0; b < batch; b++) {
    for(int i = 0; i < oh; i++) {
      for(int j = 0; j < ow; j++) {
        for(int k = 0; k < oc; k++) {
          int out_idx = b*oh*ow*oc + i*ow*oc + j*oc + k;
          output[out_idx] = input[out_idx] + add[out_idx];
        }
      }
    }
  }
#endif
}

void Shortcut::backward(float *delta) {

  int sample = iw/ow;
  int stride = ic/oc;

  int minw = (ow < iw) ? ow : iw;
  int minh = (oh < ih) ? oh : ih;
  int maxc = (oc < ic) ? oc : ic;

  for(int b = 0; b < batch; b++) {
    for(int i = 0; i < minh; i++) {
      for(int j = 0; j < minw; j++) {
        for(int k = 0; k < maxc; k++) {
          int out_idx = b*oh*ow*oc + i*ow*oc + j*oc + k;
          int idt_idx = b*ih*iw*ic + sample*i*iw*ic + sample*j*ic + k*stride;
          m_delta[out_idx] = delta[out_idx];
          activation->cut[idt_idx] = delta[out_idx];
        }
      }
    }
  }


/*
  for(int b = 0; b < batch; b++) {
    for(int i = 0; i < oh; i++) {
      for(int j = 0; j < ow; j++) {
        for(int k = 0; k < oc; k++) {
          int out_idx = b*oh*ow*oc + i*ow*oc + j*oc + k;
          m_delta[out_idx] = delta[out_idx];
          activation->cut[out_idx] = delta[out_idx];
        }
      }
    }
  }
*/
}

void Shortcut::update(update_args a) {
}

void Shortcut::save(fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Shortcut,%d,%d,%d,%d,%d,%d,%d", iw, ih, ic, 
    ow, oh, oc, actv_idx);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
}


Shortcut* Shortcut::load(char* buf) {

  int para[7] = {0};
  char *token;
  int idx = 0;
  while (buf) {
    token = strtok(NULL, ",");
    para[idx] = atoi(token);
    idx++;
    if(idx > 6)
      break;
  }

  Shortcut *shortcut = new Shortcut(para[0], para[1], para[2], para[3], para[4], para[5], para[6], NULL);
  return shortcut;

}


