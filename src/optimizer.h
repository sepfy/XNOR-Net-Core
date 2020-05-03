#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <math.h>

typedef struct UPDATE_ARGS {

  float beta1 = 0.9;
  float beta2 = 0.999;
  float epsilon = 1.0e-7;
  float lr = 1.0e-3;
  int iter = 0;
  float m_lr;
  float momentum = 0.9;
  bool adam = false;
  float decay = 0.0;

} update_args;

void adam_cpu(int n, float *x, float *grad_x, float *m_x, float *v_x, update_args a);

#ifdef GPU
void adam_gpu(int n, float *x, float *grad_x, float *m_x, float *v_x, update_args a);
void momentum_gpu(int n, float *x, float *grad_x, float *v_x, update_args a);
#endif

#endif
