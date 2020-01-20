#include <math.h>

typedef struct UPDATE_ARGS {

  float beta1 = 0.9;
  float beta2 = 0.999;
  float epsilon = 1.0e-7;
  float lr = 1.0e-3;
  int iter = 0;
  float m_lr;

} update_args;

void adam_cpu(int n, float *x, float *grad_x, float *m_x, float *v_x, update_args a);

#ifdef GPU
void adam_gpu(int n, float *x, float *grad_x, float *m_x, float *v_x, update_args a);
#endif

