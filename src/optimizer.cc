#include "optimizer.h"

void adam_cpu(int n, float *x, float *grad_x, float *m_x, float *v_x, UpdateArgs a) {

  float m_lr = a.lr * pow(1.0 - pow(a.beta2, a.iter), 0.5) / (1.0 - pow(a.beta1, a.iter));
  for(int i = 0; i < n; i++) {
    m_x[i] = (1 - a.beta1)*grad_x[i] + a.beta1*m_x[i];
    v_x[i] = (1 - a.beta2)*pow(grad_x[i], 2.0) + a.beta2*v_x[i];
    x[i] -= m_lr * m_x[i]/(pow(v_x[i], 0.5) + a.epsilon);
  }

}

