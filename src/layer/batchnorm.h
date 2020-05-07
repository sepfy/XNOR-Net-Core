#ifndef LAYER_BATCHNORM_H_
#define LAYER_BATCHNORM_H_

#include "layer.h"

class Batchnorm : public Layer {
  public:
    int N;
    float iter = 0.0;
    float *mean, *var, *std, *running_mean, *running_var;
    float *normal;
    float epsilon = 1.0e-7;
    float *gamma, *beta, *dgamma, *dbeta;
    float *m_gamma, *m_beta, *v_gamma, *v_beta;
    float *dxn;
    float *dxc;
    float *dvar;
    float *dstd;
    float *dmu;
    float *xc;
    bool runtime = false;
    float momentum = 0.9;
    float beta1 = 0.9;
    float beta2 = 0.999;
    Batchnorm(int N);
    ~Batchnorm();
    void Init();
    void Print();
    void Forward();
    void Backward(float *delta);
    void Update(UpdateArgs a) override;
    void Save(std::fstream *file);
    static Batchnorm* load(char *buf);


    void get_mean();
    void get_variance();
    void normalize();
    void scale_and_shift();

    void get_mean_gpu();
    void get_variance_gpu();
    void normalize_gpu();
    void scale_and_shift_gpu();

};

#endif //  LAYER_BATCHNORM_H_
