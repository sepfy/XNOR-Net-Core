#ifndef LAYER_BATCHNORM_H_
#define LAYER_BATCHNORM_H_

#include "layer.h"

// Batch normalization layer
class Batchnorm : public Layer {
 public:
  Batchnorm(int spatial, int channel) : spatial_(spatial), channel_(channel) {}
  ~Batchnorm();
  void Init() override;
  void Print() override;
  void Forward() override;
  void Backward(float *delta) override;
  void Update(UpdateArgs a) override;
  void Save(std::fstream *file);
  static Batchnorm* load(char *buf);
  void LoadParams(std::fstream *file, int batch) override;

  int n_;
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

  int spatial_;
  int channel_;
  
 private:
  void GetMean();
  void GetVariance();
  void Normalize();
  void ScaleAndShift();

};

#endif //  LAYER_BATCHNORM_H_
