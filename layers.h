#include <iostream>

class LayerAbs {

  public:
    virtual Tensor forward(Tensor x) = 0;
    virtual Tensor backward(Tensor dx) = 0;
};

class Affine : public LayerAbs {

  public:
    Tensor W;
    Tensor b;

    Tensor dW;
    Tensor db;

    Tensor *x;

    Affine(Tensor _W, Tensor _b): W(_W), b(_b), dW(_W), db(_b) {}

    Tensor forward(Tensor _x) {
      x = &_x;
      return *x*W + b;
    }

    Tensor backward(Tensor dout) {
      Tensor dx = W;
      return dx;
    }


};

class Softmax: public LayerAbs {
  
  public:
    Softmax() {}
    Tensor forward(Tensor t) {
      Tensor t1(t.shape[0], t.shape[1]);
      for(int i = 0; i < t.shape[0]; i++) {
        float tmp = 0;
        float max = 0;
        for(int j = 0; j < t.shape[1]; j++) 
          if(t.value[i][j] > max)
            max = t.value[i][j];
        
        for(int j = 0; j < t.shape[1]; j++) {
          t1.value[i][j] = exp(t.value[i][j] - max);
          tmp += t1.value[i][j];
        }
        for(int j = 0; j < t.shape[1]; j++) 
          t1.value[i][j] /= tmp;
      }
      return t1;
    }

    Tensor backward(Tensor dout) {

      return dout;
    }
  
};
