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
    Tensor x;

    Affine(Tensor _W, Tensor _b): W(_W), b(_b), dW(_W), db(_b) {}

    Tensor forward(Tensor _x) {
      x = _x;
      return x*W + b;
    }

    Tensor backward(Tensor dout) {

      for(int i = 0; i < db.shape[1]; i++) {
        db.value[0][i] = 0;
        for(int j = 0; j < dout.shape[0]; j++)
          db.value[0][i] += dout.value[j][i];
      }
      dW = x.T()*dout;

      dout = dout*W.T();
      return dout;
    }


};




//6000x10 *10*10 -> 6000*10

class Sigmoid: public LayerAbs {

  public:
    Tensor Y;
 
    Tensor forward(Tensor t) {
      Y.init(t.shape[0], t.shape[1]);
      for(int i = 0; i < t.shape[0]; i++) {
        for(int j = 0; j < t.shape[1]; j++) {
          Y.value[i][j] = 1.0/(1.0 + exp(-1.0*(t.value[i][j])));
        }
      }
      return Y;
    } 
//6000x10
// Y: 6000x10
// dout: 
    Tensor backward(Tensor dout) {
      Tensor _dout(dout.shape[0], dout.shape[1], 0.0);  
      
      for(int i = 0; i < dout.shape[0]; i++) 
        for(int j = 0; j < dout.shape[1]; j++) 
          _dout.value[i][j] = dout.value[i][j]*(1.0 - Y.value[i][j])*Y.value[i][j];            
      return _dout;
    }
};

class Softmax: public LayerAbs {
  
  public:
    Tensor Y;
    Softmax(Tensor _Y): Y(_Y) {}
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

//6000x10
    Tensor backward(Tensor dout) {
      float batch_size = (float)this->Y.shape[0];
      return (dout - this->Y)*(1.0/batch_size);
    }
  
};
