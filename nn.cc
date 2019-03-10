#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include "read.h"
#include <string.h>
#include <iostream>
#include "layers.h"


using namespace std;
float N = 600.0;
typedef struct _Operator {

	char *name;
} Operator;


class Network {

	map<string, Tensor> layers;

	public:
	void add(string name, Tensor t) { 
		layers.insert(pair<string, Tensor>(name, t));    
	}


	void forward() {
		//      map<string, Tensor>::iterator iter;
		//      for(iter = layer.begin(); iter != layer.end(); iter++)
		//        cout<<iter->first<<endl;
	}

};

typedef Tensor *funcPtr(Tensor t1, Tensor t2);

class Layer {

	Tensor op_mul(Tensor t1, Tensor t2) {
		return t1*t2;
	}

	Tensor op_add(Tensor t1, Tensor t2) {
		return t1+t2;
	}


};

Tensor sigmoid(Tensor t) {

	Tensor t1(t.shape[0], t.shape[1]);
	for(int i = 0; i < t.shape[0]; i++) {
		for(int j = 0; j < t.shape[1]; j++) {
			t1.value[i][j] = 1.0/(1.0 + exp(-1.0*(t.value[i][j])));
		}
	}
	return t1;
}


Tensor softmax(Tensor t) {

	Tensor t1(t.shape[0], t.shape[1]);    
	for(int i = 0; i < t.shape[0]; i++) {
		float tmp = 0;
		float max = 0;
		for(int j = 0; j < t.shape[1]; j++) {
			if(t.value[i][j] > max) 
				max = t.value[i][j];
		}
		//cout << "max is " << max << endl;
		for(int j = 0; j < t.shape[1]; j++) {
			t1.value[i][j] = exp(t.value[i][j] - max);
			tmp += t1.value[i][j];
			//cout << t1.value[i][j] << endl;
		}
		float chk = 0;
		for(int j = 0; j < t.shape[1]; j++) {
			t1.value[i][j] /= tmp;
			chk += t1.value[i][j];
			//cout << t1.value[i][j] << endl;
		}
		// cout << chk << endl;
		//        char c;
		//        scanf("%c", &c);
	}
	return t1;
}

float mean_square(Tensor t) {

	float tmp = 0;
	for(int i = 0; i < t.shape[0]; i++) {
		for(int j = 0; j < t.shape[1]; j++) {
			tmp += pow(t.value[i][j], 2.0);
		}
	}

	tmp = 0.5*tmp/N;
	return tmp;
}

Tensor argmax(Tensor t) {


  Tensor t1(t.shape[0], 1);

  int max_idx;
	for(int i = 0; i < t.shape[0]; i++) {
    float max = 0;
		for(int j = 0; j < t.shape[1]; j++) {
      if(t.value[i][j] > max) {
        max = t.value[i][j];
        max_idx = j;
      }
		}
    t1.value[i][0] = (float)max_idx;
	}

  return t1;
  
}


float cross_entropy(Tensor t1, Tensor t2) {


	float tmp = 0;
	for(int i = 0; i < t1.shape[0]; i++) {
		for(int j = 0; j < t1.shape[1]; j++) {
			tmp -= t1.value[i][j]*log(t2.value[i][j] + 1.0e-6)
				+ (1.0 - t1.value[i][j])*log(1.0 - t2.value[i][j] + 1.0e-6);
		}
	}
	tmp = tmp/N;
	return tmp;
}

float acc(Tensor t1, Tensor t2) {

  float tmp = 0;

  Tensor argt1 = argmax(t1);
  Tensor argt2 = argmax(t2);

  float err = 0;
  for(int i = 0; i < argt1.shape[0]; i++) {
    if(fabs(argt1.value[i][0] - argt2.value[i][0]) > 0.1)
      err += 1.0;
  }
  return err/(float)(argt1.shape[0]);
}

int main(void) {


	Tensor b1(1, 30);
	Tensor W1(784, 30);
	Tensor b2(1, 10);
	Tensor W2(30, 10);
//	Tensor b3(1, 10);
//	Tensor W3(30, 10);
	// X = 60000x784
	Tensor X = read_images();
	//show_image(X1, 2);
	// Y = 60000x10
	Tensor Y = read_labels();
	//show_label(Y, 0);
	//show_label(Y, 1);

	int max_epoch = 10000;
  float eta = 0.0;
	float rate = 1.0;

  Affine affine1(W1, b1);
  Affine affine2(W2, b2);
//  Affine affine3(W3, b3);
  Sigmoid sigmoid1;
//  Sigmoid sigmoid2;
  Softmax softmax1(Y);
    Tensor dout;

  Tensor Y1, Y2, Y3;

	for(int iter = 0; iter < max_epoch; iter++) {

		Tensor Y1 = sigmoid1.forward(affine1.forward(X));
		Tensor Y2 = softmax1.forward(affine2.forward(Y1));
//		Tensor Y3 = softmax1.forward(affine3.forward(Y2));

    if(iter%10 == 0)
		  printf("error = %f\n", acc(Y, Y2));

//    dout = softmax1.backward(Y3);
//    dout = affine3.backward(dout);

    dout = softmax1.backward(Y2);
    dout = affine2.backward(dout);

    dout = sigmoid1.backward(dout);
    dout = affine1.backward(dout);

//    affine3.b = affine3.b - affine3.db*rate;
//    affine3.W = affine3.W - affine3.dW*rate;
    affine2.b = affine2.b - affine2.db*rate;
    affine2.W = affine2.W - affine2.dW*rate;
    affine1.b = affine1.b - affine1.db*rate;
    affine1.W = affine1.W - affine1.dW*rate;

	}

	return 0;

}
