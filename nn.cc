#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include "read.h"
#include <string.h>
#include <iostream>
#include "layers.h"


using namespace std;
float N = 100.0;
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


	Tensor b1(1, 10);
	Tensor W1(784, 10);
	Tensor b2(1, 10);
	Tensor W2(10, 10);
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
  Softmax softmax1;
  //affine.forward(X);

  //return 0;

	for(int iter = 0; iter < max_epoch; iter++) {
		Tensor Y1 = sigmoid(affine1.forward(X));
		Tensor Y2 = softmax1.forward(Y1*W2 + b2);
    if(iter%10 == 0)
		  printf("error = %f\n", acc(Y, Y2));

		Tensor D(X.shape[0], 10);
		Tensor DD(X.shape[0], 30);
    // Output layer 10x10 
		for(int k = 0; k < X.shape[0]; k++) {
			for(int j = 0; j < W2.shape[1]; j++) {
		    D.value[k][j] = (1.0/N)*(Y2.value[k][j] - Y.value[k][j]);
        affine2.b.value[0][j] -= rate*D.value[k][j];
				for(int i = 0; i < W2.shape[0]; i++) {
				  affine2.W.value[i][j] -= rate*D.value[k][j]*Y1.value[k][i];
                         //+ (eta/N)*W2.value[i][j]);
				}
			}
    }

    // Hidden layer 784x10
		for(int k = 0; k < X.shape[0]; k++) {
		  for(int j = 0; j < W1.shape[1]; j++) {
        DD.value[k][j] = 0;
			  for(int l = 0; l < Y2.shape[1]; l++) {
          DD.value[k][j] += D.value[k][l]*W2.value[j][l]*Y1.value[k][j]*(1.0-Y1.value[k][j]);
			  } 
        affine1.b.value[0][j] -= rate*DD.value[k][j];
			  for(int i = 0; i < W1.shape[0]; i++) {
						affine1.W.value[i][j] -= rate*DD.value[k][j]*X.value[k][i];
                           //+ (eta/N)*affine.W.value[i][j]);
				}
			}
		}


	}

	return 0;

}
