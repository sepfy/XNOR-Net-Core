#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include "read.h"
#include <string.h>
#include <iostream>

using namespace std;
float N = 1000.0;
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


Tensor softmax(Tensor t) {

	Tensor t1(t.shape[0], t.shape[1]);    
	for(int i = 0; i < t.shape[0]; i++) {
		float tmp = 0;
		for(int j = 0; j < t.shape[1]; j++) {
			t1.value[i][j] = exp(t.value[i][j]);
			tmp += exp(t.value[i][j]);
		}
		for(int j = 0; j < t.shape[1]; j++) {
			t1.value[i][j] /= tmp;
		}
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


float cross_entropy(Tensor t1, Tensor t2) {

  
	float tmp = 0;
	for(int i = 0; i < t1.shape[0]; i++) {
		for(int j = 0; j < t1.shape[1]; j++) {
      tmp -= t1.value[i][j]*log(t2.value[i][j])
           + (1 - t1.value[i][j])*log(1 - t2.value[i][j]);
		}
	}

	tmp = tmp/N;
	return tmp;
}


int main(void) {

	Tensor b1(1, 30);
	Tensor W1(784, 30);
  Tensor b2(1, 10);
  Tensor W2(30, 10);
	// X = 60000x784
	Tensor X = read_images();
	//show_image(X1, 2);
	// Y = 60000x10
	Tensor Y = read_labels();
	//show_label(Y1, 2);
  map<string, Tensor> Network;

  Network.insert(pair<string, Tensor>("W1", Tensor(784, 10)));
  Network.insert(pair<string, Tensor>("b1", Tensor(1, 10)));

	//Tensor Y2 = softmax(W2*(X*W1+b1)+b2);
  int max_epoch = 1000;
  for(int iter = 0; iter < max_epoch; iter++) {
    Tensor Y1 = X*W1+b1;
    Tensor Y2 = softmax(Y1*W2+b2);
		printf("%f\n", cross_entropy(Y, Y2));
    
    for(int k = 0; k < X.shape[0]; k++) {

      Tensor D(1, 10);
      // W2 30*10
			for(int i = 0; i < W2.shape[0]; i++) {
				for(int j = 0; j < W2.shape[1]; j++) {
          D.value[0][j] = (1.0/N)*(Y2.value[k][j] - Y.value[k][j]);
					W2.value[i][j] -= 1.0*D.value[0][j]*Y1.value[k][i]; 
        }
      }

      // 784*30
			for(int i = 0; i < W1.shape[0]; i++) {
				for(int j = 0; j < W1.shape[1]; j++) {
          for(int l = 0; l < Y2.shape[1]; l++) {
					  W1.value[i][j] -= 1.0*D.value[0][l]*W2.value[j][l]*X.value[k][i];
          } 
        }
      }


    }
  }
	//cost function = 0.5||Y1 - Y||
	// 1/N 

  /*
  int max_epoch = 1000;

	for(int l =0; l < max_epoch; l++) {
		Tensor Y1 = softmax(X*W1 + b1);
		printf("%f\n", mean_square(Y1 - Y));

		for(int k = 0; k < X.shape[0]; k++) {
			for(int i = 0; i < W1.shape[0]; i++) {
				for(int j = 0; j < W1.shape[1]; j++) {
					W1.value[i][j] -= 1.0*(1.0/N)*(Y1.value[k][j] - Y.value[k][j])
						*Y1.value[k][j]*(1.0 - Y1.value[k][j])*X.value[k][i];
				}
			}
		}
	}


		for(int k = 0; k < X.shape[0]; k++) {
			for(int i = 0; i < W1.shape[0]; i++) {
				for(int j = 0; j < W1.shape[1]; j++) {
					D.value[i][j] = -1.0*(1.0/N)*(Y1.value[k][j] - Y.value[k][j])
						              *X.value[k][i];
				}
			}
		}

		for(int k = 0; k < X.shape[0]; k++) {
			for(int i = 0; i < W1.shape[0]; i++) {
				for(int j = 0; j < W1.shape[1]; j++) {
					W1.value[i][j] -= D.value[k][l]*W1.value
						              
				}
			}
		}

  */

	return 0;

}
