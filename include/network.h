#include <vector>
#include <map>
#include "layers.h"

using namespace std; 

class Network {

  public:
   //map<string, Layer*> layers;
    vector<Layer*> layers;
    float lr;
    
    Network() {
    }

    void add(Layer* layer) {
      layers.push_back(layer); 
    }

    void initial(int batch, float _lr) {
      lr = _lr;
      for(int i = 0; i < layers.size(); i++) {
          layers[i]->batch = batch;
          layers[i]->init();
      }

      for(int i = 1; i < layers.size(); i++) {
          layers[i]->input = layers[i-1]->output;
      }

    }


   void inference(float *input) {
//     ms_t start = getms();   
     layers[0]->input = input;
     for(int i = 0; i < layers.size(); i++) {
       
       layers[i]->forward();
     }
//     cout << "forward time = " << (getms()-start) << endl;
   }

   void train(float *Y) {

 //    ms_t start = getms();   
     float *delta = Y;

//       ms_t start = getms();
//       printf("inference layer = %d, time = %lld ms\n", 0, getms() - start);
     for(int i = layers.size() - 1; i >= 0; i--) {
       layers[i]->backward(delta);
       delta = layers[i]->m_delta;
     }
//     cout << "backward time = " << (getms()-start) << endl;

//     start = getms();
     for(int i = layers.size() - 1; i >= 0; i--) {
       layers[i]->update(lr);
     }
 //    cout << "update time = " << (getms()-start) << endl;

   } 


};