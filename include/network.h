#include <vector>
#include <map>
#include "layers.h"

using namespace std; 

class Network {

 public:
   //map<string, Layer*> layers;
   vector<Layer*> layers;

   void add(Layer* layer) {
     layers.push_back(layer); 
   }

   void inference() {
     for(int i = 0; i < layers.size(); i++)
       layers[i]->forward();
   }

   void train(float *Y) {

     float *delta = Y;

     for(int i = layers.size() - 1; i >= 0; i--) {
       layers[i]->backward(delta);
       delta = layers[i]->m_delta;
     }

     for(int i = layers.size() - 1; i >= 0; i--)
       layers[i]->update();
     

   } 


};
