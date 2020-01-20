#include <vector>
#include <map>
#include "layers.h"
#include <fstream>
#include <string>
using namespace std; 

class Network {
  public:
    vector<Layer*> layers;
    float lr;
    update_args a;
    Network();
    void add(Layer* layer);
    void initial(int batch, float _lr);
    float* inference(float *input);
    void train(float *Y);
    void save(char *filename);
    void load(char *filename, int batch);
    void deploy();
};
