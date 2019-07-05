#include <vector>
#include <map>
#include "layers.h"
#include <fstream>
#include <string>

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


   float* inference(float *input) {
//     ms_t start = getms();   
     layers[0]->input = input;
     for(int i = 0; i < layers.size(); i++) {
       
       layers[i]->forward();
     }
//     cout << "forward time = " << (getms()-start) << endl;
     return layers[layers.size() - 1]->output;
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

  void save() {

     for(int i = layers.size() - 1; i >= 0; i--) {
       FILE *fp;
       fp = fopen("test.net", "wb");
       layers[i]->save(fp);
       fclose(fp);
     }
  }

  void load() {
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen("test.net", "r");

    read = getline(&line, &len, fp);
    line[read-1] = '\0';
    char* token = strtok(line, ",");
    if(!strcmp(token, "Convolution")) {
      int para[8] = {0}; 
      int idx = 0;
      while (token) {
        token = strtok(NULL, ",");
        para[idx] = atoi(token);
        idx++;
        if(idx > 7)
          break;
      }
      Convolution *conv = new Convolution(para[0], para[1], 
        para[2], para[3], para[4], para[5], para[6], para[7]);
      this->add(conv); 
    }



  }

};
