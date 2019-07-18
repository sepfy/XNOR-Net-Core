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
     layers[0]->input = input;
     for(int i = 0; i < layers.size(); i++) {
       
     ms_t start = getms();   
       layers[i]->forward();
     cout << "layer: " << i << ", time = " << (getms()-start) << endl;
     }
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

     fstream file;
     file.open("test.net", ios::out);
     for(int i = 0; i < layers.size(); i++) {
       layers[i]->save(&file);
     }
     file.close();
  }

  void load(int batch) {

    char buf[64] = {0};
    fstream rfile;
    rfile.open("test.net", ios::in);

    int ii = 0;
    int ret;
    while(true) {

      rfile.read(buf, sizeof(buf));
      if(rfile.eof())
        break;
      //cout << buf << endl;
      char* token = strtok(buf, ",");
      if(!strcmp(token, "Convolution")) {
        
        Convolution *conv = Convolution::load(token);
        conv->batch = batch;
        conv->init();
        rfile.read((char*)conv->weight, conv->weight_size*sizeof(float));
        rfile.read((char*)conv->bias, conv->bias_size*sizeof(float));
        this->add(conv); 
        //cout << conv->weight[0] << endl;
        //cout << conv->bias[0] << endl;
      }
      else if(!strcmp(token, "Connected")) {
        Connected *conn = Connected::load(token);
        conn->batch = batch;
        conn->init();
        rfile.read((char*)conn->weight, conn->N*conn->M*sizeof(float));
        rfile.read((char*)conn->bias, conn->M*sizeof(float));
        //cout << conn->N << endl;
        this->add(conn); 
       
      }
      else if(!strcmp(token, "Pooling")) {
        Pooling *pool = Pooling::load(token);
        pool->batch = batch;
        pool->init();
        this->add(pool);
        //cout << pool->FC << endl;
      }
      else if(!strcmp(token, "Relu")) {
        Relu *relu = Relu::load(token);
        relu->batch = batch;
        relu->init(); 
        this->add(relu);
        //cout << relu->N << endl;
      }
      else if(!strcmp(token, "Softmax")) {
        SoftmaxWithCrossEntropy *softmax = SoftmaxWithCrossEntropy::load(token);
        softmax->batch = batch;
        softmax->init();
        this->add(softmax);
        //cout << softmax->N << endl;
      }


    }
    for(int i = 1; i < layers.size(); i++) {
      this->layers[i]->input = layers[i-1]->output;
    }

  }

};
