#include "network.h"
using namespace std; 

Network::Network() {
}

void Network::add(Layer* layer) {
  layers.push_back(layer); 
}

void Network::initial(int batch, float _lr) {
  lr = _lr;
  for(int i = 0; i < layers.size(); i++) {
    layers[i]->batch = batch;
    layers[i]->init();
  }

  for(int i = 1; i < layers.size(); i++) {
    layers[i]->input = layers[i-1]->output;
  }

}


float* Network::inference(float *input) {
  layers[0]->input = input;
  for(int i = 0; i < layers.size(); i++) {

    //ms_t start = getms();   
    layers[i]->forward();
    //cout << "layer: " << i << ", time = " << (getms()-start) << endl;
  }
  return layers[layers.size() - 1]->output;
}

void Network::train(float *Y) {

  float *delta = Y;

  for(int i = layers.size() - 1; i >= 0; i--) {
    layers[i]->backward(delta);
    delta = layers[i]->m_delta;
  }

  for(int i = layers.size() - 1; i >= 0; i--) {
    layers[i]->update(lr);
  }

} 

void Network::deploy() {
  for(int i = 0; i < layers.size(); i++) {
    layers[i]->train_flag = false;
  }

}

void Network::save(char *filename) {

  fstream file;
  file.open(filename, ios::out);
  if(!file) {
    cout << "Open model file " << filename << " failed." << endl;
    exit(-1);
  }

  for(int i = 0; i < layers.size(); i++) {
    layers[i]->save(&file);
  }
  file.close();
}

void Network::load(char *filename, int batch) {

  char buf[64] = {0};
  fstream rfile;
  rfile.open(filename, ios::in);
  if(!rfile) {
    cout << "Open model file " << filename << " failed." << endl;
    exit(-1);
  }

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
      conv->runtime = true;
      if(conv->xnor) {
        for(int i = 0; i < conv->FC; i++) {
          rfile.read((char*)conv->bitset_weight[i].bits, 
                          conv->bitset_weight[i].N*sizeof(uint64_t));
        }
        rfile.read((char*)conv->mean, conv->FC*sizeof(float));
      }
      else {
        rfile.read((char*)conv->weight, conv->weight_size*sizeof(float));
      }

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
    else if(!strcmp(token, "Batchnorm")) {
      Batchnorm *bn = Batchnorm::load(token);
      bn->batch = batch;
      bn->train_flag = false;
      bn->init();
      rfile.read((char*)bn->running_mean, bn->N*sizeof(float));
      rfile.read((char*)bn->running_var, bn->N*sizeof(float));
      rfile.read((char*)bn->gamma, bn->N*sizeof(float));
      rfile.read((char*)bn->beta, bn->N*sizeof(float));
      this->add(bn);
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

  deploy();
}

