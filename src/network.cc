#include "network.h"
using namespace std; 

Network::Network() {
}

Network::~Network() {

  delete []shared;
  if(quantized_shared)
    delete []quantized_shared;
}

void Network::add(Layer* layer) {
  layers.push_back(layer); 
}

void Network::initial(int batch, float _lr, bool use_adam) {
  a.lr = _lr;
  a.adam = use_adam;
  //a.decay = 0.01;

  size_t max = 0;
  for(int i = 0; i < layers.size(); i++) {
    layers[i]->batch = batch;
    layers[i]->init();

    layers[i]->print();

    //cout << layers[i]->shared_size << endl;
    if(layers[i]->shared_size > max)
      max = layers[i]->shared_size;
  }

  //cout << max/(1024*1024) << endl;

  // Create shared
#ifdef GPU 
  shared = malloc_gpu(max);
#else
  shared = new float[max];
#endif

  for(int i = 0; i < layers.size(); i++)
    layers[i]->shared = shared;

  for(int i = 1; i < layers.size(); i++) {
    layers[i]->input = layers[i-1]->output;
  }

}


float* Network::inference(float *input) {
  layers[0]->input = input;
  for(int i = 0; i < layers.size(); i++) {
    //ms_t start = getms();   
#ifdef GPU
    layers[i]->forward_gpu();
#else
    layers[i]->forward();
#endif
    //cout << "layer: " << i << ", time = " << (getms()-start) << endl;
  }
  return layers[layers.size() - 1]->output;
}

void Network::train(float *Y) {

  float *delta = Y;
 
  for(int i = layers.size() - 1; i >= 0; i--) {
    //ms_t start = getms();
#ifdef GPU
    layers[i]->backward_gpu(delta);
#else
    layers[i]->backward(delta);
#endif
    delta = layers[i]->m_delta;
    //cout << "layer: " << i << ", time = " << (getms()-start) << endl;
  }

  a.iter++;
  for(int i = layers.size() - 1; i >= 0; i--) {
    //ms_t start = getms();   
    layers[i]->update(a);
    //cout << "layer: " << i << ", time = " << (getms()-start) << endl;
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
      conv->train_flag = false;
      conv->init();
      conv->runtime = true;
      if(conv->xnor) {

#ifdef GEMMBITSERIAL
        size_t size = conv->ctx.rhs.nbits*conv->ctx.rhs.wordsPerBitplane()*sizeof(uint64_t);
	rfile.read((char*)conv->ctx.rhs.data, size);
#else
        for(int i = 0; i < conv->FC; i++) {
          rfile.read((char*)conv->bitset_weight[i].bits, 
                          conv->bitset_weight[i].N*sizeof(BIT_BLK));
        }
#endif
        rfile.read((char*)conv->mean, conv->FC*sizeof(float));
      }
      else {

#ifdef GPU
        float *weight_tmp = new float[conv->weight_size];
        rfile.read((char*)weight_tmp, conv->weight_size*sizeof(float));
	gpu_push_array(conv->weight, weight_tmp, conv->weight_size);
	delete []weight_tmp;
#else
        rfile.read((char*)conv->weight, conv->weight_size*sizeof(float));
#endif

      }

#ifdef GPU
      float *bias_tmp = new float[conv->bias_size];
      rfile.read((char*)bias_tmp, conv->bias_size*sizeof(float));
      gpu_push_array(conv->bias, bias_tmp, conv->bias_size);
      delete []bias_tmp;
#else
      rfile.read((char*)conv->bias, conv->bias_size*sizeof(float));
#endif

      this->add(conv); 
      //cout << conv->weight[0] << endl;
      //cout << conv->bias[0] << endl;
    }
    else if(!strcmp(token, "Connected")) {
      Connected *conn = Connected::load(token);
      conn->batch = batch;
      conn->init();

#ifdef GPU
      float *weight_tmp = new float[conn->N*conn->M];
      float *bias_tmp = new float[conn->M];
      rfile.read((char*)weight_tmp, conn->N*conn->M*sizeof(float));
      rfile.read((char*)bias_tmp, conn->M*sizeof(float));
      gpu_push_array(conn->weight, weight_tmp, conn->N*conn->M);
      gpu_push_array(conn->bias, bias_tmp, conn->M);
      delete []weight_tmp;
      delete []bias_tmp;
#else
      rfile.read((char*)conn->weight, conn->N*conn->M*sizeof(float));
      rfile.read((char*)conn->bias, conn->M*sizeof(float));
      //cout << conn->N << endl;
#endif
      this->add(conn); 

    }
    else if(!strcmp(token, "Pooling")) {
      Pooling *pool = Pooling::load(token);
      pool->batch = batch;
      pool->init();
      this->add(pool);
      //cout << pool->FC << endl;
    }
    else if(!strcmp(token, "AvgPool")) {
      AvgPool *avgpool = AvgPool::load(token);
      avgpool->batch = batch;
      avgpool->init();
      this->add(avgpool);
      //cout << pool->FC << endl;
    }
    else if(!strcmp(token, "Shortcut")) {
      Shortcut *shortcut = Shortcut::load(token);
      shortcut->batch = batch;
      size_t l = layers.size();
      //shortcut->conv = (Convolution*)(layers[l + shortcut->conv_idx]);
      shortcut->activation = (Activation*)(layers[l + shortcut->actv_idx]);
      shortcut->init();
      this->add(shortcut);
      //cout << pool->FC << endl;
    }
    else if(!strcmp(token, "Activation")) {
      Activation *actv = Activation::load(token);
      actv->batch = batch;
      actv->init(); 
      this->add(actv);
      //cout << relu->N << endl;
    }
    else if(!strcmp(token, "Batchnorm")) {
      Batchnorm *bn = Batchnorm::load(token);
      bn->batch = batch;
      bn->train_flag = false;
      bn->init();
#ifdef GPU
      float *mean_tmp = new float[bn->N];
      float *var_tmp = new float[bn->N];
      float *gamma_tmp = new float[bn->N];
      float *beta_tmp = new float[bn->N];
      rfile.read((char*)mean_tmp, bn->N*sizeof(float));
      rfile.read((char*)var_tmp, bn->N*sizeof(float));
      rfile.read((char*)gamma_tmp, bn->N*sizeof(float));
      rfile.read((char*)beta_tmp, bn->N*sizeof(float));
      gpu_push_array(bn->running_mean, mean_tmp, bn->N);
      gpu_push_array(bn->running_var, var_tmp, bn->N);
      gpu_push_array(bn->gamma, gamma_tmp, bn->N);
      gpu_push_array(bn->beta, beta_tmp, bn->N);
      delete []mean_tmp;
      delete []var_tmp;
      delete []gamma_tmp;
      delete []beta_tmp;
#else
      rfile.read((char*)bn->running_mean, bn->N*sizeof(float));
      rfile.read((char*)bn->running_var, bn->N*sizeof(float));
      rfile.read((char*)bn->gamma, bn->N*sizeof(float));
      rfile.read((char*)bn->beta, bn->N*sizeof(float));
      for(int i = 0; i < bn->N; i++)
          bn->std[i] = pow(bn->running_var[i] + bn->epsilon, 0.5);
      bn->runtime = true;
#endif
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

  size_t max = 0;
  for(int i = 0; i < layers.size(); i++) {
    layers[i]->print();
    if(layers[i]->shared_size > max)
      max = layers[i]->shared_size;
  }
#ifdef GPU 
  shared = malloc_gpu(max);
#else
  shared = new float[max];
  quantized_shared = new int8_t[max];
#endif

  for(int i = 0; i < layers.size(); i++) {
    layers[i]->shared = shared;
    layers[i]->quantized_shared = quantized_shared;
  }

  deploy();
}

