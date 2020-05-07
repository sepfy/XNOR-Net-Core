#include "network.h"
using namespace std; 

void Network::Add(Layer* layer) {
  layers_.push_back(layer); 
}

void Network::Init(int batch, float _lr, bool use_adam) {
  update_args_.lr = _lr;
  update_args_.adam = use_adam;
  //a.decay = 0.01;

  size_t max = 0;
  for(int i = 0; i < layers_.size(); i++) {
    layers_[i]->train_flag_ = true;
    layers_[i]->batch = batch;
    layers_[i]->Init();
    layers_[i]->Print();

    //cout << layers_[i]->shared_size_ << endl;
    if(layers_[i]->shared_size_ > max)
      max = layers_[i]->shared_size_;
  }

  //cout << max/(1024*1024) << endl;

  // Create shared_
#ifdef GPU 
  shared_ = malloc_gpu(max);
#else
  shared_ = new float[max];
#endif

  for(int i = 0; i < layers_.size(); i++)
    layers_[i]->shared_ = shared_;

  for(int i = 1; i < layers_.size(); i++) {
    layers_[i]->input = layers_[i-1]->output;
  }

}


void Network::Inference(float *input) {
  layers_[0]->input = input;
  for(int i = 0; i < layers_.size(); i++) {
    //ms_t start = getms();   
    layers_[i]->Forward();
    //cout << "layer: " << i << ", time = " << (getms()-start) << endl;
  }

  output_ = layers_[layers_.size() - 1]->output;
}

void Network::Train(float *correct) {

  float *delta = correct;
 
  for(int i = layers_.size() - 1; i >= 0; i--) {
    //ms_t start = getms();
    layers_[i]->Backward(delta);
    delta = layers_[i]->delta_;
    //cout << "layer: " << i << ", time = " << (getms()-start) << endl;
  }

  update_args_.iter++;
  for(int i = layers_.size() - 1; i >= 0; i--) {
    //ms_t start = getms();   
    layers_[i]->Update(update_args_);
    //cout << "layer: " << i << ", time = " << (getms()-start) << endl;
  }
} 

void Network::Deploy() {
  for(int i = 0; i < layers_.size(); i++) {
    layers_[i]->train_flag_ = false;
  }

}

void Network::Save(char *filename) {

  fstream file;
  file.open(filename, ios::out);
  if(!file) {
    cout << "Open model file " << filename << " failed." << endl;
    exit(-1);
  }

  for(int i = 0; i < layers_.size(); i++) {
    layers_[i]->Save(&file);
  }
  file.close();
}

void Network::Load(char *filename, int batch) {

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
      conv->train_flag_ = false;
      conv->Init();
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

      this->Add(conv); 
      //cout << conv->weight[0] << endl;
      //cout << conv->bias[0] << endl;
    }
    else if(!strcmp(token, "Connected")) {
      Connected *conn = Connected::load(token);
      conn->batch = batch;
      conn->Init();

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
      this->Add(conn); 

    }
    else if(!strcmp(token, "Pooling")) {
      Maxpool *pool = Maxpool::load(token);
      pool->batch = batch;
      pool->Init();
      this->Add(pool);
      //cout << pool->FC << endl;
    }
    else if(!strcmp(token, "Avgpool")) {
      Avgpool *avgpool = Avgpool::load(token);
      avgpool->batch = batch;
      avgpool->Init();
      this->Add(avgpool);
      //cout << pool->FC << endl;
    }
    else if(!strcmp(token, "Shortcut")) {
      Shortcut *shortcut = Shortcut::load(token);
      shortcut->batch = batch;
      size_t l = layers_.size();
      //shortcut->conv = (Convolution*)(layers_[l + shortcut->conv_idx]);
      shortcut->activation = (Activation*)(layers_[l + shortcut->actv_idx]);
      shortcut->Init();
      this->Add(shortcut);
      //cout << pool->FC << endl;
    }
    else if(!strcmp(token, "Activation")) {
      Activation *actv = Activation::load(token);
      actv->batch = batch;
      actv->Init(); 
      this->Add(actv);
      //cout << relu->N << endl;
    }
    else if(!strcmp(token, "Batchnorm")) {
      Batchnorm *bn = Batchnorm::load(token);
      bn->batch = batch;
      bn->train_flag_ = false;
      bn->Init();
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
      this->Add(bn);
      //cout << relu->N << endl;
    }
    else if(!strcmp(token, "Softmax")) {
      SoftmaxWithCrossEntropy *softmax = SoftmaxWithCrossEntropy::load(token);
      softmax->batch = batch;
      softmax->Init();
      this->Add(softmax);
      //cout << softmax->N << endl;
    }

  }

  for(int i = 1; i < layers_.size(); i++) {
    this->layers_[i]->input = layers_[i-1]->output;
  }

  size_t max = 0;
  for(int i = 0; i < layers_.size(); i++) {
    layers_[i]->Print();
    if(layers_[i]->shared_size_ > max)
      max = layers_[i]->shared_size_;
  }
#ifdef GPU 
  shared_ = malloc_gpu(max);
#else
  shared_ = new float[max];
  quantized_shared_ = new int8_t[max];
#endif

  for(int i = 0; i < layers_.size(); i++) {
    layers_[i]->shared_ = shared_;
    layers_[i]->quantized_shared_ = quantized_shared_;
  }
  rfile.close();
  Deploy();
}

