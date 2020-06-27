#include "network.h"

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

  layers_.front()->input = input;
  int i = 0;
  for(auto layer = layers_.begin(); layer != layers_.end(); ++layer) {
    ms_t start = getms();
    (*layer)->Forward();
    //cout << "layer: " << i << ", time = " << (getms()-start) << endl;
    i++;
  }
  output_ = layers_.back()->output;
}

void Network::Train(float *correct) {

  float *delta = correct;
  update_args_.iter++;

  for(auto layer = layers_.rbegin(); layer != layers_.rend(); ++layer) {
    //ms_t start = getms();
    (*layer)->Backward(delta);
    (*layer)->Update(update_args_);
    delta = (*layer)->delta_;
    //cout << "layer: " << i << ", time = " << (getms()-start) << endl;
  }

} 

void Network::Deploy() {
  for(auto layer = layers_.begin(); layer != layers_.end(); ++layer)
    (*layer)->train_flag_ = false;
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
    char* token = strtok(buf, ",");
    if(!strcmp(token, "Convolution")) {
      Convolution *conv = Convolution::load(token);
      conv->LoadParams(&rfile, batch);
      this->Add(conv); 
    }
    if(!strcmp(token, "BinaryConv")) {
      BinaryConv *bin_conv = BinaryConv::load(token);
      bin_conv->LoadParams(&rfile, batch);
      this->Add(bin_conv); 
    }

    else if(!strcmp(token, "Connected")) {
      Connected *conn = Connected::load(token);
      conn->LoadParams(&rfile, batch);
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
      bn->LoadParams(&rfile, batch);
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

