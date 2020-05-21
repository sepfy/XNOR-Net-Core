#include "mnist.h"

#define LEARNING_RATE 1.0e-3
#define BATCH 100
#define MAX_ITER 15000

void MnistXnorNet(Network *network) {

  Convolution *conv1 = new Convolution(28, 28, 1, 5, 5, 20, 1, 0);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(24*24, 20);
  Activation *relu1 = new Activation(24*24*20, RELU);
  Maxpool *pool1 = new Maxpool(24, 24, 20, 2, 2, 20, 2, false); 


  Batchnorm *bn2 = new Batchnorm(12*12, 20);
  BinaryConv *bin_conv2 = new BinaryConv(12, 12, 20, 5, 5, 50, 1, 0);
  Activation *relu2 = new Activation(8*8*50, RELU);
  Maxpool *pool2 = new Maxpool(8, 8, 50, 2, 2, 50, 2, false);

  Batchnorm *bn3 = new Batchnorm(4*4, 50);
  BinaryConv *bin_conv3 = new BinaryConv(4, 4, 50, 4, 4, 500, 1, 0);
  Activation *relu3 = new Activation(500, RELU);
  Dropout *dropout1 = new Dropout(500, 0.5);
  
  Connected *conn1 = new Connected(500, 10);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);

  network->Add(conv1);
  network->Add(bn1);
  network->Add(relu1);
  network->Add(pool1);
  network->Add(bn2);
  network->Add(bin_conv2);
  network->Add(relu2);
  network->Add(pool2);
  network->Add(bn3);
  network->Add(bin_conv3);
  network->Add(relu3);
  network->Add(dropout1);
  network->Add(conn1);
  network->Add(softmax);

}

void MnistNet(Network *network) {


  Convolution *conv1 = new Convolution(28, 28, 1, 5, 5, 20, 1, 0);
  conv1->xnor = false;
  Batchnorm *bn1 = new Batchnorm(24*24, 20);
  Activation *relu1 = new Activation(24*24*20, RELU);
  Maxpool *pool1 = new Maxpool(24, 24, 20, 2, 2, 20, 2, false); 

  Convolution *conv2 = new Convolution(12, 12, 20, 5, 5, 50, 1, 0);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(8*8, 50);
  Activation *relu2 = new Activation(8*8*50, RELU);
  Maxpool *pool2 = new Maxpool(8, 8, 50, 2, 2, 50, 2, false);

  Convolution *conv3 = new Convolution(4, 4, 50, 4, 4, 500, 1, 0);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(1, 500);
  Activation *relu3 = new Activation(500, RELU);
 
  Connected *conn = new Connected(500, 10);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);
  
  network->Add(conv1);
  network->Add(bn1);
  network->Add(relu1);
  network->Add(pool1);

  network->Add(conv2);
  network->Add(bn2);
  network->Add(relu2);
  network->Add(pool2);

  network->Add(conv3);
  network->Add(bn3);
  network->Add(relu3);

  network->Add(conn);
  network->Add(softmax);

}



void help() {
  cout << "Usage: ./mnist .train/Deploy> <model name> <mnist dataset>" << endl;
  exit(1);
}


int main( int argc, char** argv ) {


  if(argc < 4) {
    help();
  }

  Network network;

  if(strcmp(argv[1], "train") == 0) {

    MnistXnorNet(&network);
    //MnistNet(&network);
    network.Init(BATCH, LEARNING_RATE, true);


#ifdef GPU
    float *train_data_tmp = read_train_data(argv[3]);
    float *train_label_tmp = read_train_label(argv[3]);
    float *train_data = malloc_gpu(60000*IM_SIZE);
    float *train_label = malloc_gpu(60000*10);
    gpu_push_array(train_data, train_data_tmp, 60000*IM_SIZE);
    gpu_push_array(train_label, train_label_tmp, 60000*10);
    
    delete []train_data_tmp;
    delete []train_label_tmp;
#else 
    float *train_data = read_train_data(argv[3]);
    float *train_label = read_train_label(argv[3]);
#endif

    ms_t start = getms();
    for(int iter = 0; iter < MAX_ITER; iter++) {

      int step = (iter*BATCH)%60000;
      float *batch_xs = train_data + step*IM_SIZE;
      float *batch_ys = train_label + step*CLASSES;
      network.Inference(batch_xs);
      float *output = network.output();

      network.Train(batch_ys);
      if(iter%10 == 0) {
        float loss = cross_entropy(BATCH, 10, output, batch_ys);
        cout << "iter = " << iter << ", time = " << (getms() - start) << "ms, loss = "
         << loss << endl;
        start = getms();
      }

 
    }

    network.Save(argv[2]);
#ifdef GPU
    cudaFree(train_data);
    cudaFree(train_label);
#else
    delete []train_data;
    delete []train_label;
#endif
  }
  else if(strcmp(argv[1], "deploy") == 0) {
    network.Load(argv[2], BATCH);
  }
  else {
    help();
  }


#ifdef GPU

  float *test_data_tmp, *test_label_tmp;
  float *test_data = malloc_gpu(10000*IM_SIZE);
  float *test_label = malloc_gpu(10000*10);
  test_data_tmp = read_validate_data(argv[3]);
  test_label_tmp = read_validate_label(argv[3]);
  gpu_push_array(test_data, test_data_tmp, 10000*IM_SIZE);
  gpu_push_array(test_label, test_label_tmp, 10000*10);

  delete []test_data_tmp;
  delete []test_label_tmp;
#else
  float *test_data, *test_label;
  test_data = read_validate_data(argv[3]);
  test_label = read_validate_label(argv[3]);
#endif


  float total = 0.0;
  int batch_num = 10000/BATCH;

  network.Deploy();

  ms_t start = getms();
  for(int iter = 0; iter < batch_num; iter++) {

    int step = (iter*BATCH);
    float *batch_xs = test_data + step*IM_SIZE;
    float *batch_ys = test_label + step*CLASSES;
    network.Inference(batch_xs);
    float *output = network.output();
    total += accuracy(BATCH, CLASSES, output, batch_ys);
  }
  cout << "Validate set error = " << (total/batch_num)*100 
       << ", time = " << getms() -start  << endl;

#ifdef GPU
  cudaFree(test_data);
  cudaFree(test_label);
#else
  delete []test_data;
  delete []test_label;  
#endif

  return 0;
}


