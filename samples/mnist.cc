#include "mnist.h"

#define LEARNING_RATE 1.0e-3
#define BATCH 100
#define MAX_ITER 25000

void MnistXnorNet(Network *network) {

  Convolution *conv1 = new Convolution(28, 28, 1, 5, 5, 32, 1, false);
  conv1->xnor = false;
  Relu *relu1 = new Relu(24*24*32);
  Pooling *pool1 = new Pooling(24, 24, 32, 2, 2, 32, 2, false); 

  Batchnorm *bn1 = new Batchnorm(12*12*32);
  Convolution *conv2 = new Convolution(12, 12, 32, 5, 5, 64, 1, false);
  Relu *relu2 = new Relu(8*8*64);
  Pooling *pool2 = new Pooling(8, 8, 64, 2, 2, 64, 2, false);

  Batchnorm *bn2 = new Batchnorm(4*4*64);
  Convolution *conv3 = new Convolution(4, 4, 64, 4, 4, 512, 1, false);
  Relu *relu3 = new Relu(512);

  Connected *conn1 = new Connected(512, 10);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);

  network->add(conv1);
  network->add(relu1);
  network->add(pool1);
  network->add(bn1);
  network->add(conv2);
  network->add(relu2);
  network->add(pool2);
  network->add(bn2);
  network->add(conv3);
  network->add(relu3);
  network->add(conn1);
  network->add(softmax);

}

void MnistNet(Network *network) {

  Convolution *conv1 = new Convolution(28, 28, 1, 5, 5, 32, 1, false);
  conv1->xnor = false;
  Relu *relu1 = new Relu(24*24*32);
  Pooling *pool1 = new Pooling(24, 24, 32, 2, 2, 32, 2, false); 

  Convolution *conv2 = new Convolution(12, 12, 32, 5, 5, 64, 1, false);
  conv2->xnor = false;
  Batchnorm *bn2 = new Batchnorm(8*8*64);
  Relu *relu2 = new Relu(8*8*64);
  Pooling *pool2 = new Pooling(8, 8, 64, 2, 2, 64, 2, false);

  Convolution *conv3 = new Convolution(4, 4, 64, 4, 4, 512, 1, false);
  conv3->xnor = false;
  Batchnorm *bn3 = new Batchnorm(512);
  Relu *relu3 = new Relu(512);
  Dropout *dropout1 = new Dropout(512, 0.5);

  Connected *conn1 = new Connected(512, 10);
  SoftmaxWithCrossEntropy *softmax = new SoftmaxWithCrossEntropy(10);

  network->add(conv1);
  network->add(relu1);
  network->add(pool1);
  network->add(conv2);
  network->add(bn2);
  network->add(relu2);
  network->add(pool2);
  network->add(conv3);
  network->add(bn3);
  network->add(relu3);
  network->add(dropout1);
  network->add(conn1);
  network->add(softmax);

}



void help() {
  cout << "Usage: ./mnist <train/deploy> <model name> <mnist dataset>" << endl;
  exit(1);
}


int main( int argc, char** argv ) {


  if(argc < 4) {
    help();
  }

  Network network;

  if(strcmp(argv[1], "train") == 0) {

    //MnistXnorNet(&network);
    MnistNet(&network);
    network.initial(BATCH, LEARNING_RATE);

    float *train_data = read_train_data(argv[3]);
    float *train_label = read_train_label(argv[3]);

    for(int iter = 0; iter < MAX_ITER; iter++) {

      ms_t start = getms();
      int step = (iter*BATCH)%60000;
      float *batch_xs = train_data + step*IM_SIZE;
      float *batch_ys = train_label + step*CLASSES;

      float *output = network.inference(batch_xs);
      network.train(batch_ys);
 
      float acc = accuracy(BATCH, CLASSES, output, batch_ys);

      if(iter%1 == 0) {
        cout << "iter = " << iter << ", time = " 
  	   << (getms() - start) << "ms, accuracy = " << acc << endl;
      }
    }

    network.save(argv[2]);
    delete []train_data;
    delete []train_label;
  }
  else if(strcmp(argv[1], "deploy") == 0) {
    network.load(argv[2], BATCH);
  }
  else {
    help();
  }



  float *test_data, *test_label;
  test_data = read_validate_data(argv[3]);
  test_label = read_validate_label(argv[3]);

  float total = 0.0;
  int batch_num = 10000/BATCH;

  network.deploy();

  ms_t start = getms();
  for(int iter = 0; iter < batch_num; iter++) {

    int step = (iter*BATCH);
    float *batch_xs = test_data + step*IM_SIZE;
    float *batch_ys = test_label + step*CLASSES;
    float *output = network.inference(batch_xs);
    total += accuracy(BATCH, CLASSES, output, batch_ys);
  }
  cout << "Validate set error = " << (total/batch_num)*100 
       << ", time = " << getms() -start  << endl;

  delete []test_data;
  delete []test_label;  

  return 0;
}


