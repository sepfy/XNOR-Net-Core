# XNOR-Net-Core
This is an implemenation of [XNOR-Net](https://arxiv.org/abs/1603.05279) in C++. I have tried to compress the weights of convolution filter to uint32_t or uint64_t, so there is no dependency and easy to use.

## Pre-trained Model

### MNIST
|  LeNet             | Accuracy | Size   | Model  |
|--------------------|----------|--------|--------|
| Full Precision Net | 99.34    | 2.4MB  |[mnist-20200103.net](https://drive.google.com/file/d/18aFsiuYSouM-Vemz-ZXx76h5O6Fw0AXo/view)|
| XNOR Net           | 98.37    | 191KB  |[mnist-20200103.xnor.net](https://drive.google.com/file/d/16ugP8cMDDC5wLPR598y_bMQogCZX9EG3/view)|

To test the model
```bash
$ make mnist
$ ./samples/mnist <train/deploy> <model name> <dataset>
```

### CIFAR-10
|  LeNet             | Accuracy | Size   | Model  |
|--------------------|----------|--------|--------|
| Full Precision Net |  68.26   |        |        |
| XNOR Net           |          |        |        |

To test the model
```
$ make cifar
$ ./samples/cifar <train/deploy> <model name> <dataset>
```


## Comaprison
Inference with LeNet, 1 batch on Raspberry Pi 3B.
To consider the restriction of embedded system, I compare the performance with OpenCV dnn module, because currently popular deep learning framework is not easy to compile to mobile or embedded platform. The model was trained by Tensorflow. See the detail of model and test program in [here](https://github.com/sepfy/tensorflow-tools/tree/master/cifar)

| Framework       |  Time (ms)  | Model Size |
|-----------------|-------------|------------|
| OpenCV DNN      |             |   2.6 MB   |
| XNOR-Net-Core   |             |   438 KB   |



## Support layers
* Convolution
* Max Pooling
* Fully Connected
* Relu
* Dropout
* Batch Normalization
* Softmax

