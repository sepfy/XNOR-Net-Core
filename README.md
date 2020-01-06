# XNOR-Net-Core

![](https://github.com/sepfy/XNOR-Net-Core/workflows/build/badge.svg)

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
| Full Precision Net |  68.29   | 2.9MB  |[cifar-20191224.net](https://drive.google.com/file/d/1Pmzzx0Ie7U3Swkd3KDs0E7CThG5wB6_V/view)|
| XNOR Net           |  64.2    | 438KB  |[cifar-20200106.xnor.net](https://drive.google.com/file/d/15UU8bquuMD7DPGKGiGZ-CGRt75TyBEqh/view)|

To test the model
```
$ make cifar
$ ./samples/cifar <train/deploy> <model name> <dataset>
```


## Comaprison
Inference with LeNet, 1 batch on Raspberry Pi 3B.
To consider the restriction of embedded system, I compare the performance with OpenCV dnn module(3.4), because currently popular deep learning framework is not easy to compile to mobile or embedded platform. The model was trained by Tensorflow. See the detail of model in [here](https://github.com/sepfy/tensorflow-tools/tree/master/cifar)

| Framework       |  Time (ms)  | Model Size |
|-----------------|-------------|------------|
| OpenCV DNN      |    48       |   2.6 MB   |
| XNOR-Net-Core   |    28       |   438 KB   |

I got 60% optimization of inference time and compress model size to 15%.

## Support layers
* Convolution
* Max Pooling
* Fully Connected
* Relu
* Dropout
* Batch Normalization
* Softmax

