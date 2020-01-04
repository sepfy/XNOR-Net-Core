# NeuralNetwork
This is an implemenation of [XNOR-Net](https://arxiv.org/abs/1603.05279) in C++. There is no dependency and easy to use.

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
Inference with LeNet, 1 batch and single core on Raspberry Pi 3B.

| Framework            |  Time (ms)  |
|----------------------|-------------|
| Full Precision Net   |             |
| XNOR Net             |             |



## Support layers
* Convolution
* Max Pooling
* Fully Connected
* Relu
* Dropout
* Batch Normalization
* Softmax


## Todo
* Train with GPU.
