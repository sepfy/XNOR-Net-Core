# NeuralNetwork
This is an implemenation of [XNOR-Net](https://arxiv.org/abs/1603.05279) in C++. There is no dependency and easy to use.

## Pre-trained Model

### MNIST
|  LeNet             | Accuracy | Size   | Model  |
|--------------------|----------|--------|--------|
| Full-Precision-Net | 99.34    |        |        |
| XNOR-Net           | 98.37    |        |        |

To test the model
```bash
$ make mnist
$ ./samples/mnist <train/deploy> <model name> <dataset>
```

### CIFAR-10
|  LeNet             | Accuracy | Size   | Model  |
|--------------------|----------|--------|--------|
| Full Precision Net |  0.6826  |        |        |
| XNOR Net           |          |        |        |

To test the model
```
$ make cifar
$ ./samples/cifar <train/deploy> <model name> <dataset>
```


## Comaprison
Inference with LeNet and 1 batch.

| Framework    | Model Size   | Time (ms)  |
|--------------|--------------|------------|
| Tensorflow   |              |            |
| XNORNetCore  |              |            |



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
