# NeuralNetwork
This is a implemenation of [XNOR-Net](https://arxiv.org/abs/1603.05279) in C++. It is no dependency and easy to use.
Becuase it doesn't implemet training with GPU. So

## Pretrain model
### MNIST hand writting dataset
A simple validation for convolution nerual network.

|                    | Accuracy | Size   | Model    |
|--------------------|----------|--------|----------|
| Full-Precision-Net | 99.34    | 120 MB |          |
| XNOR-Net           | 98.37    | 10 MB  |  |

- Run training:
```bash
$ make mnist
$ cd samples
$ ./mnist
```




### INRIA person dataset
This is a classifier which predicts whehter there is a person in the image or not.

| Resnet-18          | Accuracy | Size   | Model    |
|--------------------|----------|--------|----------|
| Full-Precision-Net |          |        |          |
| XNOR-Net           |          |        |          |

- Run training
```
$ make person
$ cd samples
$ ./person
```
- Image size is 224x224
- Classes is person and no-person
- [Evaluation on Raspberry Pi Zero](#)

## Comaprison
Framework     | Model Size   | Time (ms)
--------------|--------------| --------
Tensorflow    | 120MB        |  1200    
XNORNetCore   | 1MB          |   200

## How to use

### Build library
```
$ make
```


### API


### Support layers
* Convolution
* Max pooling
* Fully connected
* Relu
* Dropout
* Batch normalization
* Softmax (with loss)


## Todo
- Train with cuDNN.
- Inference optimization(NEON...)
- Object detecion and more models.
