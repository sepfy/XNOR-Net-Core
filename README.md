# NeuralNetwork


## Pretrain model

### MNIST hand writting dataset

|                    | Accuracy | Size   | Model    |
|--------------------|----------|--------|----------|
| Full-Precision-Net | 99.34    | 120 MB |          |
| XNOR-Net           | 98.3     | 10 MB  |  |


### INRIA person dataset
This is a classifier which predicts whehter there is a person in the image or not.
- Image size is 224x224
- Classes is person and no-person

| Resnet-18          | Accuracy | Size   | Model    |
|--------------------|----------|--------|----------|
| Full-Precision-Net |     |  |          |
| XNOR-Net           |      |   |   |

- [Evaluation on Raspberry Pi Zero](#)



## API


## Todo
- Train with cuDNN.
- Inference optimization(NEON...)
- Object detecion and more models.
