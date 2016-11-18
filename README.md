Skywalker
===

Introduction
---
It's a project from Georgia Tech Center of Music Technology. It used machine learning algorithms to learn patterns of human muscle movements, which will be mapped to a newly developed 5-fingered robotic arm. The system would allow amputees as well as able-bodied musicians to control expressive and subtle arm and finger movements which would enhance expression and utility for musical performances and other tasks.

This repo is a machine learning framework based on TensorFlow that used to help developer to process the raw data, train and prepare the neural network that the system needs. Currently, it provides:
- Data preprocessing:
    * Synchronize the hand gesture data and the ultrasound images of the forearm muscle.
    * Basic image processing methods for the ultrasound images, such as downscaling, binarization, cropping, and canny.
    * Standard data storage solution - protobuf (Google).
- Data visualization:
    * Visualize the protobuf file of the ultrasound images.
    * Visualize the output of the machine learning model.
- Custom training for the neural network:
    * Customize the training data.
    * Customize the type of the neural network.
    * Customize the structure of the neural network.
    * Customize the training parameters of the neural network.  

Usage
---

### Preliminary
Make sure there are 6 folders under the root of the project, including `code/`, `data/`, `log/`, `resource`, `result`, `script`:

> `code/`: All of the source code of this project.

> `data/`: The raw data and the protobuf file.

> `log/`: Log information.

> `resource/`: The well-trained model file (TensorFlow).

> `result/`: The output and result of the training process.  

> `script/`: The one-click scripts used for different kinds of tasks.

If you want to define your own protobuf file structure, please make sure the .proto file is under the root directory.

If you want to do preprocessing/visualizing on the raw data, please make sure the raw data file (ultrasound video and hand gesture data) are in the folder `data/`

If you want to train your neural network, please make sure the training data file (protobuf file) is in the folder `data/`

### Train Your Model
There are two models, which are CNN and NN (The simple neural network), for your option. All you need to do is:

1. Configure your parameters in `script/test_new_model`:

For CNN:

> Note: The input and output layer of the CNN are determined by the training data, so you only need to define the hidden layers of the CNN (convolutional layer and fully connected layer).

```shell
# The name of the output model file
model_file_name='model.test_cnn'
# The structure of the CNN
# - Before '#': The nodes number of the convolutional layers, each of the layer is divided by the comma
# - After '#': The nodes number of the fully connected hidden layers
layers='32,64#1024'
# The proportion between the quantity of training data and testing data.
# It will divide the training data into two parts by this ratio
train_test_ratio='9'
# The learning rate
learning_rate='.001'
# The type of the neural network
model='cnn'
```

2. Run the the script:

