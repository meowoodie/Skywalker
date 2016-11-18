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

### Prepare the Training Data

Here is an example and some description that shows how to prepare your own training data by using the framework.

- Configure your parameters in `script/prepare_training_data.sh`:

```shell
# The location of the ultrasound video
VIDEO_PATH='../data/FingerBending4GB.mp4'
# The location of the hand gesture data
GLOVE_PATH='../data/finger_bending_mason_1.txt'
# The output:
# [type]_[name]_[dim]_[process_method]
PROTO_BUF_PATH='../data/protobuf_fingerbendingmason1_36'
# The timestamp of the start frame
start_time='1476463978.558'
# The start frame is the first valid frame which is decided by observing manually
start_frame='143'
# The frame rate of the video
frame_rate='30'
# The size of each of the buffer
# if you want to process all the data into only one buf, set buf_size='-1'
buf_size='-1'
# The number of the buffer that you want to output,
# if you want to process all the data, set buf_num='-1'
buf_num='-1'
# cropping number of pixels
# If you don't want to apply cropping, set -1,-1,-1,-1
# crop_box='53,18,45,169'
crop_box='-1,-1,-1,-1'
# Preprocessing mode
# Each processing step is divided by '#', e.g. [step1]#[step2]#...
# The before comma part is the name of processing, including: binary, canny, downscale
# The after comma part is the parameter of processing, including:
# - binary: threshold, if set -1 means automatically find threshold
# - downscale: the proportion of the downscaling
# - canny: sigma of the canny
proc_mode='downscale,0.05'
```

- Run the script:

```shell
cd script/
sh prepare_training_data.sh
```

### Visualize The Protobuf Data
- Configure your parameters in `script/visualize_protobuf.sh`:

```shell
# The protobuf file that you want to visualize
protobuf_path='../data/protobuf_fingerbendingmason1_36_all'
# if the frame id is set to be -1, visualize all the frame in the protobuf file
frame_id='-1'
# The size the each of the frame.
frame_size='36,36'
# The name of the output file
title='test_36'
```

- Run the script:

```shell
cd script/
sh visualize_protobuf.sh
```

### Train Your Model
There are two models, which are CNN and NN (The simple neural network), for your option. All you need to do is:

#### For CNN:
- Configure your parameters in `script/test_new_model.sh`:


```shell
# The type of the neural network
model='cnn'
# The training data file name, which should be put in the folder /data
protobuf_data_file_name='protobuf_fingerbendingmason1_36_all'
# The name of the output model file
model_file_name='model.test_cnn'
# The structure of the CNN
# - Before '#': The nodes number of the convolutional layers, each of the layer is divided by a comma
# - After '#': The nodes number of the fully connected hidden layers, each of the layer is divided by a comma
layers='32,64#1024'
# The proportion between the quantity of training data and testing data.
# It will divide the training data into two parts by this ratio
train_test_ratio='9'
# The learning rate
learning_rate='.001'
```

- Run the script:

```shell
cd script/
sh test_new_model.sh
```

#### For NN:
- Configure your parameters in `script/test_new_model.sh`:

```shell
model='dbn'
protobuf_data_file_name='protobuf_fingerbendingmason1_36_all'
model_file_name='model.test_dbn'
hidden_layers='50,100,100'
train_test_ratio='9'
learning_rate='.00005'
```

- Run the script:

```shell
cd script/
sh test_new_model.sh
```

> Note: The input and output layer of the CNN are determined by the training data, so you only need to define the hidden layers of the CNN (convolutional layer and fully connected layer).


