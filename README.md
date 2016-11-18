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

##### Preliminary
Make sure there are 6 folders under the root of the project, including `code/`, `data/`, `log/`, `resource`, `result`, `script`:

> `code/`: All of the source code of this project.

> `data/`: The raw data and the protobuf file.

> 
