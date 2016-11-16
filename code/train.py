import arrow
import numpy as np
import tensorflow as tf
import model.dbn as dbn
import model.cnn as cnn
import lib.read_training_data as rtd
import sys
import os

data_file_name   = sys.argv[1]
model_file_name  = sys.argv[2]
HIDDEN_LAYERS    = map(int, sys.argv[3].strip().split(','))
TRAIN_TEST_RATIO = float(sys.argv[4])
LEARNING_RATE    = float(sys.argv[5])
model            = sys.argv[6]
# Get the raw data from protobuf file.
data_path  = '../data/'
model_path = '../resource/'
res_path   = '../result/'
features = []
labels   = []

# Read raw data from protobuf file
data_name, features, labels, _ = rtd.read_data_from_protobuf(data_path+data_file_name)
img_height = features.shape[1]
img_width  = features.shape[2]
features = features.reshape(len(features),len(features[0].flatten()))
labels   = labels[:,0:5]

layers   = [features.shape[1]] + HIDDEN_LAYERS + [labels.shape[1]]
print '------ [ %s ] ------' % arrow.now()
print 'The layers of the network:\t%s' % (layers)
print 'The size of the image:\t%s * %s' % (img_width, img_height)
print 'The size of feature:\t%s * %s' % features.shape
print 'The size of labels:\t%s * %s' % labels.shape
print 'The protobuf name:\t%s' % data_name
print 'A example of the feature data:\n%s' % features[0]
print 'A example of the label data:\n%s' % labels[0]

# Divide the raw data into the training part and testing part
start_test = end_train = int(float(len(features)) / float(TRAIN_TEST_RATIO + 1) * TRAIN_TEST_RATIO)
training_features = np.array(features[0:end_train])
training_labels   = np.array(labels[0:end_train])
testing_features  = np.array(features[start_test:len(features)])
testing_labels    = np.array(labels[start_test:len(labels)])

# Check path
if not os.path.exists(model_path):
    print 'The resource path is not existed, create a new one.'
    os.makedirs(model_path)
if not os.path.exists(res_path):
    print 'The result path is not existed, create a new one.'
    os.makedirs(res_path)

# Training
# _, input_size  = training_features.shape
# _, output_size = training_labels.shape
print '------ [ %s ] ------' % arrow.now()
print 'Create an instance of the neural network.'
if model == 'dbn':
    network = dbn.DBN(layers=layers, iters=1000, batch_size=100, mu=LEARNING_RATE) #.0001)
elif model == 'cnn':
    network = cnn.CNN(img_width=img_width, img_height=img_height, learning_rate=LEARNING_RATE, training_iters=200000, batch_size=128, display_step=10)
with tf.Session() as sess:
    print 'Start training...'
    tr, test = network.train(sess, training_features, training_labels, testing_features, testing_labels)
    np.savetxt(res_path + 'training_result.txt', tr)
    np.savetxt(res_path + 'testing_result.txt', test)

    print '------ [ %s ] ------' % arrow.now()
    print 'Training has been done, Save the model...'
    tf_saver = tf.train.Saver()
    tf_saver.save(sess, model_path + model_file_name)
