import sys
import numpy as np
import model.dbn as dbn
import model.cnn as cnn
import tensorflow as tf
from lib.read_training_data import read_data_from_protobuf
from matplotlib import pyplot as plt

data_path       = '../data/'
model_file_path = '../resource/'

def predict(model_name, model_file_name, layers, input_features, target_size):
    # Restore the well-trained model
    if model_name == 'dbn':
        layers  = map(int, layers.strip().split(','))
        layers  = [input_features.shape[1]] + layers + [target_size]
        network = dbn.DBN(layers=layers, batch_size=100)
    elif model_name == 'cnn':
        conv_layers, hid_layers = layers.strip().split('#')
        conv_layers = map(int, conv_layers.strip().split(','))
        hid_layers  = map(int, hid_layers.strip().split(','))
        network = cnn.CNN(img_width=input_features.shape[1],img_height=input_features.shape[2], conv_layers=conv_layers, hidden_layers=hid_layers, batch_size=128)
    else:
        return -1, 'Invalid model'
    
    with tf.Session() as sess:
        tf_saver = tf.train.Saver()
        tf_saver.restore(sess, model_file_path + model_file_name)
        outputs = network.get_output(sess, input_features)
    return 0, outputs

def _output_contrasts(testing_outputs, real_outputs):
    for t,r in zip(testing_outputs, real_outputs):
        print '%s#%s' % ('\t'.join(map(str, t)), '\t'.join(map(str, r)))

if __name__ == '__main__':
    mode            = sys.argv[1]
    model_name      = sys.argv[2]
    model_file_name = sys.argv[3]
    layers          = sys.argv[4]
    
    if mode == 'offline_test':
        data_file_name   = sys.argv[5]
        train_test_ratio = int(sys.argv[6])
        # Read raw data from protobuf file
        name, features, labels, _ = read_data_from_protobuf(data_path + data_file_name) 
        features = features.reshape(len(features), len(features[0].flatten()))
        labels   = labels[:,0:5]
        feature_num = len(features[0])
        label_num   = len(labels[0])
        test_start  = int(float(train_test_ratio)/float(train_test_ratio + 1) * len(features)) 
        print >> sys.stderr, 'The input data file:\t%s' % name
        print >> sys.stderr, 'The layers:\t', layers
        # Predict
        code, outputs = predict(model_name, model_file_name, layers, features[test_start:], label_num)
        if code != 0:
            print >> sys.stderr, outputs
            exit(0)
        _output_contrasts(outputs, labels[test_start:])   
