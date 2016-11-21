import sys
import numpy as np
import tensorflow as tf

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

class CNN(object):

    def __init__(self, img_width=36, img_height=36, 
                 conv_layers=[32,64], hidden_layers=[1024], 
                 learning_rate=0.001, training_iters=200000, 
                 batch_size=128, display_step=10):
        # Parameters
        # self.learning_rate = 0.001
        self.training_iters = training_iters
        self.batch_size     = batch_size
        self.display_step   = display_step

        # Network Parameters
        self.img_width   = img_width
        self.img_height  = img_height
        self.n_input     = self.img_width * self.img_height # data input
        self.n_output    = 5       # number of CNN output 
        self.dropout     = 0.75    # Dropout, probability to keep units

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_output])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        # Store layers weight & bias
        # The conv layers
        conv_weights = []
        conv_biases  = []
        conv_layers = [1] + conv_layers
        print >> sys.stderr, '[CNN] conv layers:\t', conv_layers
        for i in range(1, len(conv_layers)):
            weight = tf.Variable(tf.random_normal([5, 5, conv_layers[i-1], conv_layers[i]]))
            bias   = tf.Variable(tf.random_normal([conv_layers[i]]))
            conv_weights.append(weight)
            conv_biases.append(bias)
        # The hidden layers
        hidden_weights = []
        hidden_biases  = []
        # The first hidden layer whose input is the output of the last conv layer
        hidden_input_size = (img_width / (2 ** (len(conv_layers) - 1))) * \
                            (img_height / (2 ** (len(conv_layers) - 1))) * \
                            conv_layers[-1]
        hidden_layers = [hidden_input_size] + hidden_layers + [self.n_output]
        print >> sys.stderr, '[CNN] hid layers:\t', hidden_layers
        for i in range(1, len(hidden_layers)):
            weight = tf.Variable(tf.random_normal([hidden_layers[i-1], hidden_layers[i]]))
            bias   = tf.Variable(tf.random_normal([hidden_layers[i]]))
            hidden_weights.append(weight)
            hidden_biases.append(bias)

        # Construct model
        self.output_layer = self.conv_net(self.x,
                                          conv_weights, conv_biases,
                                          hidden_weights, hidden_biases,
                                          self.keep_prob)

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output_layer, self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Create some wrappers for simplicity
    def _conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def _maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                            padding='SAME')


    # Create model
    def conv_net(self, x, conv_weights, conv_biases, hid_weights, hid_biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, self.img_width, self.img_height, 1])
        
        # Convolution layer
        layer = x
        for i in range(0, len(conv_weights)):
            # Conv
            layer = self._conv2d(layer, conv_weights[i], conv_biases[i])
            # Max Pooling (down-sampling)
            layer = self._maxpool2d(layer, k=2)

        # Fully connected hidden layer
        # Reshape the output of the conv to fit the first fully connected layer input
        hid = tf.reshape(layer, [-1, hid_weights[0].get_shape().as_list()[0]])
        hid = tf.add(tf.matmul(hid, hid_weights[0]), hid_biases[0])
        hid = tf.nn.relu(hid)
        hid = tf.nn.dropout(hid, dropout)
        # The remaining hidden layers
        for i in range(1, len(hid_weights) - 1):
            hid = tf.nn.relu(tf.add(tf.matmul(hid, hid_weights[i]), hid_biases[i]))
            hid = tf.nn.dropout(hid, dropout)

        # Output layer
        out = tf.add(tf.matmul(hid, hid_weights[-1]), hid_biases[-1])

        return out
    
    def _next_batch(self, inputs, targets, start_index):
        row = np.shape(inputs)[0]
        start = start_index % row
        end = (start + self.batch_size) % row
        if end < start:
            new_ins = np.append(inputs[start: row], inputs[0: end],
                                axis=0).astype(np.float32)
            new_targets = np.append(targets[start: row], targets[0: end],
                                    axis=0).astype(np.float32)
        else:
            new_ins = inputs[start: end].astype(np.float32)
            new_targets = targets[start: end].astype(np.float32)

        target_vals = new_targets
        start_index += self.batch_size
        return new_ins, target_vals, start_index

    def train(self, sess, inputs, targets, test_inputs, test_targets):
        # Initializing the variables
        init = tf.initialize_all_variables()
        sess.run(init)
        step = 1
        start_index = 0
        # Keep training until reach max iterations
        while step * self.batch_size < self.training_iters:
            # batch_x, batch_y = mnist.train.next_batch(self.batch_size)
            batch_x, batch_y, start_index = self._next_batch(inputs, targets, start_index)
            # Run optimization op (backprop)
            sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.dropout})
            if step % self.display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run(
                    [self.cost, self.accuracy], 
                    feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.}
                )
                test_loss, test_acc = sess.run(
                    [self.cost, self.accuracy],
                    feed_dict={self.x: test_inputs, self.y: test_targets, self.keep_prob: 1.}
                )
                print >> sys.stderr, '[CNN] Iter: %s' % str(step * self.batch_size)
                print >> sys.stderr, '[CNN] Train:\tloss (%.5f)\tacc (%.5f)' % (loss, acc)
                print >> sys.stderr, '[CNN] Test:\tloss (%.5f)\tacc (%.5f)' % (test_loss, test_acc)
            step += 1
        print >> sys.stderr, '[CNN] Optimization Finished!'

        # Calculate accuracy for 256 mnist test images
        # print("Testing Accuracy:", \
        #     sess.run(
        #         self.accuracy, 
        #         feed_dict={
        #             self.x: mnist.test.images[:256], 
        #             self.y: mnist.test.labels[:256], 
        #             self.keep_prob: 1.
        #         }
        #     )
        # )

        # train_outputs = sess.run(
        #     self.output_layer,
        #     feed_dict={self.x: inputs, self.keep_prob: 1.0}
        # )
        # test_output = sess.run(
        #     self.output_layer,
        #     feed_dict={self.x: test_inputs, self.keep_prob: 1.0}
        # )
        # return None, None #train_outputs, test_output

    def get_output(self, sess, inputs):
        outputs = sess.run(
                self.output_layer, 
                feed_dict={self.x: inputs, self.keep_prob: 1.0}
        )
        return outputs

if __name__ == '__main__':
    pass
