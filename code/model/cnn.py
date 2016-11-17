import sys
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

class CNN(object):

    def __init__(self, img_width=36, img_height=36, learning_rate=0.001, training_iters=200000, batch_size=128, display_step=10):
        # Parameters
        # self.learning_rate = 0.001
        self.training_iters = training_iters
        self.batch_size     = batch_size
        self.display_step   = display_step

        # Network Parameters
        self.img_width  = img_width
        self.img_height = img_height
        self.n_input    = self.img_width * self.img_height # MNIST data input (img shape: 28*28)
        self.n_output   = 5      # MNIST total classes (0-9 digits)
        self.dropout    = 0.75    # Dropout, probability to keep units

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_output])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        # Store layers weight & bias
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([9*9*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.n_output]))
        }

        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.n_output]))
        }

        # Construct model
        self.output_layer = self.conv_net(self.x, weights, biases, self.keep_prob)

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
    def conv_net(self, x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, self.img_width, self.img_height, 1])

        # Convolution Layer
        conv1 = self._conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self._maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self._conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self._maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
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
                # print "Iter " + str(step * self.batch_size) + ", Minibatch Loss= " + \
                #     "{:.6f}".format(loss) + ", Training Accuracy= " + \
                #     "{:.5f}".format(acc)
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
