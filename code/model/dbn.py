import math
import sys
import numpy as np
import tensorflow as tf

class DBN(object):
  def __init__(self, iters=20, mu=.001, layers = [128, 50, 50, 50, 5], 
               smoothing=1.0, batch_size=200):
    self.iters = iters
    self.mu = mu
    self.batch_size = batch_size
    self.smoothing = smoothing
    self.layers = layers
    self.network_layers = []
    self.weights = []
    self.biases = []
    weight_vars = []
    self.keep_prob = tf.placeholder(np.float32)
    self.input_keep_prob = tf.placeholder(np.float32)
    self.input_layer = tf.nn.dropout(
        tf.placeholder(np.float32, name='input'), self.input_keep_prob)
    self.ref_output = tf.placeholder(np.float32, name='ref_output')
    next_layer = self.input_layer
    for i in range(1, len(layers)):
      weight = tf.Variable(tf.random_normal([layers[i-1], layers[i]], stddev=1.0))
      bias = tf.Variable(tf.zeros([layers[i]]))
      self.weights.append(weight)
      self.biases.append(bias)
      weight_vars.extend((weight, bias))
      if i < len(layers) -1:
        op = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(next_layer, weight, bias)),
                           self.keep_prob)
      else:
        op = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(next_layer, weight, bias)),
                           1.0)
      self.network_layers.append(op)
      next_layer = op
    self.output_layer = next_layer

    # Experimental loss function.
    #zeros_dims = tf.pack([tf.shape(self.ref_output)[0], layers[-1]])
    #center = tf.fill(zeros_dims, 40.0)
    #distance_from_center = tf.pow(tf.abs(self.ref_output - center), 2)
    #loss = tf.pow(
    #    tf.reduce_sum(
    #        tf.mul(distance_from_center,
    #                tf.pow(tf.abs(self.ref_output - self.output_layer), 3))),
    #              .333)

    # Mean-square-error
    loss = tf.reduce_mean(tf.square(
        self.ref_output - self.output_layer))
    self.cost = tf.div(loss, self.batch_size)
    self.update_weights = tf.train.AdamOptimizer(self.mu).minimize(
        self.cost, var_list=weight_vars)

  def train(self, sess, inputs, targets, test_inputs, test_targets):
    tf.initialize_all_variables().run(session=sess)
    row, col = np.shape(inputs)
    batch, index = self._next_batch(inputs, targets, -self.batch_size)
    count = 0
    current_cost = 0
    total_updates = 0
    print >> sys.stderr, batch[0].shape
    print >> sys.stderr, batch[1].shape
    for epoch in range(self.iters):
      while count < row:
        if total_updates % 1000 == 0:
          cost = sess.run(self.cost, feed_dict={self.input_layer: test_inputs,
                                                self.ref_output: test_targets,
                                                self.keep_prob: 1.0,
                                                self.input_keep_prob: 1.0})
          print >> sys.stderr, '[DBN] test cost:\t' + str(cost)
          current_cost = sess.run(self.cost,
                                feed_dict={self.input_layer: inputs,
                                           self.ref_output: targets,
                                           self.keep_prob: 1.0,
                                           self.input_keep_prob: 1.0})
          print >> sys.stderr, '[DBN] train cost:\t' + str(current_cost)
        sess.run(self.update_weights, feed_dict={self.input_layer: batch[0],
                                                 self.ref_output: batch[1],
                                                 self.keep_prob: 1.0,
                                                 self.input_keep_prob: 1.0})
        count += self.batch_size
        total_updates += 1
        batch, index = self._next_batch(inputs, targets, index) 
      count = 0
    final_cost = sess.run(self.cost, feed_dict={self.input_layer: test_inputs,
                                               self.ref_output: test_targets,
                                               self.keep_prob: 1.0,
                                               self.input_keep_prob: 1.0})
    print >> sys.stderr, '[DBN] final cost:\t' + str(final_cost)
    train_outputs = sess.run(self.output_layer,
                             feed_dict={self.input_layer: inputs,
                                        self.keep_prob: 1.0,
                                        self.input_keep_prob: 1.0})
    test_outputs = sess.run(self.output_layer,
                            feed_dict={self.input_layer: test_inputs,
                                       self.keep_prob: 1.0,
                                       self.input_keep_prob: 1.0})
    return train_outputs, test_outputs
    

  def _next_batch(self, inputs, targets, last_index):
    row, col = np.shape(inputs)
    start = (last_index + self.batch_size) % row
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
    last_index += self.batch_size
    return [new_ins, target_vals], last_index


  def get_output(self, sess, inputs):
    outputs = sess.run(self.output_layer,
                       feed_dict={self.input_layer:inputs,
                                  self.keep_prob: 1.0,
                                  self.input_keep_prob: 1.0})
    return outputs
