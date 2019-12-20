# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/ops.py
#   + License: MIT

import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.contrib import rnn

from utils import *


def kaiming(shape, dtype, partition_info=None):
    return(tf.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        return conv

def conv2d_transpose(input_, output_shape, d_h, d_w,
                     k_h=3, k_w=3, stddev=0.02,
                     name="conv2d_transpose", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def bidirectional_lstm(input_, cond, n_hidden=256, n_steps=32, n_input=54, name='bidirec_lstm'):
    with tf.variable_scope(name):

        print('new_lstm discrim')
        # weights = tf.get_variable('weights', [4096, 1],
        #                     initializer=tf.random_normal_initializer(stddev=0.02))

        # biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0.0))


        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        input_x = tf.unstack(input_, n_steps, 1)
        # print(image.shape)s
        # print('-----------------------------------x shape: ', x[0].get_shape())

        # Calculate shifts
        x = []
        for i in range(n_steps-1):
            x.append(tf.concat([input_x[i], input_x[i+1] - input_x[i], cond], 1))


        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn.LSTMBlockCell(n_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.LSTMBlockCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = rnn.stack_bidirectional_rnn([lstm_fw_cell], [lstm_bw_cell], x,
                                                  dtype=tf.float32)
        except Exception: # Old TensorFlow version only returns outputs not states
            outputs = rnn.stack_bidirectional_rnn([lstm_fw_cell], [lstm_bw_cell], x,
                                            dtype=tf.float32)

        h = tf.concat(outputs, 1)


        h, h_w, h_b = linear(h, 1024, 'd_h3_lin', with_w=True)
        h = tf.nn.relu(h)


        h, h_w, h_b = linear(h, 1, 'd_h4_lin', with_w=True)

        return h



def bi_lstm_class(input_, n_hidden=256, n_steps=32, n_input=54, num_class = 10, name='class_bi_lstm'):
    with tf.variable_scope(name):

        input_x = tf.unstack(input_, n_steps, 1)
        lstm_fw_cell = rnn.LSTMBlockCell(n_hidden, forget_bias=1.0)
        lstm_bw_cell = rnn.LSTMBlockCell(n_hidden, forget_bias=1.0)

        x = []
        for i in range(n_steps-1):
            x.append(tf.concat([input_x[i], input_x[i+1] - input_x[i]], 1))

        try:
            outputs, _, _ = rnn.stack_bidirectional_rnn([lstm_fw_cell], [lstm_bw_cell], x,
                                                  dtype=tf.float32)
        except Exception: 
            outputs = rnn.stack_bidirectional_rnn([lstm_fw_cell], [lstm_bw_cell], x,
                                            dtype=tf.float32)

        h = tf.concat(outputs, 1)

        h, h_w, h_b = linear(h, 1024, 'd_h3_lin', with_w=True)
        h = tf.nn.relu(h)


        h, h_w, h_b = linear(h, num_class, 'd_h4_lin', with_w=True)

        return h



def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def minibatch_discrimination(input_layer, name='minibatch_discrim'):
    # batch_size = input_layer.shape[0]
    # num_features = input_layer.shape[1]
    with tf.variable_scope(name):
        # W = tf.get_variable("W", [num_features, num_kernels*dim_per_kernel], tf.float32,
        #                          tf.contrib.layers.xavier_initializer())
        # b = tf.get_variable("b", [num_kernels],
        #     initializer=tf.constant_initializer(0.0))
        # activation = tf.matmul(input_layer, W)
        # activation = tf.reshape(activation, [tf.shape(input_layer)[0], num_kernels, dim_per_kernel])
        activation = input_layer
        tmp1 = tf.expand_dims(activation, 3)
        tmp2 = tf.transpose(activation, perm=[1,2,0])
        tmp2 = tf.expand_dims(tmp2, 0)
        abs_diff = tf.reduce_sum(tf.abs(tmp1 - tmp2), reduction_indices=[2])
        f = tf.reduce_sum(tf.exp(-abs_diff), reduction_indices=[2])
        # f = f + b
        return f
