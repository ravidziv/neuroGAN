import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

from utils import *

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def deconv1d(value, input, stride, output_shape=1):
    strides = [1, 1, stride, 1]
    value = tf.reshape(value, [input.get_shape()[0].value, -1, 1])
    filters = tf.Variable(tf.truncated_normal(shape=(3, 1, 1), mean=1, stddev=0.1))
    new_x = tf.expand_dims(value, 1)
    new_W = tf.expand_dims(filters, 0)
    output_shape = calculate_output_shape(new_x, 1, 1)
    deconv = tf.nn.conv2d_transpose(new_x, new_W, output_shape, strides)
    deconv = tf.squeeze(deconv, axis=1)
    deconv = tf.reshape(deconv, [input.get_shape()[0].value, -1])
    return deconv

def calculate_output_shape(in_layer, n_kernel, kernel_size, border_mode='same'):
	"""
	Always assumes stride=1
	"""
	in_shape = in_layer.get_shape() # assumes in_shape[0] = None or batch_size
	out_shape = [s.value for s in in_shape] # copy
	out_shape[-1] = n_kernel # always true
	if border_mode=='same':
		out_shape[1] = in_shape[1].value
		out_shape[2] = in_shape[2].value
	elif border_mode == 'valid':
		out_shape[1] = in_shape[1].value+kernel_size - 1
		out_shape[2] = in_shape[2].value+kernel_size - 1
	return out_shape
