from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
import training_settings as setts


def evaluate_model(data):

    h0 = {'weights': tf.Variable(tf.random_normal(shape=[setts.model_n_0, setts.model_n_1], dtype=dtypes.float64), name="W0"),
          'biases': tf.Variable(tf.random_normal(shape=[setts.model_n_1], dtype=dtypes.float64), name="B0")}
    l0 = tf.nn.sigmoid(tf.add(tf.matmul(data, h0['weights']), h0['biases']), name="layer_0")

    h1 = {'weights': tf.Variable(tf.random_normal(shape=[setts.model_n_1, setts.model_n_2], dtype=dtypes.float64), name="W1"),
          'biases': tf.Variable(tf.random_normal(shape=[setts.model_n_2], dtype=dtypes.float64), name="B1")}
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(l0, h1['weights']), h1['biases']), name="layer_1")

    h2 = {'weights': tf.Variable(tf.random_normal(shape=[setts.model_n_2, setts.model_n_3], dtype=dtypes.float64), name="W2"),
          'biases': tf.Variable(tf.random_normal(shape=[setts.model_n_3], dtype=dtypes.float64), name="B2")}
    l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, h2['weights']), h2['biases']), name="layer_2")

    h3 = {'weights': tf.Variable(tf.random_normal(shape=[setts.model_n_3, setts.model_n_4], dtype=dtypes.float64), name="W3"),
          'biases': tf.Variable(tf.random_normal(shape=[setts.model_n_4], dtype=dtypes.float64), name="B3")}
    l3 = tf.nn.sigmoid(tf.add(tf.matmul(l2, h3['weights']), h3['biases']), name="layer_3")

    return l3