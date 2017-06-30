from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from owl_py import tbin_numpy as tbin
import training_settings as setts 
 

def extract_network():
    with tf.Session() as sess:
        # restore the saved graph
        new_saver = tf.train.import_meta_graph(setts.model_path + ".meta")
        new_saver.restore(sess, setts.model_path)
        print("Restored the model with " + str(len(sess.graph.get_operations())) + " ops from " + setts.model_path)

        h1_w = sess.graph.get_operation_by_name(name="W" + str(0)).outputs[0]
        h2_w = sess.graph.get_operation_by_name(name="W" + str(1)).outputs[0]
        h3_w = sess.graph.get_operation_by_name(name="W" + str(2)).outputs[0]
        h4_w = sess.graph.get_operation_by_name(name="W" + str(3)).outputs[0]

        h1_b = sess.graph.get_operation_by_name(name="B" + str(0)).outputs[0]
        h2_b = sess.graph.get_operation_by_name(name="B" + str(1)).outputs[0]
        h3_b = sess.graph.get_operation_by_name(name="B" + str(2)).outputs[0]
        h4_b = sess.graph.get_operation_by_name(name="B" + str(3)).outputs[0]

        weights = [h1_w.eval(None, sess), h2_w.eval(None, sess), h3_w.eval(None, sess), h4_w.eval(None, sess)]
        biases = [h1_b.eval(None, sess), h2_b.eval(None, sess), h3_b.eval(None, sess), h4_b.eval(None, sess)]

        print("Extracted " + str(len(weights)) + " weight tensors")
        print("Extracted " + str(len(biases)) + " bias tensors")

        tbin.save_multiple_tbin(setts.save_path_weights, weights)
        tbin.save_multiple_tbin(setts.save_path_biases, biases)

        print("Done")