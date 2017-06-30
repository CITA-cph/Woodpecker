from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random 
import tensorflow as tf
from tensorflow.python.framework import dtypes
from owl_py import communication_utils as comm
from timed_input import readUserInput
import training_settings as setts

# silence the tensorflow setup
comm.set_tf_message_level(comm.MessageLevel.ERROR)


def train(tens_in, tens_out):
    with tf.Session() as sess:

        # restore the saved graph
        new_saver = tf.train.import_meta_graph(setts.model_path + ".meta")
        new_saver.restore(sess, setts.model_path)
        print("Restored the model with " + str(len(sess.graph.get_operations())) + " ops from " + setts.model_path)

        # get the ops
        x = sess.graph.get_operation_by_name(name="var_x")
        y = sess.graph.get_operation_by_name(name="var_y")
        optimizer = sess.graph.get_operation_by_name("optimizer")
        cost = sess.graph.get_operation_by_name("cost")

        # get the tensors
        x = x.outputs[0]
        y = y.outputs[0]
        cost = cost.outputs[0]

        # train
        for epoch in range(setts.epochs):
            loss = 0

            for batch in range(int(tens_in.train / setts.batch_size)):

                this_seed = random.randint(1, 10000000)
                epoch_x = tens_in.next_batch_shuffled(setts.batch_size, this_seed)
                epoch_y = tens_out.next_batch_shuffled(setts.batch_size, this_seed)
                batch, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                loss += c

            if epoch % setts.print_every == 0: print("Epoch " + str(epoch) + " loss:", loss)

            if epoch % setts.save_every == 0:
                #if epoch > 0: 
                #    setts.batch_size *= 0.5 
                #    setts.batch_size = int(max(1, setts.batch_size))
                #    print("Batch size set to ", str(setts.batch_size))

                new_saver.save(sess, setts.model_path)
                print("Model with " + str(len(sess.graph.get_operations())) + " ops saved under " + setts.model_path)              

                input = readUserInput("type ""yes"" to quit the process", "no")
                if input == "yes": 
                    print("Exiting")
                    sess.close() 
                    break

    print("Ending")
    #exit()
 