from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf
from tensorflow.python.framework import dtypes
from owl_py import communication_utils as comm
from network_model import evaluate_model
from timed_input import readUserInput
import training_settings as setts

# silence the tensorflow setup
comm.set_tf_message_level(comm.MessageLevel.ERROR)


def train_network(tens_in, tens_out, x, y):
    prediction = evaluate_model(x)
    cost = tf.reduce_mean(tf.square(prediction - y), name="cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name="optimizer").minimize(cost)
    print("Starting the session")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(setts.epochs):
            loss = 0.0
            cnt = 0.0

            for batch in range(int(tens_in.train / setts.batch_size)):
                this_seed = random.randint(1, 10000000)
                epoch_x = tens_in.next_batch_shuffled(setts.batch_size, this_seed)
                epoch_y = tens_out.next_batch_shuffled(setts.batch_size, this_seed)
                batch, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                cnt += 1.0
                loss += c

            if epoch % setts.print_every == 0: print("Epoch " + str(epoch) + " loss:", loss)

            if epoch % setts.save_every == 0:
                if epoch > 0: 
                    saver.save(sess, setts.model_path)
                    print("Model with " + str(len(sess.graph.get_operations())) + " ops saved under " + setts.model_path)
 
                    input = readUserInput("type ""yes"" to quit the process", "no")
                    if input == "yes": 
                        print("Exiting")
                        sess.close() 
                        break


    print("Ending")
    #exit()


def train(tens_in, tens_out):
    # placeholders for the data
    x = tf.placeholder(dtypes.float64, [None, setts.model_n_0], name="var_x")
    y = tf.placeholder(dtypes.float64, [None, setts.model_n_4], name="var_y")
    train_network(tens_in, tens_out, x, y)

