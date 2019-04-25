import matplotlib.pyplot as plt
import tensorflow as tf
import math
import numpy as np
import pandas as pd

#READ INPUT FILES_________________________________________________________________________________________________
dir_path = ""

#TRAIN input matrix
df = pd.read_csv(dir_path + "concave_InputsTRAIN.csv", header=None)
print("Nbr columns: ", len(df.columns))
X = df[df.columns[0:3920]].values

#TRAIN output matrix
df = pd.read_csv(dir_path + "concave_OutputsTRAIN.csv", header=None)
print("Nbr columns: ", len(df.columns))
Y = df[df.columns[0:3920]].values

print(X)
print(Y)

#TEST input matrix
df = pd.read_csv(dir_path + "concave_InputsTEST.csv", header=None)
print("Nbr columns: ", len(df.columns))
A = df[df.columns[0:784]].values

#TEST output matrix
df = pd.read_csv(dir_path + "concave_OutputsTEST.csv", header=None)
print("Nbr columns: ", len(df.columns))
B = df[df.columns[0:784]].values

# print(A)
# print(B)

# Convert the dataset into train and test datasets
train_x = np.reshape(X,[5,28,28,1])
train_y = np.reshape(Y,[5,784])

test_x = np.reshape(A,[1,28,28,1])
test_y = np.reshape(B,[1,784])

# Inspect the shape of the train and test datasets
print("train_x.shape",train_x.shape)
print("train_y.shape",train_y.shape)
print("test_x.shape",test_x.shape)
print("test_y.shape",test_y.shape)

#BUILD MODEL__________________________________________________________________________________________________________

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here     // changed to 2
y_ = tf.placeholder(tf.float32, [None, 784])
#variable learning rate
lr = tf.Variable(0.001)

# Inspect the shape of the train and test datasets
print("x.shape",x.shape)
print("y_.shape",y_.shape)


# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 1568  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/784)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/784)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/784)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/784)
W5 = tf.Variable(tf.truncated_normal([N, 784], stddev=0.1))
B5 = tf.Variable(tf.ones([784])/784)

print(W1,W1.shape)
print(W2,W2.shape)
print(W3,W3.shape)
print(W4,W4.shape)
print(W5,W5.shape)

print(B1,B1.shape)
print(B2,B2.shape)
print(B3,B3.shape)
print(B4,B4.shape)
print(B5,B5.shape)

# The model
stride = 1  # output is 28x28
y1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
print("y1",y1.shape)

stride = 2  # output is 14x14
y2 = tf.nn.relu(tf.nn.conv2d(y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
print("y2",y2.shape)

stride = 2  # output is 7x7
y3 = tf.nn.relu(tf.nn.conv2d(y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
print(y3)
print("y3",y3.shape)

# reshape the output from the third convolution for the fully connected layer
yy = tf.reshape(y3, shape=[-1, 7 * 7 * M])
print("yy",yy.shape)

y4 = tf.nn.relu(tf.matmul(yy, W4) + B4)
print("y4",y4.shape)

ylogits = tf.matmul(y4, W5) + B5
print("y1ogits",ylogits.shape)

y= tf.nn.sigmoid(ylogits)
print("y",y.shape)

#INITIALIZE__________________________________________________________________________________________________________


# Initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()


training_epochs = 100
model_path = "./EW_try1"
learning_rate=0.03
# Define the cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# # Launch the graph
# sess = tf.Session()
# sess.run(init)
# sess.run(tf.global_variables_initializer())

mse_history = []
accuracy_history = []
cost_history = []

# Calculate the cost and the accuracy for each epoch
for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x: train_x, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     print("Accuracy: ", (sess.run(accuracy, feed_dict={x:test_x, y_:test_y})))
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)
    print('epoch: ', epoch, ' - ', 'cost: ', cost, " - MSE: ", mse_, "- Train Accuracy: ", accuracy)

save_path = saver.save(sess, model_path)
print("Model saved in file: %s", save_path)

plt.plot(accuracy_history)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(range(len(cost_history)), cost_history)
plt.axis([0, training_epochs, 0, np.max(cost_history) / 100])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
# Print the final mean square error
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.square(pred_y - test_y))
print("Test Accuracy: ", (sess.run(y, feed_dict={x: test_x, y_: test_y})))

# Print the final mean square error
pred_y = sess.run(y, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))

