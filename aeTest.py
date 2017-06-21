from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10


def auto_encoder(x,input_size, output_size):
    weight = tf.Variable(tf.random_normal([input_size, output_size]))
    bias = tf.Variable(tf.random_normal([output_size]))

    encoder = tf.nn.sigmoid(tf.add(tf.matmul(x, weight), bias))

    weight2 = tf.Variable(tf.random_normal([output_size, input_size]))
    bias2 = tf.Variable(tf.random_normal([input_size]))

    decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, weight2), bias2))

    return {
        'encoder': encoder,
        'decoder': decoder,
        'x': x
    }

x = tf.placeholder("float", [None, 784])
encoder_op = auto_encoder(x, 784, 256)

y_pred = encoder_op['decoder']
y_true = x

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)
for epoch in range(training_epochs):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1),
              "cost=", "{:.9f}".format(c))

print("Optimization Finished!")

encode_decode = sess.run(
    y_pred, feed_dict={y_true: mnist.test.images[:examples_to_show]})


f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

plt.show()
