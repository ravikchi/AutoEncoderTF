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

def create_encoder(weight_e, bias_e, weight_d, bias_d, x):
    encoder = tf.nn.sigmoid(tf.add(tf.matmul(x, weight_e), bias_e))
    decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, weight_d), bias_d))

    return {
        'encoder': encoder,
        'weight': weight_e,
        'bias': bias_e,
        'decoder': decoder,
        'weightd': weight_d,
        'biasd': bias_d,
        'x': x
    }

def auto_encoder(x, input_size, output_size):
    weight = tf.Variable(tf.random_normal([input_size, output_size]))
    bias = tf.Variable(tf.random_normal([output_size]))

    weight2 = tf.Variable(tf.random_normal([output_size, input_size]))
    bias2 = tf.Variable(tf.random_normal([input_size]))

    return create_encoder(weight, bias, weight2, bias2, x)



def train():
    X = tf.placeholder("float", [None, 784])
    temp_encoder = auto_encoder(X, 784, 256)

    y_pred = temp_encoder['decoder']
    y_true = temp_encoder['x']

    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        total_batch = int(mnist.train.num_examples / batch_size)
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c = session.run([optimizer, cost], feed_dict={temp_encoder['x']: batch_xs})
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        weight_e = tf.constant(session.run(temp_encoder['weight']))
        bias_e = tf.constant(session.run(temp_encoder['bias']))

        weight_d = tf.constant(session.run(temp_encoder['weightd']))
        bias_d = tf.constant(session.run(temp_encoder['biasd']))

        return create_encoder(weight_e, bias_e, weight_d, bias_d, temp_encoder['x'])

encoder = train()

with tf.Session() as def_session:
    def_session.run(tf.global_variables_initializer())
    encode_decode = def_session.run(
        encoder['decoder'], feed_dict={encoder['x']: mnist.test.images[:examples_to_show]})


    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

    plt.show()
