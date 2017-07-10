import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class RMSCost:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        self.cost = tf.reduce_mean(tf.pow(self.y_true-self.y_pred, 2))


class Data:
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels
        self.index = 0
        self.size = len(input)

    def next_batch(self, batchSize):
        cur_index = self.index
        if self.index == len(self.input):
            self.index = 0
            cur_index = self.index

        if (self.index + batchSize) > len(self.input):
            self.index = len(self.input)
        else:
            self.index += batchSize

        return self.input[cur_index:self.index], self.labels[cur_index:self.index]

class AutoEncoder:
    def __init__(self, id, input_size, hidden_size, act_func, inputX=None, sess=None, previous=None):
        self.id = id

        self.weight = tf.Variable(tf.random_normal([input_size, hidden_size]), name="weight_"+str(id))
        self.bias = tf.Variable(tf.random_normal([hidden_size]), name="bias_"+str(id))

        if inputX:
            self.inputX = inputX
        else:
            self.inputX = tf.placeholder('float', [None, input_size])

        if previous:
            self.previous = previous
        else:
            self.previous = None

        self.encoder = act_func(tf.add(tf.matmul(self.inputX, self.weight), self.bias))

        self.sess = sess

        self.weight_d = tf.transpose(self.weight)
        self.bias_d = tf.Variable(tf.random_normal([input_size]), name="bias_d_"+str(id))

        self.decoder = act_func(tf.add(tf.matmul(self.encoder, self.weight_d), self.bias_d))

        self.saver = tf.train.Saver()

    def output(self, data, batch_size):
        input_data, output_data = data.next_batch(batch_size)
        if self.previous:
            input_data = self.previous.output(input_data, self.tf_session)
            input_data = self.tf_session.run(self.encoder, feed_dict={self.inputX:input_data})

        return input_data, output_data

    def train(self, data, num_of_epoch=20, learning_rate=0.01, batch_size=256):

        y_true = self.inputX
        y_pred = self.decoder

        cost = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

        self.sess.run(tf.global_variables_initializer())

        total_batches = int(data.size / batch_size)

        for epoch in range(num_of_epoch):

            for i in range(total_batches):
                batch_xs, batch_ys = self.output(data, batch_size)
                _,c = self.sess.run([optimizer, cost],feed_dict={self.inputX: batch_xs})

            print(epoch)
            print(c)

        self.saver.save(self.sess, "/tmp/my_model")

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

examples_to_show = 10

input_data = Data(mnist.train.images, mnist.train.labels)
layers = []

with tf.Session() as sess:
    layer = AutoEncoder(1, 784, 256, tf.nn.sigmoid, sess=sess)
    layers.append(layer)
    layer.train(input_data)
    layer = AutoEncoder(2, 256, 128, tf.nn.sigmoid, sess=sess, previous=layer)
    layers.append(layer)
    layer.train(input_data)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph("/tmp/my_model.meta")
    saver.restore(sess, tf.train.latest_checkpoint('/tmp/'))
    encode_decode = sess.run(
        layer.decoder, feed_dict={layer.inputX: mnist.test.images[:examples_to_show]})

        # for i in range(examples_to_show):
        #   print(mnist.test.labels[i])
        #  print(output[i])

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

    plt.show()

