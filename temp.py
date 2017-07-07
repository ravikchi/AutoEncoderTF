import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Cost:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred


class Data:
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels
        self.index = 0

    def next_batch(self, batchSize):
        cur_index = self.index
        if (self.index + batchSize) > len(self.input):
            self.index = len(self.input)
        else:
            self.index += batchSize

        return self.input[cur_index:self.index], self.labels[cur_index:self.index]

class AutoEncoder:
    def __init__(self, input_size, hidden_size, act_func, inputX, sess=None):
        self.weight = tf.Variable(tf.random_normal([input_size, hidden_size]))
        self.bias = tf.Variable(tf.random_normal([hidden_size]))

        self.encoder = act_func(tf.add(tf.matmul(inputX, self.weight), self.bias))

        self.sess = sess

        self.weight_d = tf.transpose(self.weight)
        self.bias_d = tf.Variable(tf.random_normal([input_size]))

        self.decoder = act_func(tf.add(tf.matmul(self.encoder, self.weight_d), self.bias_d))

    def train(self, data):
        batch_xs, batch_ys = data.next_batch(256)

