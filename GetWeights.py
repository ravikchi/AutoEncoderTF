import csv
import numpy as np
from Data import Data
from DenoisingAE import Denoising
import random
import tensorflow as tf


def get_encoder(size, input_size):
    graph = tf.get_default_graph()
    inputX = tf.placeholder('float', [None, input_size])
    input = inputX
    for i in range(size):
        val = i
        weight = graph.get_tensor_by_name("weight_" + str(val) + ":0")
        bias = graph.get_tensor_by_name("bias_" + str(val) + ":0")

        input = tf.nn.sigmoid(tf.add(tf.matmul(input, weight), bias))

    return input, inputX


layers = []
sizes = [100, 100, 50, 50]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph("/tmp/trained_model.meta")
    saver.restore(sess, tf.train.latest_checkpoint('/tmp/'))

    graph = tf.get_default_graph()
    inputX = tf.placeholder('float', [None, 17])
    input = inputX
    for i in range(5):
        val = i
        weight = graph.get_tensor_by_name("weight_" + str(val) + ":0")
        bias = graph.get_tensor_by_name("bias_" + str(val) + ":0")

        print(sess.run(weight))
        print(sess.run(bias))