import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Data import Data
from DenoisingAE import Denoising

def mergeLayers(size, input_size):
    graph = tf.get_default_graph()
    inputX = tf.placeholder('float', [None, input_size])
    input = inputX
    for i in range(size):
        val = i
        weight = graph.get_tensor_by_name("weight_" + str(val) + ":0")
        bias = graph.get_tensor_by_name("bias_" + str(val) + ":0")

        input = tf.nn.sigmoid(tf.add(tf.matmul(input, weight), bias))

    for i in range(size):
        val = i + 1
        weight = tf.transpose(graph.get_tensor_by_name("weight_" + str(size - val) + ":0"))
        bias = graph.get_tensor_by_name("bias_d_" + str(size - val) + ":0")

        input = tf.nn.sigmoid(tf.add(tf.matmul(input, weight), bias))

    return input, inputX


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

examples_to_show = 10

input_data = Data(mnist.train.images, mnist.train.labels)
layers = []
sizes = [1024, 512, 256, 128]

with tf.Session() as sess:
    input_size = input_data.inp_size()

    for i in range(len(sizes)):
        size = sizes[i]
        if len(layers) == 0:
            layers.append(Denoising(i, input_size, size, tf.nn.sigmoid, sess=sess))
        else:
            layers.append(Denoising(i, input_size, size, tf.nn.sigmoid, sess=sess, previous=layers[-1]))

        input_size = size

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for layer in layers:
        layer.train(input_data, num_of_epoch=40)

    saver.save(sess, "/tmp/my_model")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph("/tmp/my_model.meta")
    saver.restore(sess, tf.train.latest_checkpoint('/tmp/'))

    decoder, inputX = mergeLayers(len(sizes), input_data.inp_size())

    encode_decode = sess.run(decoder, feed_dict={inputX: mnist.test.images[:examples_to_show]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

    plt.show()