import csv
import numpy as np
from Data import Data
from DenoisingAE import Denoising
import tensorflow as tf

def finalLayer(layers):
    input = layers[0].inputX
    for layer in layers:
        input = layer.act_func(tf.add(tf.matmul(input, layer.weight), layer.bias))

    return input, layers[0].inputX

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

def get_input_data(location):
    with open(location) as csvfile:
        csv_data = list(csv.DictReader(csvfile))

    keyList = csv_data[0].keys()

    for element in keyList:
        if element == 'DOMAIN' or element == 'NODE_ID' or element == 'RATING':
            continue
        values = set(float(data[element]) for data in csv_data)
        maximum = max(values)
        minimum = min(values)
        for data in csv_data:
            data[element] = (float(data[element]) - minimum) / (maximum - minimum)

    input_data = []
    output_data = []
    info_data = []
    for data in csv_data:
        element = []
        output = []
        info = [data['DOMAIN'], data['NODE_ID']]
        for key in keyList:
            if key == 'DOMAIN' or key == 'NODE_ID':
                continue

            if key == 'RATING':
                output.append(data[key])
                continue

            element.append(data[key])

        input_data.append(element)
        output_data.append(output)
        info_data.append(info)

    return input_data, output_data, info_data

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

first, second, third = get_input_data('data/input_data.csv')
orig_csv_data = np.array(first)
domain_info = np.array(third)

train_data = Data(orig_csv_data[:-1000], orig_csv_data[:-1000])
test_data = Data(orig_csv_data[-1000:], orig_csv_data[-1000:])

first, second, third = get_input_data('data/Supervised_data.csv')
csv_data = np.array(first)
output_data = np.array(second)

supervised_train_data = Data(csv_data[:256], output_data[:256])
supervised_test_data = Data(csv_data[256:], output_data[256:])

domain_test_input = []
domain_test_output = []
for i in range(len(domain_info)):
    domain = domain_info[i]
    if domain[0] == 'Swindon':
        domain_test_input.append(orig_csv_data[i])
        domain_test_output.append(domain)




layers = []
sizes = [1024, 512, 256, 128]

with tf.Session() as sess:
    input_size = train_data.inp_size()

    for i in range(len(sizes)):
        size = sizes[i]
        if len(layers) == 0:
            layers.append(Denoising(i, input_size, size, tf.nn.sigmoid, sess=sess))
        else:
            layers.append(Denoising(i, input_size, size, tf.nn.sigmoid, sess=sess, previous=layers[-1]))

        input_size = size

    encoder_pt, inputX = finalLayer(layers)

    layers.append(Denoising(len(sizes), input_size, 1, tf.nn.sigmoid, inputX=inputX, sess=sess, supervised=True, previous_graph=encoder_pt))

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for layer in layers[:-1]:
        layer.train(train_data, num_of_epoch=320)

    layers[-1].train(supervised_train_data, num_of_epoch=80)

    saver.save(sess, "/tmp/main_model1")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph("/tmp/main_model1.meta")
    saver.restore(sess, tf.train.latest_checkpoint('/tmp/'))

    decoder, inputX = mergeLayers(len(sizes), train_data.inp_size())

    error = tf.reduce_mean(tf.pow(decoder - inputX, 2))
    print(1 - sess.run(error, feed_dict={inputX: test_data.input}))

    encoder, inputX = get_encoder(len(sizes) + 1, train_data.inp_size())

    error = tf.reduce_mean(tf.pow(encoder - inputX, 2))
    print(1 - sess.run(error, feed_dict={inputX: supervised_test_data.input}))

    outputs = sess.run(encoder, feed_dict={inputX: domain_test_input})

    for i in range(len(outputs)):
        print(str(domain_test_output[i][1])+', '+str(outputs[i]))