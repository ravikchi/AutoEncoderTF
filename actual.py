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

def get_selected_input_data(location, nodes):
    with open(location) as csvfile:
        csv_data = list(csv.DictReader(csvfile))

    keyList = csv_data[0].keys()

    for element in keyList:
        if element == 'DOMAIN' or element == 'NODE_ID':
            continue
        values = set(float(data[element]) for data in csv_data)
        maximum = max(values)
        minimum = min(values)
        for data in csv_data:
            data[element] = (float(data[element]) - minimum) / (maximum - minimum)

    input_data = []
    for data in csv_data:
        element = []
        if data['NODE_ID'] in nodes :
            element.append(data['NODE_ID'])
            for key in keyList:
                if key == 'DOMAIN' or key == 'NODE_ID':
                    continue
                element.append(data[key])

            input_data.append(element)

    return input_data

def get_input_data(location):
    with open(location) as csvfile:
        csv_data = list(csv.DictReader(csvfile))

    keyList = csv_data[0].keys()

    for element in keyList:
        if element == 'DOMAIN' or element == 'NODE_ID':
            continue
        values = set(float(data[element]) for data in csv_data)
        maximum = max(values)
        minimum = min(values)
        for data in csv_data:
            data[element] = (float(data[element]) - minimum) / (maximum - minimum)

    input_data = []
    info_data = []
    for data in csv_data:
        element = []
        info = []
        for key in keyList:
            if key == 'DOMAIN' or key == 'NODE_ID':
                info.append(data[key])
                continue
            element.append(data[key])

        input_data.append(element)
        info_data.append(info)

    return input_data, info_data

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

first, second = get_input_data('data/input_data.csv')
orig_csv_data = np.array(first)
info_data = np.array(second)

train_part = orig_csv_data[:-1000]
test_part = orig_csv_data[-1000:]

train_data = Data(train_part, train_part)
test_data = Data(test_part, test_part)


with open('data/Supervised_data.csv') as csvfile:
    csv_data = list(csv.DictReader(csvfile))

s_data = []
for data in csv_data:
    element = []
    element.append(data['NODE_ID'])
    element.append(float(data['RATING']))

    s_data.append(element)

nodes = [row[0] for row in s_data]

supervised_input_data = get_selected_input_data('data/input_data.csv', nodes)

supervised_input_data.sort(key=lambda x: x[0])
s_data.sort(key=lambda x: x[0])

supervised_input = np.array([row[1:] for row in supervised_input_data])
supervised_labels = np.array([row[1:] for row in s_data])

supervised_data = Data(supervised_input[:256], supervised_labels[:256])

layers = []
sizes = [1024, 512, 256, 128]

# with tf.Session() as sess:
#     input_size = train_data.inp_size()
#
#     for i in range(len(sizes)):
#         size = sizes[i]
#         if len(layers) == 0:
#             layers.append(Denoising(i, input_size, size, tf.nn.sigmoid, sess=sess))
#         else:
#             layers.append(Denoising(i, input_size, size, tf.nn.sigmoid, sess=sess, previous=layers[-1]))
#
#         input_size = size
#
#     encoder_pt, inputX = finalLayer(layers)
#
#     layers.append(Denoising(len(sizes), input_size, 1, tf.nn.sigmoid, inputX=inputX, sess=sess, supervised=True, previous_graph=encoder_pt))
#
#     sess.run(tf.global_variables_initializer())
#
#     saver = tf.train.Saver()
#
#     for layer in layers[:-1]:
#         layer.train(train_data, num_of_epoch=80)
#
#     layers[-1].train(supervised_data, num_of_epoch=20)
#
#     saver.save(sess, "/tmp/main_model")

domain_test = []
domain_info = []

for i in range(len(info_data)):
    if info_data[i][0] == 'Swindon':
        domain_info.append(info_data[i])
        domain_test.append(orig_csv_data[i])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph("/tmp/main_model.meta")
    saver.restore(sess, tf.train.latest_checkpoint('/tmp/'))

    encoder, inputX = get_encoder(len(sizes) + 1, train_data.inp_size())

    error = tf.reduce_mean(tf.pow(encoder - inputX, 2))
    print(1 - sess.run(error, feed_dict={inputX: supervised_input[256:]}))

    oput = sess.run(encoder, feed_dict={inputX: domain_test})

    for i in range(len(oput)):
        print(str(domain_info[i][1]) +', '+ str(oput[i][0]))