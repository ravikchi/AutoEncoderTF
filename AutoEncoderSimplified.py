import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

min_cost = 0.000001

def runLayers(input_data, layers, sess):
    opt = input_data
    for i in range(len(layers)):
        encoder_op = layers[i]
        opt = sess.run(encoder_op['encoder'], feed_dict={inputX: opt})


    return opt

def runDecodeLayers(input_data, layers, sess, inputX):
    opt = input_data
    nn = None
    input = inputX
    for i in range(len(layers)):
        encoder_op = layers[i]
        input = tf.nn.sigmoid(tf.add(tf.matmul(input, encoder_op['weight']), encoder_op['bias']))

    layers.reverse()

    output = input
    for i in range(len(layers)):
        decoder_op = layers[i]
        output = tf.nn.sigmoid(tf.add(tf.matmul(output, decoder_op['weightd']), decoder_op['biasd']))

    return sess.run(output, feed_dict={inputX:input_data})

def auto_encoder(x, input_size, output_size):
    weight = tf.Variable(tf.random_normal([input_size, output_size]))
    bias = tf.Variable(tf.random_normal([output_size]))
    
    encoder = tf.nn.sigmoid(tf.add(tf.matmul(x, weight), bias))
    
    weight2 = tf.transpose(weight)
    bias2 = tf.Variable(tf.random_normal([input_size]))
    
    decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, weight2), bias2))
    
    return {
        'encoder':encoder,
        'weight':weight,
        'bias':bias,
        'decoder':decoder,
        'weightd':weight2,
        'biasd':bias2
    }


def train_encoder(optimiser, cost, input_data, batch_size, num_of_epoch, output_data, use_output, layers):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        input_size = len(input_data)

        total_batches = int(input_size/batch_size)

        oldcst = 10000
        cst_count = 0

        for epoch in range(num_of_epoch):
            count = 0

            cst = 0.0

            for i in range(total_batches):
                end = count + batch_size
                if end > input_size:
                    end = input_size

                batch_xs = runLayers(input_data[count:end], layers, sess)
                batch_ys = output_data[count:end]


                if use_output :
                    _, c = sess.run([optimiser, cost], feed_dict={inputX:batch_xs, outputX:batch_ys})
                else:
                    _, c = sess.run([optimiser, cost], feed_dict={inputX: batch_xs})

                cst = cst + c
                count = end

            cst = cst/total_batches

            if epoch % 1 == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=", "{:.9f}".format(cst))

            if cst < min_cost:
                break

            if cst >= oldcst :
                cst_count = cst_count + 1
            else:
                oldcst = cst
                cst_count = 0

            if cst_count > 30 :
                break

        print("finished")


def update_input_data(input_data, next_encoder_layer, size, batch_size):
    input_size = len(input_data)
    encoder = next_encoder_layer['encoder']

    output_data = np.empty([input_size, size])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        total_batches = int(input_size / batch_size)
        count = 0
        for i in range(total_batches):
            end = count + batch_size
            if end > input_size:
                end = input_size

            batch_xs = input_data[count:end]
            output = sess.run(encoder, feed_dict={inputX: batch_xs})

            output_data[count:end] = output

    return output_data

def stack_auto_encoders(inputX, input_data, steps, batch_size, num_of_epoch,outputX, output_data):
    input_len = len(input_data[1])
    
    layers = []

    for step in range(len(steps)):
        next_encoder_layer = auto_encoder(inputX, input_len, steps[step])

        y_true = inputX
        y_pred = next_encoder_layer['decoder']

        use_output = False

        #if step>=(len(steps)-1):
            #y_true = next_encoder_layer['encoder']
           # y_pred = outputX
           # use_output = True

        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)

        train_encoder(optimizer, cost, input_data, batch_size, num_of_epoch, output_data, use_output, layers)

        #input_data = update_input_data(input_data, next_encoder_layer,steps[step], batch_size)

        layers.append(next_encoder_layer)

        input_len = steps[step]

    return layers

input_len = len(mnist.train.images[1])

inputX = tf.placeholder("float", shape=None)

outputX = tf.placeholder(tf.float32, [None, 10])

layers = stack_auto_encoders(inputX, mnist.train.images, [256, 128], 256, 2000, outputX, mnist.train.labels)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    decode = runDecodeLayers(mnist.test.images[:10], layers, sess, inputX)

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(decode[i], (28, 28)))

    plt.show()


