import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

min_cost = 0.000001

def runLayers(input_data, layers):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        opt = input_data
        for i in range(len(layers)):
            encoder_op = layers[i]
            opt = sess.run(encoder_op['encoder'], feed_dict={inputX: opt})


        return opt

def auto_encoder(x, input_size, output_size):
    weight = tf.Variable(tf.random_normal([input_size, output_size]))
    bias = tf.Variable(tf.random_normal([output_size]))
    
    encoder = tf.nn.relu(tf.add(tf.matmul(x, weight), bias))
    
    weight2 = tf.transpose(weight)
    bias2 = tf.Variable(tf.random_normal([input_size]))
    
    decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, weight2), bias2))
    
    return {
        'encoder':encoder,
        'weight':weight,
        'bias':bias,
        'decoder':decoder
    }


def train_encoder(optimiser, cost, input_data, batch_size, num_of_epoch, output_data, use_output):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        input_size = len(input_data)
        
        total_batches = int(input_size/batch_size)
        
        for epoch in range(num_of_epoch):
            count = 0
            
            cst = 0.0
            for i in range(total_batches):
                end = count + batch_size
                if end > input_size:
                    end = input_size
                
                batch_xs = input_data[count:end]
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

        y_true = next_encoder_layer['decoder']
        y_pred = inputX

        use_output = False

        if step>=(len(steps)-1):
            y_true = next_encoder_layer['encoder']
            y_pred = outputX
            use_output = True

        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)

        train_encoder(optimizer, cost, input_data, batch_size, num_of_epoch, output_data, use_output)

        input_data = update_input_data(input_data, next_encoder_layer,steps[step], batch_size)
        
        layers.append(next_encoder_layer)

        input_len = steps[step]
    
    return layers

input_len = len(mnist.train.images[1])

inputX = tf.placeholder("float", shape=None)

outputX = tf.placeholder(tf.float32, [None, 10])

layers = stack_auto_encoders(inputX, mnist.train.images, [256, 128,10], 100, 40, outputX, mnist.train.labels)

print(runLayers(mnist.train.images[10:11], layers))
print(mnist.train.labels[10:11])



