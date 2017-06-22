
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

def runLayers(input_data, layers, sess):
    opt = input_data
    for i in range(len(layers)):
        encoder_op = layers[i]
        opt = sess.run(encoder_op['encoder'], feed_dict={encoder_op['x']: opt})
    return opt


# In[3]:

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


# In[4]:

def create_temp_encoder(x, input_size, output_size):
    weight = tf.Variable(tf.random_normal([input_size, output_size]))
    bias = tf.Variable(tf.random_normal([output_size]))

    weight2 = tf.Variable(tf.random_normal([output_size, input_size]))
    bias2 = tf.Variable(tf.random_normal([input_size]))

    return create_encoder(weight, bias, weight2, bias2, x)


# In[5]:

def train_layer(training_epochs, batch_size, display_step, optimizer, cost, input_data, temp_encoder, layers):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        total_batch = int(len(input_data) / batch_size)
        for epoch in range(training_epochs):
            count = 0
            cst = 0.0

            for i in range(total_batch):
                end = count + batch_size
                if end > len(input_data):
                    end = len(input_data)

                batch_xs = runLayers(input_data[count:end], layers, session)
                _, c = session.run([optimizer, cost], feed_dict={temp_encoder['x']: batch_xs})

                cst = cst + c
                count = end

            cst = cst / total_batch
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(cst))

        print("Optimization Finished!")

        weight_e = tf.constant(session.run(temp_encoder['weight']))
        bias_e = tf.constant(session.run(temp_encoder['bias']))

        weight_d = tf.constant(session.run(temp_encoder['weightd']))
        bias_d = tf.constant(session.run(temp_encoder['biasd']))

    return create_encoder(weight_e, bias_e, weight_d, bias_d, temp_encoder['x'])


# In[6]:

def train(learning_rate, training_epochs, batch_size, display_step, input_data, steps):
    
    input_size = len(input_data[1])
    
    layers = []
    
    for step in steps:

        X = tf.placeholder("float", [None, input_size])
        temp_encoder = create_temp_encoder(X, input_size, step)

        y_pred = temp_encoder['decoder']
        y_true = temp_encoder['x']

        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
        
        layer = train_layer(training_epochs, batch_size, display_step, optimizer, cost, input_data, temp_encoder, layers)
        
        input_size = step
        
        layers.append(layer)

    return layers


# In[7]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# In[28]:

layers = train(0.01, 20, 256, 1, mnist.train.images, [256, 128])




# In[ ]:



