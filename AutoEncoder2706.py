
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

def runDecodeLayers(layers):
    inputX = layers[0]['x']
    input = inputX
    for layer in layers:
        input = tf.nn.sigmoid(tf.add(tf.matmul(input, layer['weight']), layer['bias']))

    layers.reverse()

    output = input
    for layer in layers:
        output = tf.nn.sigmoid(tf.add(tf.matmul(output, layer['weightd']), layer['biasd']))

    layers.reverse()

    return output


def runEncodeLayers(layers):
    inputX = layers[0]['x']
    input = inputX
    for layer in layers:
        input = tf.nn.sigmoid(tf.add(tf.matmul(input, layer['weight']), layer['bias']))

    return input


# In[4]:

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


# In[5]:

def create_temp_encoder(x, input_size, output_size):
    weight = tf.Variable(tf.random_normal([input_size, output_size]))
    bias = tf.Variable(tf.random_normal([output_size]))

    weight2 = tf.transpose(weight)
    bias2 = tf.Variable(tf.random_normal([input_size]))

    return create_encoder(weight, bias, weight2, bias2, x)


# In[6]:

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

# In[7]:

def unsupervised_train(learning_rate, training_epochs, batch_size, display_step, input_data, steps):
    
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

def supervised_train(learning_rate, training_epochs, batch_size, display_step, input_data, step, output_data, layers, init_size):
    input_size = init_size

    X = tf.placeholder("float", [None, input_size])
    output_size = len(output_data[0])

    Y = tf.placeholder("float", [None, output_size])
    temp_encoder = create_temp_encoder(X, input_size, step)

    cost = tf.reduce_mean(tf.pow(Y - temp_encoder['encoder'], 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

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
                batch_ys = output_data[count:end]
                _, c = session.run([optimizer, cost], feed_dict={temp_encoder['x']: batch_xs, Y:batch_ys})

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

    layer = create_encoder(weight_e, bias_e, weight_d, bias_d, temp_encoder['x'])

    layers.append(layer)

    return layers



# In[ ]:




# In[8]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# In[9]:

learnin_rate = 0.01
num_epochs = 80
bat_size = 256
steps = [256]
display_steps = 5
final_step = 10
input_data = mnist.train.images
output_data = mnist.train.labels

layers = unsupervised_train(learnin_rate, num_epochs, bat_size, display_steps, input_data, steps)

examples_to_show = 10
inputX = layers[0]['x']

with tf.Session() as def_session:
    def_session.run(tf.global_variables_initializer())
    decoder = runDecodeLayers(layers)
    encode_decode = def_session.run(
        decoder, feed_dict={inputX: mnist.test.images[:examples_to_show]})

layers = supervised_train(learnin_rate, num_epochs, bat_size, display_steps, input_data, final_step, output_data, layers, steps[-1])

encoder = runEncodeLayers(layers)

Y = tf.placeholder("float", [None, final_step])

#with tf.Session() as def_session:
 #   def_session.run(tf.global_variables_initializer())
  #  output = def_session.run(encoder, feed_dict={inputX: mnist.test.images[:examples_to_show]})

correct_prediction = tf.equal(tf.argmax(encoder,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as def_session:
    def_session.run(tf.global_variables_initializer())
    print(def_session.run(accuracy, feed_dict={inputX: mnist.test.images, Y: mnist.test.labels}))

#for i in range(examples_to_show):
 #   print(mnist.test.labels[i])
  #  print(output[i])


f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

plt.show()
# In[10]:



# In[ ]:



