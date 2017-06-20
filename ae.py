from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Now, let's give the parameters that are going to be used by our NN.

# In[ ]:

learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Now we need to create our encoder. For this, we are going to use sigmoidal functions. Sigmoidal functions continue to deliver great results with this type of networks. This is due to having a good derivative that is well-suited to backpropagation. We can create our encoder using the sigmoidal function like this:

# In[ ]:

# Building the encoder
def encoder(x):
    # Encoder first layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder second layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# And the decoder:
#
# You can see that the layer_1 in the encoder is the layer_2 in the decoder and vice-versa.

# In[ ]:

# Building the decoder
def decoder(x):
    # Decoder first layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder second layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# Let's construct our model.
# In the variable `cost` we have the loss function and in the `optimizer` variable we have our gradient used for backpropagation.

# In[ ]:

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


# The training will run for 20 epochs.

# In[ ]:

# Launch the graph
# Using InteractiveSession (more convenient while using Notebooks)
sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(mnist.train.num_examples/batch_size)
# Training cycle
for epoch in range(training_epochs):
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))

print("Optimization Finished!")


# Now, let's apply encode and decode for our tests.

# In[ ]:

# Applying encode and decode over test set
encode_decode = sess.run(
    y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})


# Let's simply visualize our graphs!

# In[ ]:

# Compare original images with their reconstructions

f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

plt.show()


# As you can see, the reconstructions were successful. It can be seen that some noise was added to the image.

# ### Thanks for completing this lesson!

# Authors:
#
# - <a href = "https://www.linkedin.com/in/franciscomagioli">Francisco Magioli</a>
# - <a href = "https://ca.linkedin.com/in/erich-natsubori-sato">Erich Natsubori Sato</a>
# - Gabriel Garcez Barros Souza

# ### References:
# - https://en.wikipedia.org/wiki/Autoencoder
# - http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
# - http://www.slideshare.net/billlangjun/simple-introduction-to-autoencoder
# - http://www.slideshare.net/danieljohnlewis/piotr-mirowski-review-autoencoders-deep-learning-ciuuk14
# - https://cs.stanford.edu/~quocle/tutorial2.pdf
# - https://gist.github.com/hussius/1534135a419bb0b957b9
# - http://www.deeplearningbook.org/contents/autoencoders.html
# - http://www.kdnuggets.com/2015/03/deep-learning-curse-dimensionality-autoencoders.html/
# - https://www.youtube.com/watch?v=xTU79Zs4XKY
# - http://www-personal.umich.edu/~jizhu/jizhu/wuke/Stone-AoS82.pdf
# - Reducing the Dimensionality of Data with Neural Networks, G. E. Hinton, R. R. Salakhutdinov, Science  28 Jul 2006, Vol. 313, Issue 5786, pp. 504-507, DOI: 10.1126/science.1127647 - http://science.sciencemag.org/content/313/5786/504.full
