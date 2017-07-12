import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class RMSCost:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        self.cost = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))


class Data:
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels
        self.index = 0
        self.size = len(input)

    def inp_size(self):
        return len(self.input[0])

    def next_batch(self, batchSize):
        cur_index = self.index
        if self.index == len(self.input):
            self.index = 0
            cur_index = self.index

        if (self.index + batchSize) > len(self.input):
            self.index = len(self.input)
        else:
            self.index += batchSize

        return self.input[cur_index:self.index], self.labels[cur_index:self.index]


class Denoising(Data):
    def __init__(self, input, labels, corruption_ratio):
        Data.__init__(self, input, labels)
        self.corrfac = corruption_ratio

    def salt_and_pepper_noise(self, X, v):
        if(np.random.random() > self.corrfac):
            return X

        X_noise = X.copy()
        n_features = X.shape[1]

        mn = X.min()
        mx = X.max()

        for i, sample in enumerate(X):
            mask = np.random.randint(0, n_features, v)

            for m in mask:

                if np.random.random() < 0.5:
                    X_noise[i][m] = mn
                else:
                    X_noise[i][m] = mx

        return X_noise

    def next_batch(self, batchSize):
        cur_index = self.index
        if self.index == len(self.input):
            self.index = 0
            cur_index = self.index

        if (self.index + batchSize) > len(self.input):
            self.index = len(self.input)
        else:
            self.index += batchSize

        corruption_ratio = np.round(self.corrfac * input_data.shape[1]).astype(np.int)

        output = self.salt_and_pepper_noise(self.input[cur_index:self.index], corruption_ratio)

        return output, self.labels[cur_index:self.index]


class AutoEncoder:
    def __init__(self, id, input_size, hidden_size, act_func, inputX=None, sess=None, previous=None,
                 learning_rate=0.01, supervised=False, previous_graph=None):
        self.id = id

        self.weight = tf.Variable(tf.random_normal([input_size, hidden_size]), name="weight_" + str(id))
        self.bias = tf.Variable(tf.random_normal([hidden_size]), name="bias_" + str(id))

        self.act_func = act_func

        if inputX is not None:
            self.inputX = inputX
        else:
            self.inputX = tf.placeholder('float', [None, input_size])

        self.outputX = tf.placeholder('float', [None, hidden_size])

        if previous:
            self.previous = previous
        else:
            self.previous = None

        if previous_graph is not None:
            self.encoder = self.act_func(tf.add(tf.matmul(previous_graph, self.weight), self.bias))
        else:
            self.encoder = self.act_func(tf.add(tf.matmul(self.inputX, self.weight), self.bias))

        self.sess = sess

        self.weight_d = tf.transpose(self.weight)
        self.bias_d = tf.Variable(tf.random_normal([input_size]), name="bias_d_" + str(id))

        self.decoder = act_func(tf.add(tf.matmul(self.encoder, self.weight_d), self.bias_d))

        self.supervised = supervised

        if self.supervised :
            self.cost = tf.reduce_mean(tf.pow(self.outputX - self.encoder, 2))
        else:
            self.cost = tf.reduce_mean(tf.pow(self.inputX - self.decoder, 2))

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)

    def output(self, i_data, output_data):
        if self.previous:
            i_data, output_data = self.previous.output(i_data, output_data)

        return self.sess.run(self.encoder, feed_dict={self.inputX: i_data}), output_data

    def train(self, data, num_of_epoch=2, batch_size=256):

        total_batches = int(data.size / batch_size)

        for epoch in range(num_of_epoch):
            for i in range(total_batches):
                batch_xs, batch_ys = data.next_batch(batch_size)
                if self.previous:
                    batch_xs, batch_ys = self.previous.output(batch_xs, batch_ys)

                if self.supervised :
                    _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.inputX: batch_xs, self.outputX:batch_ys})
                else:
                    _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.inputX: batch_xs})

            if epoch % 10:
                print(epoch)
                print(c)

        print("Finished "+str(self.id))


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

def finalLayer(layers):
    input = layers[0].inputX
    for layer in layers:
        input = layer.act_func(tf.add(tf.matmul(input, layer.weight), layer.bias))

    return input, layers[0].inputX


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

examples_to_show = 10

input_data = Data(mnist.train.images, mnist.train.labels)
layers = []
sizes = [256,256,128,64]

with tf.Session() as sess:
    input_size = input_data.inp_size()

    for i in range(len(sizes)):
        size = sizes[i]
        if len(layers) == 0:
            layers.append(AutoEncoder(i, input_size, size, tf.nn.sigmoid, sess=sess))
        else:
            layers.append(AutoEncoder(i, input_size, size, tf.nn.sigmoid, sess=sess, previous=layers[-1]))

        input_size = size

    encoder_pt, inputX = finalLayer(layers)

    layers.append(AutoEncoder(len(sizes), input_size, 10, tf.nn.sigmoid, inputX=inputX, sess=sess, supervised=True, previous_graph=encoder_pt))


    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for layer in layers:
        layer.train(input_data, num_of_epoch=6)

    saver.save(sess, "/tmp/my_model")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph("/tmp/my_model.meta")
    saver.restore(sess, tf.train.latest_checkpoint('/tmp/'))

    decoder, inputX = mergeLayers(len(sizes), input_data.inp_size())

    encode_decode = sess.run(decoder, feed_dict={inputX: mnist.test.images[:examples_to_show]})

    encoder, inputX = get_encoder(len(sizes)+1, input_data.inp_size())
    outputX = tf.placeholder('float', [None, 10])

    correct_prediction = tf.equal(tf.argmax(encoder, 1), tf.argmax(outputX, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={inputX: mnist.test.images, outputX: mnist.test.labels}))

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

    plt.show()
