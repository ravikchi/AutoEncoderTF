import tensorflow as tf


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