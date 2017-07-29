import tensorflow as tf
import numpy as np
from AutoEncoder import AutoEncoder


class Denoising(AutoEncoder):
    def __init__(self, id, input_size, hidden_size, act_func, inputX=None, sess=None, previous=None, learning_rate=0.01, supervised=False, previous_graph=None, corrfac=0.5):
        AutoEncoder.__init__(self, id, input_size, hidden_size, act_func, inputX, sess, previous, learning_rate, supervised, previous_graph)

        self.corrfac = corrfac

        if not supervised:
            self.outputX = self.inputX
            self.cost = tf.reduce_mean(tf.pow(self.outputX - self.decoder, 2))
        else:
            self.cost = tf.reduce_mean(tf.pow(self.outputX - self.encoder, 2))


        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)

    def salt_and_pepper_noise(self, X, v):
        if np.random.random() > self.corrfac:
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

    def train(self, data, num_of_epoch=2, batch_size=256):

        total_batches = int(data.size / batch_size)

        corruption_ratio = np.round(self.corrfac * data.input.shape[1]).astype(np.int)

        for epoch in range(num_of_epoch):
            cost = 0.0
            for i in range(total_batches):
                batch_xs, batch_ys = data.next_batch(batch_size)
                if self.previous:
                    batch_xs, batch_ys = self.previous.output(batch_xs, batch_ys)

                corrupted_xs = self.salt_and_pepper_noise(batch_xs, corruption_ratio)

                if self.supervised:
                    _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.inputX: batch_xs, self.outputX:batch_ys})
                else:
                    _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.inputX: corrupted_xs, self.outputX:batch_xs})

                cost = cost + c

            cost = cost/total_batches
            if epoch % 100 == 0:
                print(epoch)
                print(cost)

            if cost < 0.0000001:
                break

        print("Finished "+str(self.id))