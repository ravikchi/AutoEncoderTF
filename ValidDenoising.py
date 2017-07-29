import tensorflow as tf
import numpy as np
from DenoisingAE import Denoising

class ValidDenoising(Denoising):
    def __init__(self, id, input_size, hidden_size, act_func, inputX=None, sess=None, previous=None, learning_rate=0.01, supervised=False, previous_graph=None, corrfac=0.5):
        Denoising.__init__(self, id, input_size, hidden_size, act_func, inputX, sess, previous, learning_rate, supervised, previous_graph, corrfac)

        self.validation_weight = tf.identity(self.weight)
        self.validation_bias = tf.identity(self.bias)

        self.validation_bias_d = tf.identity(self.bias_d)

    def train(self, data, num_of_epoch=2, batch_size=256):

        total_batches = int(data.size / batch_size)

        corruption_ratio = np.round(self.corrfac * data.input.shape[1]).astype(np.int)

        old_val_c = 1000000
        val_count = 0

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

            batch_xs = data.validInput
            batch_ys = data.validLabels
            if self.previous:
                batch_xs, batch_ys = self.previous.output(data.validInput, data.validLabels)

            if self.supervised:
                _, val_c = self.sess.run([self.optimizer, self.cost],feed_dict={self.inputX: batch_xs, self.outputX: batch_ys})
            else:
                _, val_c = self.sess.run([self.optimizer, self.cost], feed_dict={self.inputX: batch_xs, self.outputX:batch_xs})

            if val_c < old_val_c:
                self.validation_weight = tf.identity(self.weight)
                self.validation_bias = tf.identity(self.bias)

                self.validation_bias_d = tf.identity(self.bias_d)

                old_val_c = val_c

                val_count = 0
            elif val_count > 100 :
                break
            else :
                val_count = val_count + 1

            if cost < 0.000001:
                break

        self.weight = self.validation_weight
        self.bias = self.validation_bias
        self.bias_d = self.validation_bias_d
        print("Finished "+str(self.id))