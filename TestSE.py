import matplotlib.pyplot as plt
import numpy as np
import SparseAE as se
from keras import regularizers
from keras.layers import Dense
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

layer_sizes = [1024, 512, 256]

encoders_list, decoders_list = se.get_encoders(layer_sizes, 784)

model = se.train(encoders_list, decoders_list,x_train, x_train, x_test, x_test, num_epochs=20, patience=10, layer_wise=True)

decoded_imgs = model.predict(x_test)

encoders_list.append(Dense(1, activation='relu', activity_regularizer=regularizers.l1(10e-8)))

model = se.train(encoders_list, decoders_list, x_train, y_train, x_test, y_test, num_epochs=20,patience=10, layer_wise=False, final_layer=True)

metrics = model.evaluate(x_test, y_test)

print()

for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()