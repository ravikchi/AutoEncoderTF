import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Activation
from keras.models import Model, Sequential
from keras import regularizers

from keras.datasets import mnist


def getEncoders(layer_sizes, input_size):
    encoders = []
    decoders = []

    j = len(layer_sizes) -2

    for i in range(len(layer_sizes)):
        size = layer_sizes[i]
        if j < 0 :
            decoder_size = input_size
        else:
            decoder_size = layer_sizes[j]

        if i == 0:
            encoder = Dense(size, activation='relu', activity_regularizer=regularizers.l1(10e-8), input_dim=input_size)
        else:
            encoder = Dense(size, activation='relu', activity_regularizer=regularizers.l1(10e-8))

        decoder = Dense(decoder_size, activation='relu', activity_regularizer=regularizers.l1(10e-8))

        encoders.append(encoder)
        decoders.append(decoder)
        j -= 1

    return encoders, decoders


def train_layer_wise(encoders, decoders, x_train_local, y_train_local, x_test_local, y_test_local, num_epochs=20, batch_sz=256):
    decoders.reverse()
    for i in range(len(encoders)):
        model = Sequential()
        local_encoders = encoders[:i+1]
        local_decoders = decoders[:i+1]
        local_decoders.reverse()
        for k in range(len(encoders)):
            j = k + 1
            if k > i:
                break

            if k < i:
                encoders[k].trainable = False

            encoder = local_encoders[k]

            model.add(encoder)

        for k in range(len(encoders)):
            j = k + 1
            if k > i:
                break

            if k < i:
                encoders[k].trainable = False

            decoder = local_decoders[k]

            model.add(decoder)

        model.compile(optimizer='adam', loss='mse')
        model.fit(x_train_local, y_train_local, epochs=num_epochs,
                  batch_size=batch_sz,
                  shuffle=True,
                  validation_data=(x_test_local, y_test_local))

    decoders.reverse()
    return model


def train_all_layers(encoders, x_train_local, y_train_local, x_test_local, y_test_local, num_epochs=20, batch_sz=256):
    model = Sequential()
    for i in range(len(encoders)):
        encoders[i].trainable = True
        model.add(encoders[i])

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(x_train_local, y_train_local, epochs=num_epochs,
              batch_size=batch_sz,
              shuffle=True,
              validation_data=(x_test_local, y_test_local))

    return model


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

layer_sizes = [1024, 512, 256]

encoder_list, decoder_list = getEncoders(layer_sizes, 784)

trained_model = train_layer_wise(encoder_list, decoder_list, x_train, x_train, x_test, x_test, num_epochs=20)

decoded_imgs = trained_model.predict(x_test)

encoder_list.append(Dense(1, activation='relu'))
decoder_list.append(Dense(layer_sizes[-1], activation='relu'))

greedy_model = train_all_layers(encoder_list, x_train, y_train, x_test, y_test, num_epochs=20)

metrics = greedy_model.evaluate(x_test, y_test)

print("\n")

for i in range(len(greedy_model.metrics_names)):
    print(str(greedy_model.metrics_names[i]) + ": " + str(metrics[i]))

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