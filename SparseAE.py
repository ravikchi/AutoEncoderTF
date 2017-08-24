from keras.layers import Dense, GaussianNoise
from keras.models import Sequential, load_model
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint


def get_encoders(layer_sizes, input_size):
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

        decoder = Dense(decoder_size, activation='relu')

        encoders.append(encoder)
        decoders.append(decoder)
        j -= 1

    return encoders, decoders


def train(encoders, decoders, x_train_loc, y_train_loc, x_test_loc, y_test_loc, num_epochs=20, layer_wise=False, final_layer=False, patience=2, optimizer=Adam(lr=0.001), model_chk_path='tmp\model_data', batchsize=256):
    mcp = ModelCheckpoint(model_chk_path, monitor="val_loss", save_best_only=True, save_weights_only=False)
    callbacks = [mcp, EarlyStopping(monitor='val_loss', patience=patience, verbose=0)]
    #callbacks = [mcp]

    if layer_wise:
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

            model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
            model.fit(x_train_loc, y_train_loc, epochs=num_epochs,
                      batch_size=batchsize,
                      shuffle=True,
                      validation_data=(x_test_loc, y_test_loc), callbacks=callbacks)
            load_model(model_chk_path)
    else:
        model = Sequential()
        for i in range(len(encoders)):
            encoder = encoders[i]
            encoder.trainable = True
            model.add(encoder)

        if not final_layer:
            for i in range(len(decoders)):
                decoder = decoders[i]
                encoder.trainable = True
                model.add(decoder)

        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        model.fit(x_train_loc, y_train_loc, epochs=num_epochs,
                  batch_size=batchsize,
                  shuffle=True,
                  validation_data=(x_test_loc, y_test_loc), callbacks=callbacks)
        load_model(model_chk_path)

    return model

