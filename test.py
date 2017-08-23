import SparseAE as se
import FetchData as fd
import numpy as np
from keras.layers import Dense
from keras import regularizers
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers import Dense, GaussianNoise
from keras.models import Sequential, load_model


first, second, third, original_data = fd.get_input_data('data/input_data.csv', False)
orig_csv_data = np.array(first)
domain_info = np.array(third)

first, second, third, fourth = fd.get_input_data('data/Supervised_data.csv', False)
csv_data = np.array(first)
output_data = np.array(second)

domain_test_input = []
domain_test_output = []
for i in range(len(domain_info)):
    domain = domain_info[i]
    domain_test_input.append(orig_csv_data[i])
    domain_test_output.append(domain)

model = Sequential()
model.add(Dense(50, activation='relu', activity_regularizer=regularizers.l1(10e-8), input_dim=len(csv_data[0])))
model.add(Dense(1, activation='relu', activity_regularizer=regularizers.l1(10e-8)))

model.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['accuracy'])
model.fit(csv_data[:1000], output_data[:1000], epochs=4000,
          batch_size=100,
          shuffle=True,
          validation_data=(csv_data[1000:], output_data[1000:]))

