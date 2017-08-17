import SparseAE as se
import FetchData as fd
import numpy as np
from keras.layers import Dense
from keras import regularizers
from keras.optimizers import RMSprop, Adam, SGD


first, second, third, original_data = fd.get_input_data('data/input_data.csv', False)
orig_csv_data = np.array(first)
domain_info = np.array(third)

first, second, third, fourth = fd.get_input_data('data/Supervised_data.csv', True)
csv_data = np.array(first)
output_data = np.array(second)

domain_test_input = []
domain_test_output = []
for i in range(len(domain_info)):
    domain = domain_info[i]
    domain_test_input.append(orig_csv_data[i])
    domain_test_output.append(domain)

layer_sizes = [100, 100, 50, 50]

encoders_list, decoders_list = se.get_encoders(layer_sizes, len(orig_csv_data[0]))

model = se.train(encoders_list, decoders_list,orig_csv_data[:-1000], orig_csv_data[:-1000], orig_csv_data[-1000:], orig_csv_data[-1000:],patience=2, num_epochs=100, layer_wise=True, optimizer=Adam(lr=0.001))

encoders_list.append(Dense(1, activation='relu', activity_regularizer=regularizers.l1(10e-8)))

model = se.train(encoders_list, decoders_list, csv_data[:1000], output_data[:1000], csv_data[1000:], output_data[1000:], num_epochs=200,patience=2, layer_wise=False, final_layer=True, optimizer=Adam(lr=0.001))

outputs = model.predict(orig_csv_data)

file_name = "data/output.csv"

thefile = open(file_name, 'w')
thefile.write("NODE_ID,DOMAIN,EASTING,NORTHING,LAT,LONGITUDE,RATING\n")
for i in range(len(outputs)):
  thefile.write("{},{},{}, {}, {},{},{}\n".format(domain_test_output[i][1], original_data[i]['DOMAIN'], original_data[i]['EASTING'], original_data[i]['NORTHING'], original_data[i]['LAT'], original_data[i]['LONGITUDE'], outputs[i][0]))