import random
import csv
import copy


def get_input_data(location, norm_ratings):
    with open(location) as csvfile:
        csv_data = list(csv.DictReader(csvfile))

    random.shuffle(csv_data)

    keyList = ['NODE_ID','DOMAIN','EASTING','NORTHING','LAT','LONGITUDE','EXCHNORTHDIST','EXCHSOUTHDIST','EXCHEASTDIST','EXCHWESTDIST','MEANEXCHNODEDIST','MEDIANNODEDIST','MEANRESOURCEDIST','MEDIANRESOURCEDIST','NO_RESOURCES','TOTAL_TASK','TOTAL_TASKO','TOTAL_TASK_TIME','TOTAL_TASK_TIMEO','RATING']

    original_data = copy.deepcopy(csv_data)

    for element in keyList:
        if element == 'DOMAIN' or element == 'NODE_ID' or element == 'EASTING' or element == 'NORTHING' or element == 'LAT' or element == 'LONGITUDE':
            continue

        if element == 'RATING' and not norm_ratings:
            continue
        values = set(float(data[element]) for data in csv_data)
        maximum = max(values)
        minimum = min(values)
        for data in csv_data:
            data[element] = (float(data[element]) - minimum) / (maximum - minimum)

    input_data = []
    output_data = []
    info_data = []
    for data in csv_data:
        element = []
        output = []
        info = [data['DOMAIN'], data['NODE_ID']]
        for key in keyList:
            if key == 'DOMAIN' or key == 'NODE_ID' or key == 'EASTING' or key == 'NORTHING' or key == 'LAT' or key == 'LONGITUDE':
                continue


            if key == 'RATING':
                output.append(data[key])
                continue

            element.append(data[key])

        input_data.append(element)
        output_data.append(output)
        info_data.append(info)

    return input_data, output_data, info_data, original_data