import gmplot
import csv
from math import sin, cos, sqrt, atan2, radians


def getDistance(lat1, long1, lat2, long2):
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(long1)
    lat2 = radians(lat2)
    lon2 = radians(long2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

with open("data/output_neighbour_model.csv") as csvfile:
    csv_data = list(csv.DictReader(csvfile))

latitudes = []
longitudes = []

for data in csv_data:
    latitudes.append(float(data['LAT']))
    longitudes.append(float(data['LONGITUDE']))

minDist = 100000000000000000000
finalI = 0


# for i in range(len(latitudes)):
#     lati = 0.0
#     for j in range(len(latitudes)):
#         lati = lati + getDistance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
#
#     if lati < minDist :
#         minDist = lati
#         finalI = i

#gmap = gmplot.GoogleMapPlotter(latitudes[finalI], longitudes[finalI], 10)

good_lat_list = []
good_long_list = []
latitudes = []
longitudes = []

for i in range(len(csv_data)):
    data = csv_data[i]
    if data['DOMAIN'] == 'Peterborough':
        latitude = float(data['LAT'])
        longitude = float(data['LONGITUDE'])
        if float(data['RATING']) > 0.8:
            good_lat_list.append(latitude)
            good_long_list.append(longitude)
        else:
            latitudes.append(latitude)
            longitudes.append(longitude)

gmap = gmplot.GoogleMapPlotter(latitudes[finalI], longitudes[finalI], 10)
gmap.scatter(latitudes, longitudes, '#000000', size=500, marker=False)
gmap.scatter(good_lat_list, good_long_list, '#F00000', size=500, marker=False)



gmap.draw("maps/Peterborough.html")