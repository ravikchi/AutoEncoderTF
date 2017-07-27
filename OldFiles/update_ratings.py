import csv
import os

allData = []

for file in os.listdir("ratings"):
    if file == 'AllRatings.csv':
        continue
    filename = "ratings/"+file
    data = []
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        count = 1
        for row in readCSV:
            if count > 1:
                element = [row[0], float(row[1]), 0]
                data.append(element)
            count = count + 1

    data.sort(key=lambda x:x[1], reverse=True)

    total_batches = int(round(len(data)/10))

    count = 1
    val = 10
    for row in data:
        if count % total_batches == 0:
            val = val -1
        row[2] = val

        allData.append(row)

        count = count + 1

    thefile = open(filename, 'w')
    thefile.write('NODE_ID,RATING,RANK\n')
    for item in data:
      thefile.write('{},{},{}\n'.format(item[0],item[1],item[2]))


thefile = open('ratings/AllRatings.csv', 'w')
thefile.write('NODE_ID,RATING,RANK\n')
for item in allData:
    thefile.write('{},{},{}\n'.format(item[0],item[1],item[2]))
