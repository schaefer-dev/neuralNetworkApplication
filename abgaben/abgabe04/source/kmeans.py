import matplotlib.pyplot as plt
import random as random


############### Parameters ############################

plotDataSet = True
k = 2


############### Reading of data points ################

# Array used to store data
data = []
with open('data_kmeans.txt') as f:
    for line in f: # read lines
        data.append([float(x) for x in line.split()])


# Plotting of array
if (plotDataSet):
    for i in range(0,len(data)):
        plt.plot(data[i][0],data[i][1], 'ro')


# Compute maximum and minimum x and y
max_x = 0
min_x = 0
max_y = 0
min_y = 0
for entry in data:
    if entry[0] > max_x:
        max_x = entry[0]
    elif entry[0] < min_x:
        min_x = entry[0]
    if entry[1] > max_y:
        max_y = entry[1]
    elif entry[1] < min_y:
        min_y = entry[1]


########### k-Means Clustering #############

# contains the cluster-centers
clusters = []

# list which matches point to a certain cluster
clusterpoints = []

# variable to interupt clustering process
clustering = True

# generate random clusters
for i in range(k):
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    clusters.append([x, y])
    print("generated random cluster at (" + str(x) + ", " + str(y) + ")")


while (clustering):
    # TODO impelemnt Clustering here like in Slides

    clustering = False

plt.show()