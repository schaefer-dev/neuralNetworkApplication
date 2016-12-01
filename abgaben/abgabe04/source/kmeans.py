import matplotlib.pyplot as plt
import random as random
import math as math


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
max_x = 0.0
min_x = 0.0
max_y = 0.0
min_y = 0.0
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
    print("generated random cluster " + str(i+1) + " at (" + str(x) + ", " + str(y) + ")")


iteration = 0
while (clustering):

    iteration += 1
    maxdist = math.hypot(max_x - min_x, max_y - min_y)
    noCluster = len(clusters)
    clustermatches = []

    # clear/initiate the lists which match a point to the closest cluster
    for i in range(k):
        clusterpoints.append([])

    # Assign each data point to its nearest center
    for point in data:
        dist = maxdist
        clustermatch = noCluster

        # search for the cluster which has the smallest distance to this point
        # and match this point to this cluster
        clusteriter = 0
        for cluster in clusters:
            if dist > math.hypot(point[0] - cluster[0], point[1] - cluster[1]):
                dist = math.hypot(point[0] - cluster[0], point[1] - cluster[1])
                clustermatch = clusteriter
            clusteriter += 1
        clusterpoints[clustermatch].append(point)

    # Recalculate cluster centers
    clusters = []
    for i in range(k):
        sum_x = 0.0
        sum_y = 0.0
        size = float(len(clusterpoints[i]))
        for point in clusterpoints[i]:
            sum_x += point[0]
            sum_y += point[1]
        if (size == 0):
            cluster_x = 0
            cluster_y = 0
        else:
            cluster_x = sum_x / size
            cluster_y = sum_y / size
        clusters.append([cluster_x, cluster_y])
        print("cluster " + str(i+1) + " moved to (" + str(cluster_x) + ", " + str(cluster_y) + ")")

    # TODO abbruchbedingung implementieren
    if (iteration == 10):
        clustering = False

plt.show()