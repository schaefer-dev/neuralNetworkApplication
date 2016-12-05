import matplotlib.pyplot as plt
import random as random
import math as math


############### Parameters BEGIN ###################

k = 2

# spectate = True if you want to see the movement of the clusters in single steps!
spectate = False

# enable plots for exercise c (only works for k = 2)
ex41c = False


############### Parameters END ###################


def plotSet(set, color):
    for i in range(0, len(set)):
        plt.plot(set[i][0], set[i][1], color)


############### Reading of data points ################

# Array used to store data
data = []
with open('data_kmeans.txt') as f:
    for line in f: # read lines
        data.append([float(x) for x in line.split()])


# initial Plotting of dataSet
plotSet(data, 'go')


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


# generate random clusters
for i in range(k):
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    clusters.append([x, y])
    print("generated random cluster " + str(i+1) + " at (" + str(x) + ", " + str(y) + ")")

# show result after generating random clusters
if (spectate):
    plotSet(clusters, 'ro')
    print("--------------------- close Plot to continue computation ---------------------")
    plt.show()

# variable to interupt clustering process
clustering = True
iteration = 0
J = 0.0
while (clustering):

    iteration += 1
    # maxdist = math.hypot(max_x - min_x, max_y - min_y)
    maxdist = 9999999999999
    noCluster = len(clusters)
    clusterpoints = []

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

    if (ex41c):
        plotSet(clusterpoints[0], 'go')
        plotSet(clusterpoints[1], 'yo')
        plotSet(clusters, 'ro')
        print("--------------------- close Plot to continue computation ---------------------")
        plt.show()

    # Recalculate cluster centers
    oldclusters = clusters
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
        print("Step " + str(iteration) + ": cluster " + str(i+1) + " moved to (" + str(cluster_x) + ", " + str(cluster_y) + ")")


    # Recalculate J Objective:
    oldJ = J
    J = 0
    for i in range(0,k):
        for x in data:
            J += (x[0] - clusters[i][0])**2 + (x[1] - clusters[i][1])**2


    if (spectate):
        plotSet(data, 'go')
        plotSet(clusters, 'ro')
        print("--------------------- close Plot to continue computation ---------------------")
        plt.show()



    # Termination conditions
    sameClusters = True
    for i in range(1,k):
        # if there is a cluster which moved we do NOT terminate already!
        if (clusters[i] != oldclusters[i]):
            sameClusters = False
    if (sameClusters):
        print("Termination because the clusters did not change!")
        clustering = False

    # IF J doesnt change we terminate
    if (oldJ == J):
        print("Termination because the J Objective did not change!")
        clustering = False
