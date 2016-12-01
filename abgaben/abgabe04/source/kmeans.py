
# Array used to store data
array = []


# Reading of data points
with open('data_kmeans.txt') as f:
    for line in f: # read lines
        array.append([float(x) for x in line.split()])


# Plotting of array

