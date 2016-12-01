import matplotlib.pyplot as plt

# Parameters
plotting = True

# Array used to store data
array = []


# Reading of data points
with open('data_kmeans.txt') as f:
    for line in f: # read lines
        array.append([float(x) for x in line.split()])


# Plotting of array
if (plotting):
    for i in range(0,len(array)):
        plt.plot(array[i][0],array[i][1], 'ro')
    plt.show()
