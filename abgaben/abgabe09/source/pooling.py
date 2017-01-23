from scipy import misc
import numpy as np

def readImage(path):
    return misc.imread(path)

def maxPool(data,split):
    res = np.zeros((split,split))
    foundmax = np.zeros((split,split))
    step = data.shape[0]/split
    w = 0
    for i in range(split):
        h = 0
        for j in range(split):
            res[i,j] = data[w:w+step,h:h+step].max()
            x = data[w:w+step,h:h+step].argmax()%step + w
            y = data[w:w+step,h:h+step].argmax()/step + h
            print(x,y)
            h += step
        w+=step

    return res 

def averagePool(data,split):
    res = np.zeros((split,split))
    foundmax = np.zeros((split,split))
    step = data.shape[0]/split
    w = 0
    for i in range(split):
        h = 0
        for j in range(split):
            res[i,j] = data[w:w+step,h:h+step].mean()
            h += step
        w+=step

    return res

im = readImage('clock.png')
clock_max = maxPool(im,128)
print(im,clock_max)
#misc.imsave('clockMax.png',clock_max)
clock_mean = averagePool(im,128)
print(im,clock_mean)
#misc.imsave('clockMean.png',clock_mean)