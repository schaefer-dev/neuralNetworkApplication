from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

def readImage(path):
    return misc.imread(path)

def conv(image, kernel):

    kernel_width = kernel.shape[0]
    kernel_height = kernel.shape[1]
    image_width = image.shape[0]
    image_height = image.shape[1]

    conv_image = np.zeros((image_width,image_height))

    for i in range(0,(image_width-1)-kernel_width):
        for j in range(0,(image_height-1)-kernel_height):
            for k in range(0,(kernel_width-1)):
                for l in range(0,(kernel_height-1)):
                    conv_image[i,j] += image[i+k,j+k]*kernel[k,l]


    return conv_image


def min_max_rescale(image):
    s = image.shape
    ix = s[0]
    iy = s[1]

    maxV = image.max()
    minV = image.min()

    print minV, maxV

    res = np.zeros((ix,iy))

    rangeV = maxV - minV

    for x in range(ix):
        for y in range(iy):
            res[x][y] = (image[x][y] - minV) / float(rangeV) * 255

    return res

def main():
   # ex 8.2 b)
   h = 1./9.
   k1=np.matrix([[h,h,h],[h,h,h],[h,h,h]])
   im_noise = readImage('clock_noise.png')
   im_conv = conv(im_noise,k1)
   plt.imshow(im_conv, cmap=plt.cm.gray)
   im_conv = min_max_rescale(im_conv)
   misc.imsave('im_conv.png',im_conv)

   # ex 8.2 c)
   k2=np.matrix([[0,0,0],[1,-2,1],[0,0,0]])
   im = readImage('clock.png')
   im2_conv = conv(im,k2)
   im2_conv = min_max_rescale(im2_conv)
   misc.imsave('im_conv2.png',im2_conv)
if __name__=="__main__":
    main()
