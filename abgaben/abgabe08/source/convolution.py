from scipy import misc
import matplotlib.pyplot as plt
import numpy as np


def readImage(path):
    return misc.imread(path)


def convfix(image,kernel):

   kernel_width = kernel.shape[0]
   kernel_height = kernel.shape[1]
   image_width = image.shape[0]
   image_height = image.shape[1]

   width_shift = kernel_width // 2
   height_shift = kernel_height // 2

   padded_image = np.zeros((image_width + width_shift, image_height + height_shift))

   # image padded with as many zeros as necessary
   for x in range (0,image_width-1):
      for y in range(0,image_height-1):
         padded_image[x + width_shift, y + height_shift] = image[x,y]

   conv_image = np.zeros((image_width,image_height))

   # convolution process
   for x in range(0,(image_width - 1)):
       for y in range(0,(image_height - 1)):
           for a in range(-width_shift, width_shift):
               for b in range(-height_shift, height_shift):
                   conv_image[x,y] += padded_image[x + width_shift + a, y + height_shift + b] * kernel[a + width_shift ,b + height_shift]


   return conv_image



def conv(image, kernel):
    # ex 8.2 a)

    kernel_width = kernel.shape[0]
    kernel_height = kernel.shape[1]
    image_width = image.shape[0]
    image_height = image.shape[1]

    conv_image = np.zeros((image_width,image_height))

    for i in range(0,(image_width-1)- kernel_width // 2):
        for j in range(0,(image_height-1)- kernel_height // 2):
            for k in range(0,(kernel_width-1)):
                for l in range(0,(kernel_height-1)):
                    conv_image[i,j] += image[i+k,j+l]*kernel[k,l]


    return conv_image


def min_max_rescale(image):
    s = image.shape
    ix = s[0]
    iy = s[1]

    maxV = image.max()
    minV = image.min()

    print (minV, maxV)

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
   im_conv = convfix(im_noise,k1)
   plt.imshow(im_conv, cmap=plt.cm.gray)
   im_conv = min_max_rescale(im_conv)
   plt.imshow(im_conv, cmap=plt.cm.gray)
   plt.show()
   misc.imsave('im_conv_b.png',im_conv)

   # ex 8.2 c)
   k2=np.matrix([[0.0,0.0,0.0],[1.0,-2.0,1.0],[0.0,0.0,0.0]])
   im2 = readImage('clock.png')
   im2_conv = convfix(im2,k2)
   im2_conv = min_max_rescale(im2_conv)
   plt.imshow(im2_conv, cmap=plt.cm.gray)
   plt.show()
   misc.imsave('im_conv_c.png',im2_conv)

if __name__=="__main__":
    main()
