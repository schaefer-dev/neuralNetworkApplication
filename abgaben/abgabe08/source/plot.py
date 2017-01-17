import matplotlib.pyplot as plt
import numpy as np

f = [3,1,8,6,3,9,5,1]

s1 = [1.5,2,4.5,7,4.5,6,7,0.5] 
s2= [0.75,1.75,3.25,5.75,5.75,5.25,6.5,5,1.75,0.25]

plt.plot(f);
plt.plot(s1);
plt.plot(s2);

plt.legend(['f', 's1', 's2'], loc='upper left')
plt.show()
