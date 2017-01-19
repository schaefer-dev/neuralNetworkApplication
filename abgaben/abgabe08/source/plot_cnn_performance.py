import matplotlib.pyplot as plt
import numpy as np


perf = [0.9421, 0.9665, 0.9732, .9744, .9789, .9811, .9807, .9831, .9844, .983, .9853, .9848, .9857, .9836, .9853]

plt.plot(perf);

plt.legend(['accuracy'], loc='upper left')
plt.show()
