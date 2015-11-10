import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt
 
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
 
y = np.loadtxt('logs/rewards/rewards_2015-11-10 01:42:55.800567.log')
x = np.arange(np.size(y))

yMA = movingaverage(y, 30000)

plt.title("Reward over time, exploration=0.25")
plt.xlabel('iteration') 
plt.ylabel('reward') 
plt.plot(x[len(x)-len(yMA):],yMA)
plt.show()