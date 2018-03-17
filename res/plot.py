import matplotlib.pyplot as plt
import numpy as np
import sys

filename = sys.argv[-1]
stats = np.load(filename)

n,m = stats.shape
stats = np.delete(stats, n-1, 0)

line1, = plt.plot(stats[:, 0], label="Reward")
line2, = plt.plot(stats[:, 1], label="Time")
plt.legend(handles=[line1, line2])
plt.xlabel("Epochs (in hundreds)")

plt.show()