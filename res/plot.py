import matplotlib.pyplot as plt
import numpy as np
import sys

mode = sys.argv[-2]
filename = sys.argv[-1]
stats = np.load(filename)

n,m = stats.shape
stats = np.delete(stats, n-1, 0)

if mode == "-l":
    learning_curve = []
    s = 0
    for i in range(n-1):
        s += stats[i, 0]
        learning_curve.append(s)

    line, = plt.plot(stats[:, 0], label="Learning Curve")
    plt.legend(handles=[line])
    plt.xlabel("Epochs (in hundreds)")
else:
    line1, = plt.plot(stats[:, 0], label="Reward")
    line2, = plt.plot(stats[:, 1], label="Number of Steps per Epoch")
    plt.legend(handles=[line1, line2])
    plt.xlabel("Epochs (in hundreds)")

plt.show()